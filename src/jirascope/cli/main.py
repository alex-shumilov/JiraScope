"""Main CLI entry point for JiraScope."""

import asyncio
from pathlib import Path

import click

from ..analysis.content_analyzer import ContentAnalyzer
from ..analysis.cross_epic_analyzer import CrossEpicAnalyzer
from ..analysis.similarity_analyzer import SimilarityAnalyzer
from ..analysis.structural_analyzer import StructuralAnalyzer
from ..analysis.template_inference import TemplateInferenceEngine
from ..clients.lmstudio_client import LMStudioClient
from ..clients.qdrant_client import QdrantVectorClient
from ..core.config import Config
from ..extractors.jira_extractor import JiraExtractor
from ..pipeline.embedding_processor import EmbeddingProcessor
from ..pipeline.incremental_processor import IncrementalProcessor
from ..pipeline.quality_validator import EmbeddingQualityValidator
from ..utils.logging import setup_logging


@click.group()
@click.option(
    "--config", "-c", type=click.Path(exists=True, path_type=Path), help="Configuration file path"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-file", type=click.Path(path_type=Path), help="Log file path")
@click.pass_context
def cli(ctx, config, verbose, log_file):
    """JiraScope - AI-powered Jira work item analysis and management tool."""
    ctx.ensure_object(dict)

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    cost_tracker = setup_logging(log_level=log_level, log_file=log_file)

    # Load configuration
    try:
        app_config = Config.load(config)
        ctx.obj["config"] = app_config
        ctx.obj["cost_tracker"] = cost_tracker

        click.echo(f"JiraScope initialized with config from: {config or 'environment/defaults'}")

    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.pass_context
def health(ctx):
    """Check health of all connected services."""
    config = ctx.obj["config"]
    click.echo("Checking service health...")

    async def check_services():
        results = {}

        # Check Qdrant
        try:
            async with QdrantVectorClient(config) as qdrant:
                results["qdrant"] = await qdrant.health_check()
        except Exception as e:
            results["qdrant"] = False
            click.echo(f"Qdrant error: {e}")

        # Check LMStudio
        try:
            async with LMStudioClient(config) as lm:
                results["lmstudio"] = await lm.health_check()
        except Exception as e:
            results["lmstudio"] = False
            click.echo(f"LMStudio error: {e}")

        return results

    results = asyncio.run(check_services())

    click.echo(f"Qdrant: {'✓' if results.get('qdrant') else '✗'}")
    click.echo(f"LMStudio: {'✓' if results.get('lmstudio') else '✗'}")

    if all(results.values()):
        click.echo("All services healthy!")
    else:
        click.echo("Some services are not responding")
        ctx.exit(1)


@cli.command()
@click.option("--project", "-p", required=True, help="Jira project key (e.g., PROJ)")
@click.option("--incremental", "-i", is_flag=True, help="Perform incremental sync only")
@click.option("--jql", default="", help="Additional JQL query to filter work items")
@click.pass_context
def fetch(ctx, project, incremental, jql):
    """Fetch work items from Jira and generate embeddings."""
    config = ctx.obj["config"]

    async def fetch_and_process():
        extractor = JiraExtractor(config)
        processor = EmbeddingProcessor(config)
        incremental_proc = IncrementalProcessor(config)

        do_incremental = incremental

        if do_incremental:
            click.echo(f"Performing incremental sync for project {project}...")

            # Get last sync timestamp
            last_sync = incremental_proc.get_last_sync_timestamp(project)
            if not last_sync:
                click.echo("No previous sync found, performing full sync instead")
                do_incremental = False
            else:
                click.echo(f"Last sync: {last_sync}")

                # Get tracked items for incremental query
                tracked_epics = incremental_proc.get_tracked_epics(project)
                tracked_items = incremental_proc.get_tracked_items(project)

                # Get updated items
                updated_items = await extractor.get_incremental_updates(
                    project, last_sync, tracked_epics, tracked_items
                )

                if not updated_items:
                    click.echo("No updates found since last sync")
                    return

                click.echo(f"Found {len(updated_items)} updated items")

                # Process incremental updates
                result = await incremental_proc.process_incremental_updates([], updated_items)

                click.echo(
                    f"Processed: {result.processed_items}, Skipped: {result.skipped_items}, Failed: {result.failed_items}"
                )

                # Update sync timestamp
                incremental_proc.update_last_sync_timestamp(project)

        if not do_incremental:
            click.echo(f"Fetching all active hierarchies for project {project}...")

            # Full extraction
            hierarchies = await extractor.extract_active_hierarchies(project)
            click.echo(f"Found {len(hierarchies)} epic hierarchies")

            # Collect all work items
            all_items = []
            for hierarchy in hierarchies:
                all_items.extend(hierarchy.all_items)

            click.echo(f"Total work items: {len(all_items)}")

            if all_items:
                # Process embeddings
                result = await processor.process_work_items(all_items)
                click.echo(
                    f"Embedding results: {result.processed_items} processed, {result.failed_items} failed"
                )

                # Update tracking
                incremental_proc = IncrementalProcessor(config)
                await incremental_proc.process_incremental_updates(all_items, [])
                incremental_proc.update_last_sync_timestamp(project)

        # Show cost summary
        cost = extractor.calculate_extraction_cost()
        click.echo(f"\nExtraction cost: {cost.api_calls} API calls, ${cost.estimated_cost:.4f}")

    asyncio.run(fetch_and_process())


@cli.command()
@click.option("--query", "-q", required=True, help="Search query text")
@click.option("--limit", "-l", default=5, help="Number of results to return")
@click.pass_context
def search(ctx, query, limit):
    """Search for similar work items using semantic search."""
    config = ctx.obj["config"]

    async def search_items():
        async with LMStudioClient(config) as lm_client:
            async with QdrantVectorClient(config) as qdrant_client:
                # Generate query embedding
                embeddings = await lm_client.generate_embeddings([query])
                if not embeddings:
                    click.echo("Failed to generate embedding for query")
                    return

                # Search for similar items
                results = await qdrant_client.search_similar_work_items(embeddings[0], limit=limit)

                if not results:
                    click.echo("No similar work items found")
                    return

                click.echo(f"\nFound {len(results)} similar work items for: '{query}'\n")

                for i, result in enumerate(results, 1):
                    work_item = result["work_item"]
                    score = result["score"]

                    click.echo(f"{i}. {work_item['key']}: {work_item['summary']}")
                    click.echo(
                        f"   Score: {score:.3f} | Type: {work_item['issue_type']} | Status: {work_item['status']}"
                    )
                    if work_item.get("description"):
                        desc = (
                            work_item["description"][:100] + "..."
                            if len(work_item["description"]) > 100
                            else work_item["description"]
                        )
                        click.echo(f"   Description: {desc}")
                    click.echo()

    asyncio.run(search_items())


@cli.command()
@click.pass_context
def validate(ctx):
    """Validate embedding quality with test queries."""
    config = ctx.obj["config"]

    async def run_validation():
        validator = EmbeddingQualityValidator(config)

        click.echo("Running embedding quality validation...")
        report = await validator.validate_embedding_quality()

        click.echo("\nQuality Report:")
        click.echo(f"Overall Score: {report.overall_score:.1f}%")
        click.echo(f"Tests Passed: {report.passed_tests}/{report.total_tests}")

        if report.recommendations:
            click.echo("\nRecommendations:")
            for rec in report.recommendations:
                click.echo(f"  • {rec}")

        if report.results:
            click.echo("\nDetailed Results:")
            for result in report.results[:5]:  # Show first 5
                status = "✓" if result["passed"] else "✗"
                click.echo(
                    f"  {status} {result['query']} (score: {result.get('avg_similarity', 0):.3f})"
                )

    asyncio.run(run_validation())


@cli.command()
@click.pass_context
def cost(ctx):
    """Show cost summary and cache statistics."""
    config = ctx.obj["config"]
    cost_tracker = ctx.obj.get("cost_tracker")

    if cost_tracker:
        summary = cost_tracker.get_session_summary()
        click.echo("\nSession Cost Summary:")
        click.echo(f"Total Cost: ${summary['total_cost']:.4f}")
        click.echo(f"Duration: {summary['session_duration_seconds']:.1f}s")
        click.echo(f"Operations: {summary['operation_count']}")

        if summary["costs_by_service"]:
            click.echo("\nCosts by Service:")
            for service, cost in summary["costs_by_service"].items():
                click.echo(f"  {service}: ${cost:.4f}")
    else:
        click.echo("Cost tracking not enabled.")

    # Show cache statistics
    incremental_proc = IncrementalProcessor(config)
    cache_stats = incremental_proc.get_cache_statistics()

    click.echo("\nCache Statistics:")
    click.echo(f"Tracked Items: {cache_stats.get('tracked_items_count', 0)}")
    click.echo(f"Projects: {cache_stats.get('projects_tracked', 0)}")
    click.echo(f"Cache Size: {cache_stats.get('total_cache_size_bytes', 0)} bytes")


@cli.command()
@click.option("--days", default=30, help="Clean cache files older than N days")
@click.pass_context
def cleanup(ctx, days):
    """Clean up old cache files."""
    config = ctx.obj["config"]

    incremental_proc = IncrementalProcessor(config)
    incremental_proc.cleanup_old_cache(days)

    click.echo(f"Cleaned cache files older than {days} days")


# Analysis commands
@cli.group()
def analyze():
    """Analysis commands for work items."""
    pass


@analyze.command("duplicates")
@click.option("--project", "-p", help="Project key to analyze")
@click.option("--threshold", default=0.70, help="Similarity threshold (0.0-1.0)")
@click.pass_context
def analyze_duplicates(ctx, project, threshold):
    """Find potential duplicate work items."""
    config = ctx.obj["config"]

    async def find_duplicates():
        async with SimilarityAnalyzer(config) as analyzer:
            # Get work items from project
            if project:
                click.echo(f"Analyzing duplicates in project {project} (threshold: {threshold})")
            else:
                click.echo(f"Analyzing duplicates across all projects (threshold: {threshold})")

            # For now, we'll get items from Qdrant directly
            # In a full implementation, you'd filter by project

            # Mock implementation - in reality you'd get actual work items
            from datetime import datetime

            from ..models import WorkItem

            mock_items = [
                WorkItem(
                    key=f"{project or 'TEST'}-1",
                    summary="Sample work item",
                    issue_type="Story",
                    status="Open",
                    created=datetime.now(),
                    updated=datetime.now(),
                    reporter="system",
                )
            ]

            report = await analyzer.find_potential_duplicates(mock_items, threshold)

            click.echo("\nDuplicate Analysis Results:")
            click.echo(f"Total candidates: {report.total_candidates}")
            click.echo(f"Processing cost: ${report.processing_cost:.4f}")

            for level, candidates in report.candidates_by_level.items():
                if candidates:
                    click.echo(f"\n{level.upper()} confidence ({len(candidates)} items):")
                    for candidate in candidates[:3]:  # Show first 3
                        click.echo(f"  • {candidate.original_key} ↔ {candidate.duplicate_key}")
                        click.echo(f"    Similarity: {candidate.similarity_score:.3f}")
                        click.echo(f"    Action: {candidate.suggested_action}")

    asyncio.run(find_duplicates())


@analyze.command("cross-epic")
@click.option("--project", "-p", help="Project key to analyze")
@click.pass_context
def analyze_cross_epic(ctx, project):
    """Find work items that might belong to different Epics."""
    config = ctx.obj["config"]

    async def cross_epic_analysis():
        async with CrossEpicAnalyzer(config) as analyzer:
            click.echo(f"Analyzing cross-Epic relationships for {project or 'all projects'}")

            report = await analyzer.find_misplaced_work_items(project)

            click.echo("\nCross-Epic Analysis Results:")
            click.echo(f"Epics analyzed: {report.epics_analyzed}")
            click.echo(f"Misplaced items found: {len(report.misplaced_items)}")
            click.echo(f"Processing cost: ${report.processing_cost:.4f}")

            if report.misplaced_items:
                click.echo("\nMisplaced Work Items:")
                for item in report.misplaced_items[:5]:  # Show first 5
                    click.echo(f"  • {item.work_item_key}")
                    click.echo(f"    Current Epic: {item.current_epic_key}")
                    click.echo(f"    Suggested Epic: {item.suggested_epic_key}")
                    click.echo(f"    Confidence: {item.confidence_score:.3f}")
                    click.echo(f"    Reasoning: {item.reasoning}")
                    click.echo()

    asyncio.run(cross_epic_analysis())


@analyze.command("quality")
@click.argument("work_item_key")
@click.pass_context
def analyze_quality(ctx, work_item_key):
    """Analyze content quality of a work item."""
    config = ctx.obj["config"]

    async def quality_analysis():
        from ..clients.mcp_client import MCPClient

        # Get the work item first
        async with MCPClient(config) as mcp_client:
            work_item = await mcp_client.get_work_item(work_item_key)

            if not work_item:
                click.echo(f"Work item {work_item_key} not found")
                return

            async with ContentAnalyzer(config) as analyzer:
                click.echo(f"Analyzing content quality for {work_item_key}")

                analysis = await analyzer.analyze_description_quality(work_item)

                click.echo("\nQuality Analysis Results:")
                click.echo(f"Overall Score: {analysis.overall_score:.1f}/5.0")
                click.echo(f"Risk Level: {analysis.risk_level}")
                click.echo(f"Analysis Cost: ${analysis.analysis_cost:.4f}")

                click.echo("\nDetailed Scores:")
                click.echo(f"  Clarity: {analysis.clarity_score}/5")
                click.echo(f"  Completeness: {analysis.completeness_score}/5")
                click.echo(f"  Actionability: {analysis.actionability_score}/5")
                click.echo(f"  Testability: {analysis.testability_score}/5")

                if analysis.improvement_suggestions:
                    click.echo("\nImprovement Suggestions:")
                    for suggestion in analysis.improvement_suggestions:
                        click.echo(f"  • {suggestion}")

    asyncio.run(quality_analysis())


@analyze.command("template")
@click.option("--issue-type", required=True, help="Issue type to generate template for")
@click.option("--project", "-p", help="Project key to analyze")
@click.pass_context
def analyze_template(ctx, issue_type, project):
    """Generate template from high-quality examples."""
    config = ctx.obj["config"]

    async def template_analysis():
        async with TemplateInferenceEngine(config) as engine:
            click.echo(f"Generating template for {issue_type} from {project or 'all'} projects")

            # Mock high-quality samples - in reality you'd query for actual high-quality items
            from datetime import datetime

            from ..models import WorkItem

            mock_samples = [
                WorkItem(
                    key=f"{project or 'TEST'}-{i}",
                    summary=f"Sample {issue_type} {i}",
                    description=f"Sample description for {issue_type} {i}",
                    issue_type=issue_type,
                    status="Done",
                    created=datetime.now(),
                    updated=datetime.now(),
                    reporter="system",
                    components=["frontend", "backend"],
                    labels=["high-quality", "template"],
                )
                for i in range(1, 4)
            ]

            template = await engine.infer_templates_from_samples(issue_type, mock_samples)

            click.echo("\nTemplate Generated:")
            click.echo(f"Confidence Score: {template.confidence_score:.2f}")
            click.echo(f"Sample Count: {template.sample_count}")
            click.echo(f"Generation Cost: ${template.generation_cost:.4f}")

            click.echo("\nTitle Template:")
            click.echo(f"  {template.title_template}")

            click.echo("\nDescription Template:")
            click.echo(f"  {template.description_template}")

            if template.required_fields:
                click.echo("\nRequired Fields:")
                for field in template.required_fields:
                    click.echo(f"  • {field}")

            if template.common_components:
                click.echo("\nCommon Components:")
                for component in template.common_components:
                    click.echo(f"  • {component}")

    asyncio.run(template_analysis())


@analyze.command("tech-debt")
@click.option("--project", "-p", help="Project key to analyze")
@click.pass_context
def analyze_tech_debt(ctx, project):
    """Cluster technical debt items for prioritization."""
    config = ctx.obj["config"]

    async def tech_debt_analysis():
        async with StructuralAnalyzer(config) as analyzer:
            click.echo(f"Clustering technical debt items for {project or 'all projects'}")

            report = await analyzer.tech_debt_clustering()

            click.echo("\nTech Debt Clustering Results:")
            click.echo(f"Total tech debt items: {report.total_tech_debt_items}")
            click.echo(f"Clusters found: {len(report.clusters)}")
            click.echo(f"Processing cost: ${report.processing_cost:.4f}")

            if report.clusters:
                click.echo("\nTech Debt Clusters:")
                for cluster in report.clusters:
                    click.echo(f"\nCluster {cluster.cluster_id}: {cluster.theme}")
                    click.echo(f"  Items: {len(cluster.work_item_keys)}")
                    click.echo(f"  Priority: {cluster.priority_score:.2f}")
                    click.echo(f"  Estimated Effort: {cluster.estimated_effort}")
                    click.echo(f"  Impact: {cluster.impact_assessment}")
                    click.echo(f"  Approach: {cluster.recommended_approach}")

    asyncio.run(tech_debt_analysis())


@cli.command()
@click.pass_context
def auth(ctx):
    """Manage SSE authentication for Jira MCP endpoints."""
    config = ctx.obj["config"]

    async def manage_auth():
        from ..clients.auth import SSEAuthenticator

        # Check if this is an SSE endpoint
        if not (
            config.jira_mcp_endpoint
            and ("/sse" in config.jira_mcp_endpoint or "atlassian.com" in config.jira_mcp_endpoint)
        ):
            click.echo("❌ SSE authentication is only needed for SSE-based MCP endpoints.")
            click.echo(f"Current endpoint: {config.jira_mcp_endpoint}")
            click.echo("For Atlassian Cloud, use: https://mcp.atlassian.com/v1/sse")
            return

        authenticator = SSEAuthenticator(
            config.jira_mcp_endpoint,
            client_id=config.jira_sse_client_id,
            client_secret=config.jira_sse_client_secret or None,
        )

        try:
            click.echo("🔐 Starting SSE authentication flow...")
            tokens = await authenticator.get_auth_tokens(force_refresh=True)
            click.echo("✅ Authentication successful!")
            click.echo(f"   Token expires: {tokens.expires_at}")
            click.echo(f"   Saved to: {authenticator.cache_file}")

        except Exception as e:
            click.echo(f"❌ Authentication failed: {e}")
            ctx.exit(1)

    asyncio.run(manage_auth())


@cli.command()
@click.pass_context
def auth_status(ctx):
    """Check SSE authentication status."""
    config = ctx.obj["config"]

    import time

    from ..clients.auth import SSEAuthenticator

    if not (
        config.jira_mcp_endpoint
        and ("/sse" in config.jira_mcp_endpoint or "atlassian.com" in config.jira_mcp_endpoint)
    ):
        click.echo("ℹ️  SSE authentication is not applicable for this endpoint.")
        click.echo(f"Current endpoint: {config.jira_mcp_endpoint}")
        return

    authenticator = SSEAuthenticator(
        config.jira_mcp_endpoint,
        client_id=config.jira_sse_client_id,
        client_secret=config.jira_sse_client_secret or None,
    )

    cached_tokens = authenticator._load_cached_tokens()

    if not cached_tokens:
        click.echo("❌ No authentication tokens found.")
        click.echo("Run 'jirascope auth' to authenticate.")
        return

    current_time = time.time()
    if cached_tokens.is_expired:
        click.echo("⚠️  Authentication tokens are expired.")
        click.echo("Run 'jirascope auth' to re-authenticate.")
    else:
        time_left = (
            cached_tokens.expires_at - current_time if cached_tokens.expires_at else "unknown"
        )
        if isinstance(time_left, float):
            hours_left = int(time_left // 3600)
            minutes_left = int((time_left % 3600) // 60)
            click.echo("✅ Authentication tokens are valid.")
            click.echo(f"   Expires in: {hours_left}h {minutes_left}m")
        else:
            click.echo("✅ Authentication tokens are valid.")

    click.echo(f"   Cache file: {authenticator.cache_file}")


@cli.command()
@click.pass_context
def auth_clear(ctx):
    """Clear cached SSE authentication tokens."""
    config = ctx.obj["config"]

    from ..clients.auth import SSEAuthenticator

    authenticator = SSEAuthenticator(
        config.jira_mcp_endpoint,
        client_id=config.jira_sse_client_id,
        client_secret=config.jira_sse_client_secret or None,
    )

    authenticator.clear_cache()
    click.echo("🗑️  Cleared authentication cache.")
    click.echo("Run 'jirascope auth' to authenticate again.")


# Helper function to run async commands
def run_async(coro):
    """Helper to run async coroutines in CLI commands."""
    return asyncio.run(coro)


if __name__ == "__main__":
    cli()
