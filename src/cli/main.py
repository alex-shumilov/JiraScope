"""Main CLI entry point for JiraScope."""

import asyncio
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from jirascope.core.config import Config
from .utils import CostTracker, load_config_file

console = Console()


@click.group()
@click.version_option(version="1.0.0")
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--dry-run', is_flag=True, help='Preview actions without executing')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config: Optional[str], dry_run: bool, verbose: bool):
    """JiraScope - Semantic Work Item Analysis Platform"""
    ctx.ensure_object(dict)
    
    # Load configuration
    if config:
        config_path = Path(config)
        ctx.obj['config'] = load_config_file(config_path)
    else:
        ctx.obj['config'] = Config()
    
    ctx.obj['dry_run'] = dry_run
    ctx.obj['verbose'] = verbose
    ctx.obj['cost_tracker'] = CostTracker()
    
    if verbose:
        console.print(f"[dim]Config loaded, dry_run={dry_run}[/dim]")


@cli.command()
@click.option('--project', '-p', required=True, help='Jira project key')
@click.option('--batch-size', default=100, help='Batch size for processing')
@click.option('--show-costs', is_flag=True, help='Display cost breakdown')
@click.option('--incremental', is_flag=True, help='Only sync changed items')
@click.pass_context
def sync(ctx, project: str, batch_size: int, show_costs: bool, incremental: bool):
    """Sync Jira data and generate embeddings"""
    asyncio.run(_sync_async(ctx, project, batch_size, show_costs, incremental))


async def _sync_async(ctx, project: str, batch_size: int, show_costs: bool, incremental: bool):
    config = ctx.obj['config']
    cost_tracker = ctx.obj['cost_tracker']
    dry_run = ctx.obj['dry_run']
    
    if dry_run:
        console.print(f"[yellow]DRY RUN: Would sync project {project}[/yellow]")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Phase 1: Extract Jira data
        extract_task = progress.add_task("Extracting Jira data...", total=None)
        
        try:
            from jirascope.extraction.jira_extractor import JiraExtractor
            
            async with JiraExtractor(config) as extractor:
                if incremental:
                    extraction_result = await extractor.extract_incremental_updates(project)
                else:
                    extraction_result = await extractor.extract_project_data(project)
            
            progress.update(extract_task, completed=True)
            cost_tracker.track_operation("extraction", extraction_result.cost)
            
            # Phase 2: Process embeddings
            if extraction_result.new_items or extraction_result.updated_items:
                embed_task = progress.add_task("Processing embeddings...", total=None)
                
                from jirascope.processing.embedding_processor import EmbeddingProcessor
                
                async with EmbeddingProcessor(config) as processor:
                    processing_result = await processor.process_work_items(
                        extraction_result.new_items + extraction_result.updated_items,
                        batch_size=batch_size
                    )
                
                progress.update(embed_task, completed=True)
                cost_tracker.track_operation("embeddings", processing_result.total_cost)
            else:
                processing_result = type('ProcessingResult', (), {
                    'processed_items': 0,
                    'processing_time': 0.0,
                    'total_cost': 0.0
                })()
        
        except Exception as e:
            console.print(f"[red]Error during sync: {e}[/red]")
            raise click.ClickException(f"Sync failed: {e}")
    
    # Display results
    results_table = Table(title="Sync Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    
    results_table.add_row("Items Extracted", str(extraction_result.total_items))
    results_table.add_row("Items Processed", str(processing_result.processed_items))
    results_table.add_row("Processing Time", f"{processing_result.processing_time:.2f}s")
    
    if show_costs:
        total_cost = extraction_result.cost + processing_result.total_cost
        results_table.add_row("Total Cost", f"${total_cost:.4f}")
        results_table.add_row("API Calls", str(extraction_result.api_calls))
    
    console.print(results_table)


@cli.group()
def analyze():
    """Analysis commands"""
    pass


@analyze.command()
@click.option('--threshold', default=0.8, help='Similarity threshold (0.0-1.0)')
@click.option('--cost-estimate', is_flag=True, help='Show cost estimate before running')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def duplicates(ctx, threshold: float, cost_estimate: bool, output: Optional[str]):
    """Find potential duplicate work items"""
    asyncio.run(_duplicates_async(ctx, threshold, cost_estimate, output))


async def _duplicates_async(ctx, threshold: float, cost_estimate: bool, output: Optional[str]):
    config = ctx.obj['config']
    cost_tracker = ctx.obj['cost_tracker']
    dry_run = ctx.obj['dry_run']
    
    if not (0.0 <= threshold <= 1.0):
        raise click.BadParameter("Threshold must be between 0.0 and 1.0")
    
    from jirascope.analysis.similarity_analyzer import SimilarityAnalyzer
    
    if cost_estimate:
        # Simple cost estimation
        estimated_cost = 0.05 * threshold  # Rough estimate
        console.print(f"[yellow]Estimated cost: ${estimated_cost:.4f}[/yellow]")
        
        if not click.confirm("Continue with analysis?"):
            return
    
    if dry_run:
        console.print("[yellow]DRY RUN: Would analyze duplicates[/yellow]")
        return
    
    try:
        with Progress() as progress:
            task = progress.add_task("Finding duplicates...", total=None)
            
            async with SimilarityAnalyzer(config) as analyzer:
                # Get work items from database/cache
                from jirascope.extraction.jira_extractor import JiraExtractor
                async with JiraExtractor(config) as extractor:
                    work_items = await extractor.get_all_work_items()
                
                duplicate_report = await analyzer.find_potential_duplicates(work_items, threshold)
            
            progress.update(task, completed=True)
        
        cost_tracker.track_operation("duplicate_analysis", duplicate_report.processing_cost)
        
        # Display results
        display_duplicate_results(duplicate_report, output)
        
    except Exception as e:
        console.print(f"[red]Error during duplicate analysis: {e}[/red]")
        raise click.ClickException(f"Analysis failed: {e}")


@analyze.command()
@click.option('--project', '-p', help='Jira project key to analyze')
@click.option('--use-claude', is_flag=True, help='Use Claude for content analysis')
@click.option('--budget', type=float, help='Maximum cost budget')
@click.pass_context
def quality(ctx, project: Optional[str], use_claude: bool, budget: Optional[float]):
    """Analyze work item quality"""
    asyncio.run(_quality_async(ctx, project, use_claude, budget))


@analyze.command()
@click.argument('epic_key')
@click.option('--depth', type=click.Choice(['basic', 'full']), default='basic')
@click.option('--use-claude', is_flag=True, help='Use Claude for detailed analysis')
@click.pass_context
def epic(ctx, epic_key: str, depth: str, use_claude: bool):
    """Analyze Epic comprehensively"""
    asyncio.run(_epic_async(ctx, epic_key, depth, use_claude))


@analyze.command()
@click.option('--project', '-p', help='Jira project key to analyze')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.pass_context
def drift(ctx, project: Optional[str], start_date: Optional[str], end_date: Optional[str]):
    """Detect scope drift in work items"""
    asyncio.run(_drift_async(ctx, project, start_date, end_date))


async def _quality_async(ctx, project: Optional[str], use_claude: bool, budget: Optional[float]):
    config = ctx.obj['config']
    cost_tracker = ctx.obj['cost_tracker']
    dry_run = ctx.obj['dry_run']
    
    if use_claude and budget:
        console.print(f"[yellow]Budget set to ${budget:.2f}[/yellow]")
    
    if dry_run:
        console.print(f"[yellow]DRY RUN: Would analyze quality for project {project or 'all'}[/yellow]")
        return
    
    try:
        with Progress() as progress:
            task = progress.add_task("Analyzing quality...", total=None)
            
            if use_claude:
                from jirascope.analysis.content_analyzer import ContentAnalyzer
                
                async with ContentAnalyzer(config) as analyzer:
                    # Get work items
                    from jirascope.extraction.jira_extractor import JiraExtractor
                    async with JiraExtractor(config) as extractor:
                        if project:
                            work_items = await extractor.get_project_work_items(project)
                        else:
                            work_items = await extractor.get_all_work_items()
                    
                    # Analyze first 10 items for demo
                    quality_analyses = []
                    total_cost = 0.0
                    
                    for item in work_items[:10]:
                        if budget and total_cost >= budget:
                            console.print(f"[yellow]Budget limit reached: ${budget:.2f}[/yellow]")
                            break
                        
                        analysis = await analyzer.analyze_description_quality(item)
                        quality_analyses.append(analysis)
                        total_cost += analysis.analysis_cost
                        
                        if budget and total_cost >= budget * 0.8:
                            console.print(f"[yellow]Warning: 80% of budget used[/yellow]")
            
            progress.update(task, completed=True)
        
        cost_tracker.track_operation("quality_analysis", total_cost)
        
        # Display results
        display_quality_results(quality_analyses)
        
    except Exception as e:
        console.print(f"[red]Error during quality analysis: {e}[/red]")
        raise click.ClickException(f"Analysis failed: {e}")


async def _epic_async(ctx, epic_key: str, depth: str, use_claude: bool):
    config = ctx.obj['config']
    cost_tracker = ctx.obj['cost_tracker']
    dry_run = ctx.obj['dry_run']
    
    if dry_run:
        console.print(f"[yellow]DRY RUN: Would analyze epic {epic_key}[/yellow]")
        return
    
    try:
        with Progress() as progress:
            task = progress.add_task(f"Analyzing epic {epic_key}...", total=None)
            
            # Get epic data
            from jirascope.extraction.jira_extractor import JiraExtractor
            async with JiraExtractor(config) as extractor:
                epic_data = await extractor.get_epic_hierarchy(epic_key)
            
            if depth == "full" and use_claude:
                # Comprehensive analysis with Claude
                from jirascope.analysis.cross_epic_analyzer import CrossEpicAnalyzer
                async with CrossEpicAnalyzer(config) as analyzer:
                    epic_report = await analyzer.analyze_epic_comprehensively(epic_data)
                
                cost_tracker.track_operation("epic_analysis", epic_report.analysis_cost)
            else:
                # Basic analysis
                epic_report = {"epic_key": epic_key, "basic_stats": "completed"}
            
            progress.update(task, completed=True)
        
        # Display results
        display_epic_results(epic_report)
        
    except Exception as e:
        console.print(f"[red]Error during epic analysis: {e}[/red]")
        raise click.ClickException(f"Epic analysis failed: {e}")


async def _drift_async(ctx, project: Optional[str], start_date: Optional[str], end_date: Optional[str]):
    config = ctx.obj['config']
    cost_tracker = ctx.obj['cost_tracker']
    dry_run = ctx.obj['dry_run']
    
    if dry_run:
        console.print(f"[yellow]DRY RUN: Would analyze scope drift for {project or 'all projects'}[/yellow]")
        return
    
    try:
        # Parse dates if provided
        start_dt = None
        end_dt = None
        if start_date:
            from datetime import datetime
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            from datetime import datetime
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        with Progress() as progress:
            task = progress.add_task("Analyzing scope drift...", total=None)
            
            from jirascope.analysis.temporal_analyzer import TemporalAnalyzer
            async with TemporalAnalyzer(config) as analyzer:
                if project:
                    drift_report = await analyzer.detect_scope_drift_for_project(
                        project, start_date=start_dt, end_date=end_dt
                    )
                else:
                    # Analyze all projects (simplified for demo)
                    drift_report = await analyzer.detect_scope_drift_for_project(
                        "ALL", start_date=start_dt, end_date=end_dt
                    )
            
            progress.update(task, completed=True)
        
        cost_tracker.track_operation("drift_analysis", drift_report.total_analysis_cost)
        
        # Display results
        display_drift_results(drift_report)
        
    except Exception as e:
        console.print(f"[red]Error during drift analysis: {e}[/red]")
        raise click.ClickException(f"Drift analysis failed: {e}")


@cli.command()
@click.pass_context
def health_check(ctx):
    """Check health of all services"""
    asyncio.run(_health_check_async(ctx))


async def _health_check_async(ctx):
    config = ctx.obj['config']
    health_status = {}
    
    # Check each service
    services = [
        ("Jira MCP", _check_jira_health),
        ("LM Studio", _check_lm_health),
        ("Qdrant", _check_qdrant_health),
        ("Claude", _check_claude_health)
    ]
    
    for service_name, check_func in services:
        try:
            await check_func(config)
            health_status[service_name] = 'healthy'
        except Exception as e:
            health_status[service_name] = f'unhealthy: {str(e)[:50]}'
    
    # Display results
    all_healthy = all(status == 'healthy' for status in health_status.values())
    
    if all_healthy:
        console.print("[green]✅ All services healthy[/green]")
    else:
        console.print("[red]❌ Some services unhealthy[/red]")
    
    for service, status in health_status.items():
        color = "green" if status == "healthy" else "red"
        console.print(f"[{color}]{service}: {status}[/{color}]")
    
    if not all_healthy:
        raise click.ClickException("Health check failed")


@cli.command()
@click.pass_context
def summary(ctx):
    """Display session cost summary"""
    cost_tracker = ctx.obj['cost_tracker']
    cost_tracker.display_cost_summary()


# Health check functions
async def _check_jira_health(config):
    from jirascope.clients.jira_client import JiraClient
    async with JiraClient(config) as client:
        await client.test_connection()


async def _check_lm_health(config):
    from jirascope.clients.lmstudio_client import LMStudioClient
    async with LMStudioClient(config) as client:
        await client.test_connection()


async def _check_qdrant_health(config):
    from jirascope.clients.qdrant_client import QdrantVectorClient
    async with QdrantVectorClient(config) as client:
        await client.test_connection()


async def _check_claude_health(config):
    from jirascope.clients.claude_client import ClaudeClient
    async with ClaudeClient(config) as client:
        await client.test_connection()


# Display functions
def display_duplicate_results(report, output_path: Optional[str]):
    """Display duplicate analysis results"""
    results_table = Table(title="Duplicate Analysis Results")
    results_table.add_column("Level", style="cyan")
    results_table.add_column("Count", style="green")
    
    for level, candidates in report.candidates_by_level.items():
        results_table.add_row(level.title(), str(len(candidates)))
    
    results_table.add_row("Total", str(report.total_candidates), style="bold")
    results_table.add_row("Cost", f"${report.processing_cost:.4f}", style="dim")
    
    console.print(results_table)
    
    if output_path:
        from .export import export_duplicate_report
        export_duplicate_report(report, output_path)
        console.print(f"[green]Results exported to {output_path}[/green]")


def display_quality_results(analyses):
    """Display quality analysis results"""
    if not analyses:
        console.print("[yellow]No quality analyses to display[/yellow]")
        return
    
    results_table = Table(title="Quality Analysis Results")
    results_table.add_column("Work Item", style="cyan")
    results_table.add_column("Overall Score", style="green")
    results_table.add_column("Risk Level", style="yellow")
    
    for analysis in analyses:
        results_table.add_row(
            analysis.work_item_key,
            f"{analysis.overall_score:.1f}/5.0",
            analysis.risk_level
        )
    
    console.print(results_table)


def display_epic_results(epic_report):
    """Display epic analysis results"""
    if isinstance(epic_report, dict):
        # Simple dict result
        results_table = Table(title="Epic Analysis Results")
        results_table.add_column("Property", style="cyan")
        results_table.add_column("Value", style="green")
        
        for key, value in epic_report.items():
            results_table.add_row(str(key), str(value))
        
        console.print(results_table)
    else:
        # Comprehensive report object
        console.print(f"[green]Epic Analysis Complete[/green]")
        console.print(f"Epic: {epic_report.epic_key}")
        console.print(f"Analysis Cost: ${epic_report.analysis_cost:.4f}")


def display_drift_results(drift_report):
    """Display scope drift analysis results"""
    results_table = Table(title="Scope Drift Analysis Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    
    results_table.add_row("Project", drift_report.project_key)
    results_table.add_row("Items Analyzed", str(drift_report.total_items_analyzed))
    results_table.add_row("Items with Drift", str(drift_report.items_with_drift))
    results_table.add_row("Analysis Cost", f"${drift_report.total_analysis_cost:.4f}")
    
    console.print(results_table)
    
    if drift_report.items_with_drift > 0:
        console.print("[yellow]⚠️  Scope drift detected in some work items[/yellow]")
    else:
        console.print("[green]✅ No significant scope drift detected[/green]")


if __name__ == "__main__":
    cli()