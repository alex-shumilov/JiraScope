"""Template inference from high-quality work item examples."""

import json
import time

from ..clients.claude_client import ClaudeClient
from ..core.config import Config
from ..models import TemplateInference, WorkItem
from ..utils.logging import StructuredLogger

logger = StructuredLogger(__name__)


class TemplateInferenceEngine:
    """Generate templates from high-quality work item examples."""

    def __init__(self, config: Config):
        self.config = config
        self.claude_client: ClaudeClient | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.claude_client = ClaudeClient(self.config)
        await self.claude_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.claude_client:
            await self.claude_client.__aexit__(exc_type, exc_val, exc_tb)

    async def infer_templates_from_samples(
        self, issue_type: str, high_quality_samples: list[WorkItem]
    ) -> TemplateInference:
        """Generate templates from high-quality work item examples."""
        logger.info(f"Inferring template for {issue_type} from {len(high_quality_samples)} samples")
        start_time = time.time()

        if not self.claude_client:
            raise RuntimeError(
                "TemplateInferenceEngine not initialized. Use async context manager."
            )

        if len(high_quality_samples) < 3:
            raise ValueError("Need at least 3 high-quality samples to infer a template")

        try:
            # Prepare sample texts
            sample_texts = []
            all_components = set()
            all_labels = set()

            for i, item in enumerate(high_quality_samples, 1):
                sample_text = f"""
Example {i}:
Title: {item.summary}
Description: {item.description or "No description"}
Components: {', '.join(item.components) if item.components else "None"}
Labels: {', '.join(item.labels) if item.labels else "None"}
Status: {item.status}
"""
                sample_texts.append(sample_text)
                all_components.update(item.components)
                all_labels.update(item.labels)

            # Create inference prompt
            prompt = f"""Based on these high-quality {issue_type} examples, create a comprehensive template that captures common patterns:

{chr(10).join(sample_texts)}

Analyze these examples and generate:
1. A title template with placeholders (use {{placeholder}} format)
2. A description template with standard sections that appear across examples
3. A checklist of required fields that should always be filled
4. Common components that frequently appear for this issue type
5. Common labels that are typically used

Focus on patterns that appear in multiple examples. Look for:
- Common structure in titles
- Recurring sections in descriptions (like "Acceptance Criteria", "Technical Notes", etc.)
- Required information that's always present
- Patterns in component and label usage

Respond in JSON format:
{{
    "title_template": "template with {{placeholders}}",
    "description_template": "multi-line template with standard sections",
    "required_fields": ["field1", "field2", "field3"],
    "common_components": ["component1", "component2"],
    "common_labels": ["label1", "label2"],
    "confidence_score": 0.0-1.0,
    "template_notes": "explanation of patterns found"
}}"""

            # Call Claude
            response = await self.claude_client.analyze(
                prompt=prompt, analysis_type="template_inference"
            )

            # Parse response
            try:
                template_data = json.loads(response.content)
            except json.JSONDecodeError:
                template_data = self._parse_fallback_template_response(
                    response.content, issue_type, high_quality_samples
                )

            processing_time = time.time() - start_time

            # Create template inference result
            template = TemplateInference(
                issue_type=issue_type,
                title_template=template_data.get(
                    "title_template", f"{issue_type}: {{description}}"
                ),
                description_template=template_data.get(
                    "description_template",
                    "## Description\n\n{{description}}\n\n## Acceptance Criteria\n\n- [ ] {{criteria}}",
                ),
                required_fields=template_data.get("required_fields", ["summary", "description"]),
                common_components=template_data.get("common_components", list(all_components)[:5]),
                common_labels=template_data.get("common_labels", list(all_labels)[:5]),
                confidence_score=template_data.get("confidence_score", 0.7),
                sample_count=len(high_quality_samples),
                generation_cost=response.cost,
            )

            logger.log_operation(
                "infer_templates_from_samples",
                processing_time,
                success=True,
                issue_type=issue_type,
                sample_count=len(high_quality_samples),
                confidence_score=template.confidence_score,
            )

            return template

        except Exception as e:
            logger.error(f"Failed to infer template for {issue_type}", error=str(e))
            raise

    async def infer_multiple_templates(
        self, samples_by_type: dict[str, list[WorkItem]]
    ) -> dict[str, TemplateInference]:
        """Infer templates for multiple issue types."""
        logger.info(f"Inferring templates for {len(samples_by_type)} issue types")

        templates = {}

        for issue_type, samples in samples_by_type.items():
            if len(samples) >= 3:  # Minimum samples required
                try:
                    template = await self.infer_templates_from_samples(issue_type, samples)
                    templates[issue_type] = template
                except Exception as e:
                    logger.warning(f"Failed to infer template for {issue_type}: {e!s}")
            else:
                logger.warning(
                    f"Insufficient samples for {issue_type}: {len(samples)} (need at least 3)"
                )

        return templates

    def _parse_fallback_template_response(
        self, content: str, issue_type: str, samples: list[WorkItem]
    ) -> dict[str, any]:
        """Fallback parser when JSON parsing fails."""
        logger.warning("Template inference JSON parsing failed, using fallback")

        # Extract common components and labels from samples
        all_components = set()
        all_labels = set()

        for item in samples:
            all_components.update(item.components)
            all_labels.update(item.labels)

        # Basic template based on issue type
        title_templates = {
            "Story": "User Story: {feature_description}",
            "Task": "Task: {task_description}",
            "Bug": "Bug: {issue_description}",
            "Epic": "Epic: {epic_theme}",
            "Improvement": "Improvement: {enhancement_description}",
        }

        description_templates = {
            "Story": """## User Story
As a {user_type}, I want {functionality} so that {benefit}.

## Acceptance Criteria
- [ ] {criterion_1}
- [ ] {criterion_2}
- [ ] {criterion_3}

## Technical Notes
{technical_details}""",
            "Task": """## Description
{task_description}

## Requirements
- {requirement_1}
- {requirement_2}

## Definition of Done
- [ ] {done_criterion_1}
- [ ] {done_criterion_2}""",
            "Bug": """## Summary
{bug_summary}

## Steps to Reproduce
1. {step_1}
2. {step_2}
3. {step_3}

## Expected Behavior
{expected_behavior}

## Actual Behavior
{actual_behavior}

## Environment
- Browser: {browser}
- OS: {operating_system}""",
            "Epic": """## Epic Description
{epic_overview}

## Business Value
{business_value}

## Success Criteria
- {success_criterion_1}
- {success_criterion_2}

## Out of Scope
{out_of_scope}""",
            "Improvement": """## Current State
{current_state}

## Proposed Improvement
{improvement_description}

## Expected Benefits
- {benefit_1}
- {benefit_2}

## Implementation Notes
{implementation_notes}""",
        }

        return {
            "title_template": title_templates.get(issue_type, f"{issue_type}: {{description}}"),
            "description_template": description_templates.get(
                issue_type, "## Description\n\n{description}\n\n## Requirements\n\n{requirements}"
            ),
            "required_fields": ["summary", "description", "issue_type"],
            "common_components": list(all_components)[:5],
            "common_labels": list(all_labels)[:5],
            "confidence_score": 0.5,  # Lower confidence for fallback
            "template_notes": "Generated using fallback method due to parsing error",
        }

    def validate_template_quality(self, template: TemplateInference) -> dict[str, any]:
        """Validate the quality of an inferred template."""
        validation_results = {"is_valid": True, "issues": [], "suggestions": []}

        # Check title template
        if not template.title_template or "{" not in template.title_template:
            validation_results["issues"].append("Title template missing placeholders")
            validation_results["is_valid"] = False

        # Check description template
        if not template.description_template:
            validation_results["issues"].append("Description template is empty")
            validation_results["is_valid"] = False
        elif len(template.description_template) < 50:
            validation_results["suggestions"].append("Description template might be too brief")

        # Check required fields
        if len(template.required_fields) < 2:
            validation_results["suggestions"].append(
                "Consider more required fields for completeness"
            )

        # Check confidence score
        if template.confidence_score < 0.6:
            validation_results["suggestions"].append(
                "Low confidence score - consider adding more samples"
            )

        return validation_results
