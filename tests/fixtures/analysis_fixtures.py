"""Test fixtures for analysis components."""

from datetime import datetime, timedelta

import pytest

from jirascope.models import EpicHierarchy, WorkItem


class AnalysisFixtures:
    """Centralized test fixtures for analysis components."""

    @staticmethod
    def create_sample_work_items() -> list[WorkItem]:
        """Create sample work items for testing."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        return [
            # Duplicate candidates
            WorkItem(
                key="TEST-1",
                summary="User login functionality",
                description="Implement user authentication system with OAuth2 support",
                issue_type="Story",
                status="Open",
                created=base_time,
                updated=base_time + timedelta(days=1),
                reporter="john.doe",
                components=["frontend", "backend"],
                labels=["authentication", "security"],
            ),
            WorkItem(
                key="TEST-2",
                summary="User authentication feature",
                description="Create user login system with OAuth2 integration",
                issue_type="Story",
                status="In Progress",
                created=base_time + timedelta(days=1),
                updated=base_time + timedelta(days=2),
                reporter="jane.smith",
                components=["frontend", "backend"],
                labels=["auth", "security"],
            ),
            # Tech debt items
            WorkItem(
                key="TEST-3",
                summary="Refactor legacy payment processing",
                description="The current payment system uses deprecated APIs and needs modernization",
                issue_type="Task",
                status="Open",
                created=base_time + timedelta(days=2),
                updated=base_time + timedelta(days=2),
                reporter="tech.lead",
                components=["backend"],
                labels=["technical-debt", "refactor"],
            ),
            WorkItem(
                key="TEST-4",
                summary="Cleanup outdated database queries",
                description="Remove legacy SQL queries that are no longer used",
                issue_type="Improvement",
                status="Open",
                created=base_time + timedelta(days=3),
                updated=base_time + timedelta(days=3),
                reporter="tech.lead",
                components=["backend"],
                labels=["cleanup", "database"],
            ),
            # High quality examples
            WorkItem(
                key="TEST-5",
                summary="User Dashboard Analytics Widget",
                description="""## User Story
As a business user, I want to see analytics widgets on my dashboard so that I can track key metrics at a glance.

## Acceptance Criteria
- [ ] Widget displays monthly revenue chart
- [ ] Widget shows active user count
- [ ] Widget updates in real-time
- [ ] Widget is responsive on mobile devices

## Technical Notes
- Use Chart.js for visualization
- Connect to analytics API endpoint
- Implement WebSocket for real-time updates""",
                issue_type="Story",
                status="Done",
                created=base_time + timedelta(days=4),
                updated=base_time + timedelta(days=10),
                reporter="product.manager",
                components=["frontend"],
                labels=["dashboard", "analytics", "high-quality"],
            ),
            # Complex item that should be split
            WorkItem(
                key="TEST-6",
                summary="Complete E-commerce Platform Overhaul",
                description="""Redesign the entire e-commerce platform including:
- New modern UI/UX design
- Payment system integration with multiple providers
- Inventory management system
- Order tracking and notifications
- Customer support chat system
- Admin dashboard with analytics
- Mobile app development
- API documentation
- Performance optimization
- Security audit and implementation""",
                issue_type="Epic",
                status="Open",
                created=base_time + timedelta(days=5),
                updated=base_time + timedelta(days=5),
                reporter="cto",
                components=["frontend", "backend", "mobile"],
                labels=["epic", "overhaul"],
            ),
            # Cross-epic analysis candidates
            WorkItem(
                key="TEST-7",
                summary="Database schema migration",
                description="Migrate user tables to new schema format",
                issue_type="Task",
                status="Open",
                created=base_time + timedelta(days=6),
                updated=base_time + timedelta(days=6),
                reporter="developer",
                epic_key="EPIC-INFRASTRUCTURE",
                components=["backend"],
                labels=["database", "migration"],
            ),
            WorkItem(
                key="TEST-8",
                summary="User profile API endpoints",
                description="Create REST API endpoints for user profile management",
                issue_type="Story",
                status="Open",
                created=base_time + timedelta(days=7),
                updated=base_time + timedelta(days=7),
                reporter="developer",
                epic_key="EPIC-USER-FEATURES",
                components=["backend"],
                labels=["api", "user-profile"],
            ),
        ]

    @staticmethod
    def create_sample_epics() -> list[WorkItem]:
        """Create sample epics for testing."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        return [
            WorkItem(
                key="EPIC-USER-FEATURES",
                summary="User Management Features",
                description="Collection of user-related functionality including profiles, authentication, and preferences",
                issue_type="Epic",
                status="In Progress",
                created=base_time,
                updated=base_time + timedelta(days=5),
                reporter="product.manager",
            ),
            WorkItem(
                key="EPIC-INFRASTRUCTURE",
                summary="Infrastructure and DevOps",
                description="Backend infrastructure, database, and deployment automation",
                issue_type="Epic",
                status="Open",
                created=base_time + timedelta(days=1),
                updated=base_time + timedelta(days=3),
                reporter="tech.lead",
            ),
            WorkItem(
                key="EPIC-ECOMMERCE",
                summary="E-commerce Platform",
                description="Complete e-commerce functionality including payments, orders, and inventory",
                issue_type="Epic",
                status="Open",
                created=base_time + timedelta(days=2),
                updated=base_time + timedelta(days=2),
                reporter="cto",
            ),
        ]

    @staticmethod
    def create_sample_hierarchies() -> list[EpicHierarchy]:
        """Create sample epic hierarchies for testing."""
        epics = AnalysisFixtures.create_sample_epics()
        work_items = AnalysisFixtures.create_sample_work_items()

        # Group work items by epic
        user_features_items = [item for item in work_items if item.epic_key == "EPIC-USER-FEATURES"]
        infrastructure_items = [
            item for item in work_items if item.epic_key == "EPIC-INFRASTRUCTURE"
        ]

        # Add some items without explicit epic assignment
        unassigned_items = [item for item in work_items if not item.epic_key]
        user_features_items.extend(unassigned_items[:2])  # Assign first 2 unassigned
        infrastructure_items.extend(unassigned_items[2:])  # Assign rest

        return [
            EpicHierarchy(
                epic=epics[0],  # EPIC-USER-FEATURES
                stories=user_features_items[:3],
                tasks=user_features_items[3:],
                subtasks=[],
            ),
            EpicHierarchy(
                epic=epics[1],  # EPIC-INFRASTRUCTURE
                stories=infrastructure_items[:2],
                tasks=infrastructure_items[2:],
                subtasks=[],
            ),
            EpicHierarchy(
                epic=epics[2],  # EPIC-ECOMMERCE
                stories=[work_items[5]],  # The complex item
                tasks=[],
                subtasks=[],
            ),
        ]

    @staticmethod
    def create_mock_embeddings() -> list[list[float]]:
        """Create mock embeddings for testing."""
        import random

        random.seed(42)  # For reproducible tests

        # Create 8 embeddings (one for each work item)
        embeddings = []
        for i in range(8):
            # Create 384-dimensional embeddings (common size)
            embedding = [random.uniform(-1, 1) for _ in range(384)]
            embeddings.append(embedding)

        # Make first two embeddings similar (for duplicate testing)
        for j in range(384):
            embeddings[1][j] = embeddings[0][j] + random.uniform(-0.1, 0.1)

        return embeddings

    @staticmethod
    def create_scope_drift_history():
        """Create mock change history for scope drift testing."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        return [
            {
                "timestamp": base_time,
                "description": "Simple user login form with username and password",
                "author": "product.manager",
                "fields": {"description": "Simple user login form with username and password"},
            },
            {
                "timestamp": base_time + timedelta(days=7),
                "description": "User login form with username, password, and remember me option",
                "author": "developer",
                "fields": {
                    "description": "User login form with username, password, and remember me option"
                },
            },
            {
                "timestamp": base_time + timedelta(days=14),
                "description": "Complete authentication system with OAuth2, 2FA, password reset, social logins, and security auditing",
                "author": "tech.lead",
                "fields": {
                    "description": "Complete authentication system with OAuth2, 2FA, password reset, social logins, and security auditing"
                },
            },
        ]


# Pytest fixtures
@pytest.fixture
def sample_work_items():
    """Fixture providing sample work items."""
    return AnalysisFixtures.create_sample_work_items()


@pytest.fixture
def sample_epics():
    """Fixture providing sample epics."""
    return AnalysisFixtures.create_sample_epics()


@pytest.fixture
def sample_hierarchies():
    """Fixture providing sample epic hierarchies."""
    return AnalysisFixtures.create_sample_hierarchies()


@pytest.fixture
def mock_embeddings():
    """Fixture providing mock embeddings."""
    return AnalysisFixtures.create_mock_embeddings()


@pytest.fixture
def scope_drift_history():
    """Fixture providing scope drift change history."""
    return AnalysisFixtures.create_scope_drift_history()


@pytest.fixture
def mock_claude_responses():
    """Fixture providing mock Claude API responses."""
    return {
        "quality_analysis": {
            "content": '{"clarity_score": 4, "completeness_score": 3, "actionability_score": 4, "testability_score": 3, "overall_score": 3.5, "improvement_suggestions": ["Add more specific acceptance criteria", "Include technical implementation notes"], "risk_level": "Low"}',
            "cost": 0.02,
        },
        "split_analysis": {
            "content": '{"should_split": true, "complexity_score": 0.8, "reasoning": "This item contains multiple distinct features that can be developed independently", "suggested_splits": [{"title": "UI/UX Redesign", "description": "Focus on frontend design", "estimated_effort": "Medium", "dependencies": []}, {"title": "Payment Integration", "description": "Backend payment system", "estimated_effort": "Large", "dependencies": ["UI/UX Redesign"]}]}',
            "cost": 0.03,
        },
        "template_inference": {
            "content": '{"title_template": "User Story: {feature_description}", "description_template": "## User Story\\nAs a {user_type}, I want {functionality} so that {benefit}.\\n\\n## Acceptance Criteria\\n- [ ] {criterion_1}\\n- [ ] {criterion_2}\\n\\n## Technical Notes\\n{technical_details}", "required_fields": ["summary", "description", "acceptance_criteria"], "common_components": ["frontend", "backend"], "common_labels": ["user-story", "feature"], "confidence_score": 0.85, "template_notes": "Strong pattern found across samples"}',
            "cost": 0.04,
        },
        "scope_change_analysis": {
            "content": '{"change_type": "expansion", "impact_level": "major", "summary": "Significant scope expansion from simple login to complete authentication system"}',
            "cost": 0.025,
        },
        "tech_debt_cluster_analysis": {
            "content": '{"theme": "Legacy System Modernization", "priority_score": 0.75, "estimated_effort": "Large", "dependencies": ["database-migration", "api-updates"], "impact_assessment": "High impact on system performance and maintainability", "recommended_approach": "Prioritize payment system refactor first, then database cleanup"}',
            "cost": 0.035,
        },
    }
