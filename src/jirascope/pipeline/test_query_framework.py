"""Test query framework for RAG quality testing."""

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..clients.claude_client import ClaudeClient
from ..clients.mcp_client import MCPClient
from ..core.config import Config
from ..utils.logging import StructuredLogger
from .rag_quality_tester import RagTestQuery

logger = StructuredLogger(__name__)


class TestCategory(BaseModel):
    """Category of test queries."""

    name: str = Field(..., description="Category name")
    description: str = Field(..., description="Category description")
    min_coverage: int = Field(3, description="Minimum number of tests required in this category")
    tests: list[RagTestQuery] = Field(default_factory=list, description="Tests in this category")


class TestQueryCollection(BaseModel):
    """Collection of test queries organized by category."""

    categories: dict[str, TestCategory] = Field(default_factory=dict)
    total_tests: int = 0

    def add_test(self, test: RagTestQuery):
        """Add a test to the appropriate category."""
        if test.category not in self.categories:
            self.categories[test.category] = TestCategory(
                name=test.category, description=f"{test.category.capitalize()} tests", tests=[]
            )

        self.categories[test.category].tests.append(test)
        self.total_tests += 1

    def get_tests_by_category(self, category: str) -> list[RagTestQuery]:
        """Get all tests in a specific category."""
        return self.categories.get(
            category, TestCategory(name=category, description="", tests=[])
        ).tests

    def get_all_tests(self) -> list[RagTestQuery]:
        """Get all tests across all categories."""
        all_tests = []
        for category in self.categories.values():
            all_tests.extend(category.tests)
        return all_tests

    def check_coverage(self) -> dict[str, Any]:
        """Check if all categories meet minimum coverage requirements."""
        coverage_issues = {}

        for category_name, category in self.categories.items():
            if len(category.tests) < category.min_coverage:
                coverage_issues[category_name] = {
                    "current": len(category.tests),
                    "required": category.min_coverage,
                    "missing": category.min_coverage - len(category.tests),
                }

        return {
            "has_issues": len(coverage_issues) > 0,
            "issues": coverage_issues,
            "total_tests": self.total_tests,
        }


class TestQueryManager:
    """Manages test queries for RAG quality testing."""

    def __init__(self, config: Config):
        self.config = config
        self.collection = TestQueryCollection()
        self.default_categories = {
            "functional": TestCategory(
                name="functional",
                description="Tests for functional aspects of the system",
                min_coverage=3,
                tests=[],
            ),
            "technical": TestCategory(
                name="technical",
                description="Tests for technical components and capabilities",
                min_coverage=3,
                tests=[],
            ),
            "business": TestCategory(
                name="business",
                description="Tests for business processes and requirements",
                min_coverage=2,
                tests=[],
            ),
        }

    def load_default_tests(self):
        """Load default test queries."""
        # Add all default categories
        for category_name, category in self.default_categories.items():
            if category_name not in self.collection.categories:
                self.collection.categories[category_name] = category

        # Add sample tests based on category
        functional_tests = [
            RagTestQuery(
                id="auth_functionality",
                query_text="user authentication and login functionality",
                expected_work_items=[],
                minimum_similarity=0.7,
                category="functional",
                description="Should find authentication-related work items",
            ),
            RagTestQuery(
                id="user_management",
                query_text="user account management and permissions",
                expected_work_items=[],
                minimum_similarity=0.7,
                category="functional",
                description="Should find user management-related work items",
            ),
            RagTestQuery(
                id="reporting_features",
                query_text="reporting and analytics features",
                expected_work_items=[],
                minimum_similarity=0.6,
                category="functional",
                description="Should find reporting-related work items",
            ),
        ]

        technical_tests = [
            RagTestQuery(
                id="database_migration",
                query_text="database schema changes and migrations",
                expected_work_items=[],
                minimum_similarity=0.6,
                category="technical",
                description="Should identify database-related tasks",
            ),
            RagTestQuery(
                id="performance_optimization",
                query_text="application performance and speed optimization",
                expected_work_items=[],
                minimum_similarity=0.7,
                category="technical",
                description="Should identify performance-related tasks",
            ),
            RagTestQuery(
                id="security_implementation",
                query_text="security implementation and vulnerability fixes",
                expected_work_items=[],
                minimum_similarity=0.7,
                category="technical",
                description="Should identify security-related tasks",
            ),
        ]

        business_tests = [
            RagTestQuery(
                id="api_documentation",
                query_text="REST API documentation and endpoint specifications",
                expected_work_items=[],
                minimum_similarity=0.6,
                category="business",
                description="Should find documentation tasks",
            ),
            RagTestQuery(
                id="business_rules",
                query_text="business logic and workflow rules implementation",
                expected_work_items=[],
                minimum_similarity=0.6,
                category="business",
                description="Should find business logic tasks",
            ),
        ]

        # Add all tests
        for test in functional_tests + technical_tests + business_tests:
            self.collection.add_test(test)

    async def load_from_file(self, file_path: Path) -> bool:
        """Load test queries from a JSON file."""
        if not file_path.exists():
            logger.error(f"Test query file not found: {file_path}")
            return False

        try:
            with open(file_path) as f:
                data = json.load(f)

            # Reset collection
            self.collection = TestQueryCollection()

            # Load categories
            for category_data in data.get("categories", []):
                category = TestCategory(
                    name=category_data["name"],
                    description=category_data.get("description", ""),
                    min_coverage=category_data.get("min_coverage", 3),
                    tests=[],
                )

                self.collection.categories[category.name] = category

            # Load tests
            for test_data in data.get("tests", []):
                test = RagTestQuery(
                    id=test_data["id"],
                    query_text=test_data["query_text"],
                    expected_work_items=test_data.get("expected_work_items", []),
                    minimum_similarity=test_data.get("minimum_similarity", 0.6),
                    category=test_data["category"],
                    description=test_data.get("description", ""),
                )

                self.collection.add_test(test)

            logger.info(f"Loaded {self.collection.total_tests} test queries from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load test queries from {file_path}", error=str(e))
            return False

    async def save_to_file(self, file_path: Path) -> bool:
        """Save test queries to a JSON file."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(file_path.parent, exist_ok=True)

            # Create output data structure
            output = {
                "categories": [
                    {
                        "name": category.name,
                        "description": category.description,
                        "min_coverage": category.min_coverage,
                    }
                    for category in self.collection.categories.values()
                ],
                "tests": [
                    {
                        "id": test.id,
                        "query_text": test.query_text,
                        "expected_work_items": test.expected_work_items,
                        "minimum_similarity": test.minimum_similarity,
                        "category": test.category,
                        "description": test.description,
                    }
                    for test in self.collection.get_all_tests()
                ],
            }

            with open(file_path, "w") as f:
                json.dump(output, f, indent=2)

            logger.info(f"Saved {self.collection.total_tests} test queries to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save test queries to {file_path}", error=str(e))
            return False

    async def update_expected_results(self, jira_client: MCPClient, limit: int = 5):
        """Update expected results for tests that don't have them."""
        tests_to_update = [
            test for test in self.collection.get_all_tests() if not test.expected_work_items
        ]

        if not tests_to_update:
            logger.info("No tests need expected result updates")
            return

        logger.info(f"Updating expected results for {len(tests_to_update)} tests")

        # Get expected results from Jira using JQL
        for test in tests_to_update:
            try:
                # Create a JQL query based on keywords from the test query
                keywords = test.query_text.split()

                # Take up to 3 most meaningful keywords
                search_keywords = [kw for kw in keywords if len(kw) > 3][:3]

                if search_keywords:
                    search_query = " OR ".join([f'text ~ "{kw}"' for kw in search_keywords])
                    jql = f"({search_query}) ORDER BY created DESC"

                    # Execute JQL query
                    results = await jira_client.search(jql=jql, limit=limit)

                    # Extract keys from results
                    expected_keys = [item["key"] for item in results.get("issues", [])]

                    if expected_keys:
                        test.expected_work_items = expected_keys
                        logger.debug(
                            f"Updated test {test.id} with {len(expected_keys)} expected results"
                        )
                    else:
                        logger.debug(f"No results found for test {test.id}")

            except Exception as e:
                logger.error(f"Failed to update expected results for test {test.id}", error=str(e))

    async def generate_test_from_epic(
        self, epic_key: str, jira_client: MCPClient, claude_client: ClaudeClient
    ) -> RagTestQuery | None:
        """Generate a test query from an Epic."""
        try:
            # Get Epic details
            epic_data = await jira_client.get_issue(epic_key)

            if not epic_data:
                logger.error(f"Failed to get Epic data for {epic_key}")
                return None

            # Use Claude to generate a search query from the Epic description
            prompt = f"""
            Based on this Epic description, create a semantic search query that should find related work items:

            Epic: {epic_data.get('summary', '')}
            Description: {epic_data.get('description', '')}

            Generate a natural language query (2-8 words) that captures the main theme.
            """

            response = await claude_client.generate_text(prompt)
            query = response.strip()

            # Get all work items in the Epic as expected results
            epic_work_items = await jira_client.get_epic_issues(epic_key)
            work_item_keys = [item["key"] for item in epic_work_items]

            test = RagTestQuery(
                id=f"epic_{epic_key.lower()}",
                query_text=query,
                expected_work_items=work_item_keys,
                minimum_similarity=0.5,  # Lower threshold for generated queries
                category="generated",
                description=f"Generated from Epic {epic_key}",
            )

            # Add to collection
            self.collection.add_test(test)

            logger.info(f"Generated test query from Epic {epic_key}: '{query}'")
            return test

        except Exception as e:
            logger.error(f"Failed to generate test from Epic {epic_key}", error=str(e))
            return None

    def get_test_queries(self, category: str | None = None) -> list[RagTestQuery]:
        """Get test queries, optionally filtered by category."""
        if category:
            return self.collection.get_tests_by_category(category)
        return self.collection.get_all_tests()
