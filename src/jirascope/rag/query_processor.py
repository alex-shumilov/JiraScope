"""Query understanding and expansion for Jira-specific queries."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass
class FilterSet:
    """Structured filters extracted from natural language query."""

    item_types: list[str] | None = None
    statuses: list[str] | None = None
    priorities: list[str] | None = None
    components: list[str] | None = None
    labels: list[str] | None = None
    teams: list[str] | None = None
    epic_keys: list[str] | None = None
    assignees: list[str] | None = None
    date_range: dict[str, datetime] | None = None
    sprint_names: list[str] | None = None

    def __post_init__(self):
        """Initialize list fields if None."""
        if self.item_types is None:
            self.item_types = []
        if self.statuses is None:
            self.statuses = []
        if self.priorities is None:
            self.priorities = []
        if self.components is None:
            self.components = []
        if self.labels is None:
            self.labels = []
        if self.teams is None:
            self.teams = []
        if self.epic_keys is None:
            self.epic_keys = []
        if self.assignees is None:
            self.assignees = []
        if self.sprint_names is None:
            self.sprint_names = []

    def to_qdrant_filter(self) -> dict[str, Any]:
        """Convert to Qdrant filter format."""
        filters = {}

        if self.item_types:
            filters["item_type"] = self.item_types
        if self.statuses:
            filters["status"] = self.statuses
        if self.priorities:
            filters["priority"] = self.priorities
        if self.components:
            filters["components"] = self.components
        if self.labels:
            filters["labels"] = self.labels
        if self.teams:
            filters["team"] = self.teams
        if self.epic_keys:
            filters["epic_key"] = self.epic_keys
        if self.assignees:
            filters["assignee"] = self.assignees
        if self.sprint_names:
            filters["sprint_names"] = self.sprint_names

        return filters


@dataclass
class ExpandedQuery:
    """Query expanded with synonyms and related terms."""

    original_query: str
    expanded_terms: list[str]
    semantic_variants: list[str]
    jira_synonyms: dict[str, list[str]]

    @property
    def full_query_text(self) -> str:
        """Combined query text with expansions."""
        all_terms = [self.original_query, *self.expanded_terms, *self.semantic_variants]
        return " ".join(all_terms)


@dataclass
class QueryPlan:
    """Complete query understanding and execution plan."""

    original_query: str
    expanded_query: ExpandedQuery
    filters: FilterSet
    intent: str  # search, analysis, report, etc.
    expected_output: str  # list, summary, analysis, etc.
    priority_boost: dict[str, float] | None = None

    def __post_init__(self):
        """Initialize optional fields."""
        if self.priority_boost is None:
            self.priority_boost = {}


class JiraQueryProcessor:
    """Processes natural language queries for Jira search and analysis."""

    def __init__(self):
        self.jira_synonyms = self._load_jira_synonyms()
        self.time_patterns = self._load_time_patterns()

    def analyze_query(self, user_query: str) -> QueryPlan:
        """
        Understand query intent and extract structured information.

        Args:
            user_query: Natural language query from user

        Returns:
            QueryPlan with structured filters and expanded query
        """
        # Extract filters first
        filters = self.extract_filters(user_query)

        # Expand query with synonyms and related terms
        expanded_query = self.expand_query(user_query)

        # Determine intent and expected output
        intent = self._determine_intent(user_query)
        expected_output = self._determine_output_format(user_query)

        # Calculate priority boosts based on query terms
        priority_boost = self._calculate_priority_boost(user_query, filters)

        return QueryPlan(
            original_query=user_query,
            expanded_query=expanded_query,
            filters=filters,
            intent=intent,
            expected_output=expected_output,
            priority_boost=priority_boost,
        )

    def expand_query(self, original_query: str) -> ExpandedQuery:
        """
        Expand query with synonyms and related terms.

        Args:
            original_query: Original user query

        Returns:
            ExpandedQuery with expanded terms
        """
        expanded_terms = []
        semantic_variants = []
        used_synonyms = {}

        # Add Jira-specific synonyms
        for term, synonyms in self.jira_synonyms.items():
            if term.lower() in original_query.lower():
                expanded_terms.extend(synonyms)
                used_synonyms[term] = synonyms

        # Add semantic variants for common concepts
        semantic_variants.extend(self._get_semantic_variants(original_query))

        return ExpandedQuery(
            original_query=original_query,
            expanded_terms=expanded_terms,
            semantic_variants=semantic_variants,
            jira_synonyms=used_synonyms,
        )

    def extract_filters(self, query: str) -> FilterSet:
        """
        Extract structured filters from natural language.

        Args:
            query: Natural language query

        Returns:
            FilterSet with extracted filters
        """
        filters = FilterSet()

        # Extract item types
        filters.item_types = self._extract_item_types(query)

        # Extract statuses
        filters.statuses = self._extract_statuses(query)

        # Extract priorities
        filters.priorities = self._extract_priorities(query)

        # Extract components and teams
        filters.components = self._extract_components(query)
        filters.teams = self._extract_teams(query)

        # Extract time-based filters
        filters.date_range = self._extract_date_range(query)
        filters.sprint_names = self._extract_sprints(query)

        # Extract assignees
        filters.assignees = self._extract_assignees(query)

        # Extract Epic keys
        filters.epic_keys = self._extract_epic_keys(query)

        return filters

    def _extract_item_types(self, query: str) -> list[str]:
        """Extract Jira item types from query."""
        types = []
        type_patterns = {
            "Story": r"\b(story|stories|user story)\b",
            "Task": r"\b(task|tasks)\b",
            "Bug": r"\b(bug|bugs|defect|defects|issue)\b",
            "Epic": r"\b(epic|epics)\b",
        }

        for item_type, pattern in type_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                types.append(item_type)

        return types

    def _extract_statuses(self, query: str) -> list[str]:
        """Extract status filters from query."""
        statuses = []
        status_patterns = {
            "Open": r"\b(open|new|created)\b",
            "In Progress": r"\b(in progress|active|working|started)\b",
            "Done": r"\b(done|completed|finished|closed|resolved)\b",
            "Blocked": r"\b(blocked|waiting|on hold)\b",
        }

        for status, pattern in status_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                statuses.append(status)

        return statuses

    def _extract_priorities(self, query: str) -> list[str]:
        """Extract priority filters from query."""
        priorities = []
        priority_patterns = {
            "High": r"\b(high|urgent|critical|important)\b",
            "Medium": r"\b(medium|normal|standard)\b",
            "Low": r"\b(low|minor|trivial)\b",
        }

        for priority, pattern in priority_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                priorities.append(priority)

        return priorities

    def _extract_components(self, query: str) -> list[str]:
        """Extract component filters from query."""
        components = []
        component_map = {
            "frontend": ["frontend", "front-end", "ui"],
            "backend": ["backend", "back-end", "api"],
            "mobile": ["mobile", "ios", "android"],
            "platform": ["platform", "infrastructure"],
        }

        for component, terms in component_map.items():
            for term in terms:
                if re.search(rf"\b{term}\b", query, re.IGNORECASE):
                    components.append(component)
                    break

        return list(set(components))

    def _extract_teams(self, query: str) -> list[str]:
        """Extract team filters from query."""
        teams = []
        team_map = {
            "Platform": ["platform team", "platform"],
            "Frontend": ["frontend team", "frontend"],
            "Backend": ["backend team", "backend"],
            "Mobile": ["mobile team", "mobile"],
        }

        for team, terms in team_map.items():
            for term in terms:
                if re.search(rf"\b{term}\b", query, re.IGNORECASE):
                    teams.append(team)
                    break

        return list(set(teams))

    def _extract_date_range(self, query: str) -> dict[str, datetime] | None:
        """Extract date range from time references."""
        now = datetime.now()

        for pattern, days_back in self.time_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                start_date = now - timedelta(days=days_back)
                return {"start": start_date, "end": now}

        return None

    def _extract_sprints(self, query: str) -> list[str]:
        """Extract sprint references from query."""
        sprints = []

        # Sprint pattern matching
        sprint_patterns = [r"sprint (\d+)", r"current sprint", r"last sprint", r"next sprint"]

        for pattern in sprint_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                if "current" in match.group().lower():
                    sprints.append("current")
                elif "last" in match.group().lower():
                    sprints.append("previous")
                elif "next" in match.group().lower():
                    sprints.append("next")
                else:
                    sprints.append(match.group())

        return sprints

    def _extract_assignees(self, query: str) -> list[str]:
        """Extract assignee references from query."""
        assignees = []

        # Assignee patterns
        assignee_patterns = [r"assigned to (\w+)", r"assignee (\w+)", r"@(\w+)"]

        for pattern in assignee_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            assignees.extend(match.group(1) for match in matches)

        return assignees

    def _extract_epic_keys(self, query: str) -> list[str]:
        """Extract Epic key references from query."""
        epic_keys = []

        # Epic key patterns (PROJ-123 format)
        epic_pattern = r"\b([A-Z]+-\d+)\b"
        matches = re.finditer(epic_pattern, query)

        epic_keys = [match.group(1) for match in matches]

        return epic_keys

    def _determine_intent(self, query: str) -> str:
        """Determine the intent of the query."""
        analysis_keywords = ["analyze", "analysis", "report", "summary", "trends", "patterns"]
        search_keywords = ["find", "search", "show", "list", "get"]

        query_lower = query.lower()

        if any(keyword in query_lower for keyword in analysis_keywords):
            return "analysis"
        if any(keyword in query_lower for keyword in search_keywords):
            return "search"
        return "search"  # Default to search

    def _determine_output_format(self, query: str) -> str:
        """Determine expected output format."""
        if any(word in query.lower() for word in ["summary", "report", "analysis"]):
            return "summary"
        if any(word in query.lower() for word in ["list", "show", "find"]):
            return "list"
        return "list"  # Default

    def _calculate_priority_boost(self, query: str, filters: FilterSet) -> dict[str, float]:
        """Calculate priority boosts based on query terms."""
        boosts = {}

        # Boost recent items for time-sensitive queries
        if any(word in query.lower() for word in ["urgent", "critical", "important"]):
            boosts["priority_boost"] = 1.5

        # Boost specific statuses
        if "blocked" in query.lower():
            boosts["status_boost"] = 2.0

        return boosts

    def _get_semantic_variants(self, query: str) -> list[str]:
        """Get semantic variants for query terms."""
        variants = []

        # Technical debt related terms
        if any(term in query.lower() for term in ["debt", "refactor", "cleanup"]):
            variants.extend(["technical debt", "refactoring", "code quality", "maintenance"])

        # Performance related terms
        if any(term in query.lower() for term in ["slow", "performance", "speed"]):
            variants.extend(["performance", "optimization", "latency", "speed"])

        # Bug related terms
        if any(term in query.lower() for term in ["bug", "error", "issue"]):
            variants.extend(["defect", "problem", "failure", "broken"])

        return variants

    def _load_jira_synonyms(self) -> dict[str, list[str]]:
        """Load Jira-specific terminology synonyms."""
        return {
            "story": ["user story", "feature", "requirement"],
            "bug": ["defect", "issue", "problem", "error"],
            "task": ["work item", "todo", "action item"],
            "epic": ["initiative", "theme", "project"],
            "blocked": ["waiting", "on hold", "dependency"],
            "done": ["completed", "finished", "resolved", "closed"],
            "in progress": ["active", "working", "started", "wip"],
            "high priority": ["urgent", "critical", "important"],
            "technical debt": ["refactoring", "cleanup", "maintenance"],
        }

    def _load_time_patterns(self) -> dict[str, int]:
        """Load time reference patterns with days back."""
        return {
            r"\blast week\b": 7,
            r"\blast month\b": 30,
            r"\blast quarter\b": 90,
            r"\bthis week\b": 7,
            r"\bthis month\b": 30,
            r"\bthis quarter\b": 90,
            r"\byesterday\b": 1,
            r"\btoday\b": 0,
            r"\brecent\b": 14,
            r"\blast sprint\b": 14,  # Assuming 2-week sprints
            r"\bcurrent sprint\b": 14,
        }
