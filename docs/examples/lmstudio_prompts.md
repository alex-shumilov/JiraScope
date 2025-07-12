# LMStudio Prompts for JiraScope

This guide provides comprehensive prompt patterns and examples for using JiraScope with LMStudio through the MCP integration.

## Getting Started

Once JiraScope MCP server is configured in LMStudio, you can interact with your Jira data using natural language. The AI has access to all your work items and can perform semantic search and analysis.

## Basic Search Patterns

### Simple Queries
```
Find high priority bugs in the authentication system
```
```
Show me all stories assigned to the mobile team
```
```
What work items were created last week?
```

### Filtered Searches
```
Find all open tasks related to performance optimization
```
```
Show me critical bugs that are still unresolved
```
```
List all work items in Epic PLATFORM-123
```

## Analysis Patterns

### Technical Debt Analysis
```
Analyze technical debt in our payment processing components and prioritize the most critical items
```
```
Find all refactoring tasks and group them by system component
```
```
What technical debt should we address this quarter based on impact and effort?
```

### Scope Drift Detection
```
Analyze Epic MOBILE-456 for scope drift and show what changed since initial planning
```
```
Which Epics have expanded beyond their original scope this quarter?
```
```
Show me work items that were added to Epic ABC-123 after it started
```

### Dependency Mapping
```
Map all dependencies for the Q2 release and identify potential blockers
```
```
What work items are currently blocked and what are they waiting for?
```
```
Show me cross-team dependencies between frontend and backend teams
```

### Quality Assessment
```
Evaluate the quality of work item descriptions in project PROJ and suggest improvements
```
```
Find work items with incomplete acceptance criteria
```
```
Which stories need better technical specifications?
```

## Advanced Query Patterns

### Sprint Planning
```
Help me plan the next sprint for the mobile team. Analyze their current velocity, identify available work items, and suggest a realistic sprint scope based on:
- Team capacity (5 developers, 2-week sprint)
- Current velocity (average 40 story points)
- Priority items from product backlog
- Any blockers or dependencies
```

### Risk Assessment
```
Perform a risk analysis for our Q3 release. Look for:
- Work items with unclear requirements
- Dependencies on external teams
- Items that have been in progress too long
- Scope changes in critical Epics
```

### Team Performance Analysis
```
Analyze the performance and patterns for the backend team over the last 3 months:
- Velocity trends
- Types of work (features vs bugs vs debt)
- Blocker patterns
- Quality metrics
```

### Epic Health Check
```
Give me a comprehensive health check for Epic PLATFORM-789:
- Scope stability
- Progress against timeline
- Child work item quality
- Dependencies and blockers
- Recommendation for course correction
```

## Specialized Prompts

### Duplicate Detection
```
Find potential duplicate work items across all projects. Focus on:
- Similar descriptions or titles
- Same functional requirements
- Redundant bug reports
- Overlapping feature requests
```

### Template Generation
```
Create a template for Story work items based on the highest quality examples from project PROJ. Include:
- Structure and format
- Required fields
- Acceptance criteria patterns
- Best practices
```

### Cross-Project Analysis
```
Compare work patterns between the iOS and Android projects:
- Common issues and bugs
- Feature parity gaps
- Different approaches to similar problems
- Opportunities for code sharing
```

### Technical Architecture Analysis
```
Analyze work items to understand our system architecture:
- Component relationships
- Data flow patterns
- Integration points
- Technical dependencies
```

## Workflow-Specific Prompts

### Daily Standup Support
```
Prepare a summary for today's standup meeting:
- What was completed yesterday?
- Any new blockers or issues?
- Work items at risk of missing deadlines
- Items needing team discussion
```

### Retrospective Analysis
```
Analyze the last sprint for retrospective insights:
- What types of work took longer than expected?
- Common causes of blockers
- Quality issues that led to rework
- Process improvements we could make
```

### Release Planning
```
Help plan the next release by analyzing:
- Feature completion status
- Critical bugs that must be fixed
- Technical debt that could impact release
- Dependencies between teams and components
```

### Onboarding New Team Members
```
Create an onboarding guide for new developers joining the authentication team:
- Recent work items in the authentication domain
- Common patterns and approaches
- Current technical debt and known issues
- Upcoming planned work
```

## Integration Patterns

### Combining Multiple Analyses
```
I need a comprehensive project health report. Please:
1. Analyze scope drift across all active Epics
2. Identify technical debt clusters by priority
3. Map current dependencies and blockers
4. Assess overall delivery risk for Q4 goals
5. Provide actionable recommendations
```

### Iterative Deep Dive
```
Let's do a deep dive on performance issues:
1. First, find all work items related to performance
2. Then analyze them for common patterns
3. Group by system component
4. Prioritize by user impact
5. Suggest a remediation plan
```

### Comparative Analysis
```
Compare our current sprint performance to historical averages:
1. Analyze velocity trends over the last 6 sprints
2. Identify factors that improved or hindered performance
3. Compare work item complexity and types
4. Recommend adjustments for next sprint
```

## Tips for Effective Prompts

### Be Specific
Instead of: "Find bugs"
Use: "Find critical priority bugs in the authentication system that were reported in the last 30 days"

### Provide Context
Instead of: "Analyze this Epic"
Use: "Analyze Epic MOBILE-123 for scope drift, focusing on work items added after the initial planning session"

### Ask for Actionable Output
Instead of: "Show me technical debt"
Use: "Identify technical debt items and prioritize them by impact, with specific recommendations for when to address each cluster"

### Use Multi-Step Requests
```
Please help me prepare for sprint planning:
1. Show me available work items prioritized by product owner
2. Estimate effort based on similar completed items
3. Identify any dependencies or blockers
4. Suggest a balanced sprint scope for our team of 6 developers
```

### Request Specific Formats
```
Create a summary table of all high-priority bugs with columns:
- Jira Key
- Component
- Days Open
- Assignee
- Impact Level
```

## Troubleshooting Prompts

### When Results Seem Incomplete
```
The previous search might have missed some items. Please search more broadly for work items related to [topic] and include:
- Different terminology that might be used
- Related components or systems
- Historical items that might provide context
```

### When You Need More Detail
```
Can you provide more detailed analysis of [specific item/Epic/component], including:
- Technical specifications
- Acceptance criteria
- Dependencies
- Historical context
- Similar items for comparison
```

### When Results Need Refinement
```
The previous results were helpful but too broad. Please narrow down to:
- Only items from the last 60 days
- Critical and high priority only
- Specific to the mobile platform
- Excluding completed items
```

## Best Practices

1. **Start Broad, Then Narrow**: Begin with general queries, then ask for more specific analysis
2. **Use Domain Language**: Include project names, component names, and team terminology
3. **Ask for Reasoning**: Request explanations for recommendations and analysis
4. **Iterate and Refine**: Build on previous responses to get exactly what you need
5. **Combine Perspectives**: Ask for analysis from different viewpoints (technical, business, timeline)

## Example Conversation Flow

```
User: "Help me understand the current state of our mobile project"

AI: [Provides overview of mobile project work items, recent activity, status summary]

User: "What are the biggest risks to our Q4 mobile release?"

AI: [Analyzes dependencies, scope changes, blockers, technical debt]

User: "Focus on the authentication integration Epic - what specific issues should I be concerned about?"

AI: [Deep dive on authentication Epic with specific risk factors and recommendations]

User: "Create an action plan to mitigate these risks"

AI: [Provides prioritized action items with timeline and ownership suggestions]
```

This conversational approach leverages JiraScope's semantic understanding to provide increasingly focused and actionable insights about your Jira data.
