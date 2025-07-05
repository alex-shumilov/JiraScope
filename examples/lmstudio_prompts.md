# LMStudio + JiraScope: Prompt Patterns & Usage Examples

This guide provides proven prompt patterns for getting the most out of JiraScope's AI-powered Jira analysis through LMStudio integration.

## 🚀 Quick Start

Once JiraScope MCP server is connected to LMStudio, you can use natural language to analyze your Jira data. The AI will automatically call the appropriate tools based on your request.

## 📊 Core Analysis Patterns

### Technical Debt Analysis

**Pattern**: Ask about code quality, maintenance, or architectural issues

```
Find technical debt in our authentication system components
```

```
What technical debt patterns exist in frontend code from the last quarter?
```

```
Show me all refactoring tasks that have been delayed or deprioritized
```

**Expected Tools**: `analyze_technical_debt`

### Scope Drift Detection

**Pattern**: Analyze how Epic requirements have changed over time

```
Analyze scope drift for Epic MOBILE-247
```

```
What changes have occurred in Epic AUTH-156 since it started?
```

```
Show me how the requirements for Epic PAYMENTS-89 have evolved
```

**Expected Tools**: `detect_scope_drift`

### Dependency Mapping

**Pattern**: Understand blockers and cross-team dependencies

```
Map all dependencies for the mobile team's current sprint
```

```
What's blocking Epic INFRASTRUCTURE-45?
```

```
Show cross-team dependencies that might affect our Q1 delivery
```

**Expected Tools**: `map_dependencies`

### General Search & Discovery

**Pattern**: Use natural language to find specific issues or patterns

```
Find all high-priority bugs assigned to the frontend team
```

```
Show me performance-related issues from the last 30 days
```

```
What security vulnerabilities were reported this quarter?
```

**Expected Tools**: `search_jira_issues`

## 🎯 Sprint Planning Workflows

### Sprint Capacity Planning

```
Help me plan our next sprint. Analyze the mobile team's backlog and identify:
- High-priority items
- Dependencies that might block progress
- Technical debt that should be addressed
- Estimated complexity based on similar past work
```

### Risk Assessment

```
What risks should we consider for our upcoming sprint? Look for:
- Items with unclear requirements
- Dependencies on other teams
- Technical debt that might slow us down
- Historical patterns of scope creep
```

### Team Coordination

```
Identify cross-team coordination needs for Epic PLATFORM-123:
- Which teams are involved?
- What are the key handoff points?
- Are there any blocking dependencies?
```

## 🔍 Deep Analysis Patterns

### Root Cause Analysis

```
Analyze the root causes of delays in Epic CUSTOMER-456:
- What scope changes occurred?
- Were there unexpected dependencies?
- What technical challenges emerged?
```

### Quality Trends

```
Analyze quality trends for the payments component:
- Bug introduction rates
- Technical debt accumulation
- Test coverage gaps
- Performance issues
```

### Team Performance Insights

```
How is the authentication team performing this quarter?
- Velocity trends
- Technical debt management
- Cross-team collaboration patterns
- Quality metrics
```

## 🛠️ Advanced Use Cases

### Architecture Planning

```
Based on our current technical debt analysis, what should be our architecture priorities for the next quarter? Consider:
- Most critical refactoring needs
- Components with highest maintenance burden
- Dependencies that create bottlenecks
```

### Release Planning

```
Analyze readiness for our Q2 release:
- Outstanding high-priority issues
- Technical debt that could impact stability
- Dependencies between features
- Risk factors and mitigation strategies
```

### Onboarding New Team Members

```
Create an onboarding guide for new developers joining the frontend team:
- Current technical debt they should be aware of
- Key components and their dependencies
- Common issues and their solutions
- Areas needing immediate attention
```

## 💡 Pro Tips

### Combining Multiple Perspectives

Ask for comprehensive analysis that uses multiple tools:

```
Give me a complete picture of Epic MOBILE-789:
- Current scope and any drift that has occurred
- Technical debt in related components
- Dependencies and potential blockers
- Risk assessment for on-time delivery
```

### Historical Context

Include time ranges for better insights:

```
Compare technical debt levels between Q3 and Q4 for the API team
```

### Specific Filtering

Be specific about what you want to focus on:

```
Find authentication-related technical debt that's marked as high priority
```

### Actionable Recommendations

Ask for concrete next steps:

```
Based on our current technical debt analysis, what are the top 3 items we should address this sprint to improve team velocity?
```

## 🚨 Common Patterns to Avoid

### Too Vague
❌ "Tell me about our project"
✅ "Analyze technical debt in the user authentication components"

### Too Broad
❌ "Show me everything"
✅ "Find high-priority bugs in frontend components from the last month"

### Missing Context
❌ "What's wrong with Epic ABC-123?"
✅ "Analyze scope drift and dependencies for Epic ABC-123"

## 🔧 Troubleshooting

### If Tools Aren't Being Called

1. **Be more specific**: Instead of "How's the project?", try "Analyze technical debt in payment processing"
2. **Use keywords**: Include terms like "technical debt", "dependencies", "scope drift", or "search"
3. **Mention specific components**: Reference team names, Epic keys, or component names

### If Results Are Too Generic

1. **Add constraints**: Specify time ranges, teams, or priorities
2. **Request specific analysis**: Ask for patterns, trends, or recommendations
3. **Follow up with details**: "Can you drill down into the authentication issues you found?"

### If You Need Different Information

1. **Rephrase your request**: Different wording might trigger different tools
2. **Break down complex requests**: Ask for one analysis at a time
3. **Use follow-up questions**: Build on previous results with more specific queries

## 📚 Integration Examples

### With Project Management

```
Generate a project status report for Epic PLATFORM-567:
- Scope changes and timeline impact
- Technical risks and dependencies
- Recommendations for stakeholder communication
```

### With Architecture Reviews

```
Prepare for our architecture review meeting:
- Most critical technical debt requiring architect input
- Cross-component dependencies needing design decisions
- Performance and scalability concerns from recent issues
```

### With Code Reviews

```
Before our next code review, identify:
- Components with highest technical debt
- Areas where patterns could be improved
- Dependencies that might affect code organization
```

This prompt guide will help you get the most value from JiraScope's AI-powered analysis through natural, conversational interactions with LMStudio.
