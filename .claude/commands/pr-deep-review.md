# PR Deep Review

Performs deep AI-powered validation of PR review comments by launching parallel sub-agents to investigate each issue.

## Usage
```
/pr-deep-review
/pr-deep-review <pr_number>
```

If no PR number is provided, uses the current branch's PR.

## Prerequisites

- `gh` - GitHub CLI for PR operations
- `git` - Version control operations

## Purpose
Takes PR review comments and launches parallel expert agents to:
- Deep dive into relevant files for each issue
- Validate if the reviewer's concern is correct
- Recommend whether to address it and why
- Return findings for user decision

## What This Command Does

### Phase 1: Collect PR Issues
1. **Get PR Number**: From argument or current branch
2. **Collect Comments**: Run `gh pr view <pr> --json comments,reviews`
3. **Parse Issues**: Extract all review comments and categorize
4. **Confirm Scope**: Show count and ask "Launching N parallel agents to investigate these issues. Continue?"

### Phase 2: Parallel Deep Investigation
5. **Launch Sub-Agents**: For each issue, create Task agent with:
   - **Context**: Full list of ALL issues in PR for awareness
   - **Focus**: Single specific issue to investigate
   - **Mission**:
     - Read relevant files mentioned in issue
     - Understand the code context
     - Validate if reviewer's concern is correct
     - Assess severity and impact
     - Recommend: ADDRESS (with reason) or SKIP (with reason)
   - **Return**: Structured recommendation

6. **Execute in Parallel**: Launch all Task agents simultaneously (single message, multiple tool calls)

### Phase 3: Collect and Synthesize
7. **Gather Results**: Collect all agent recommendations
8. **Present Summary**: Format as numbered action list:
   ```
   PR #[number]: Deep Review Analysis
   Analyzed X issues with parallel expert agents

   🟢 RECOMMENDED TO ADDRESS (X items)
   [1] Original Issue - file.ext:line
       Reviewer: @name: "quote"
       Agent Finding: [validation of issue]
       Recommendation: Address because [specific reasoning]
       Estimated Effort: ~X min

   🟡 UNCERTAIN / NEEDS DISCUSSION (X items)
   [N] Original Issue - file.ext:line
       Agent Finding: [concerns or uncertainties]
       Recommendation: Discuss with team because [reasoning]

   🔴 RECOMMENDED TO SKIP (X items)
   [N] Original Issue - file.ext:line
       Agent Finding: [why issue is not valid/relevant]
       Recommendation: Skip because [specific reasoning]

   → Which should I fix? Reply with numbers (e.g. "1,4", "all recommended", or "none"):
   ```

### Phase 4: USER SELECTION HANDLING (ONLY after user responds)
9. **STOP AND WAIT** for user selection
10. **ONLY AFTER user responds**:
   - Parse selection (e.g., "1,4", "all recommended", "none")
   - Create TodoWrite tasks for selected items
   - Begin implementation with task tracking

## Implementation Details

### Phase 1: Collect Comments
```bash
# Get PR number
if [[ -z "$PR_NUMBER" ]]; then
  BRANCH=$(git branch --show-current)
  PR_NUMBER=$(gh pr list --head "$BRANCH" --json number --jq '.[0].number')
fi

# Get all comments and reviews
gh pr view $PR_NUMBER --json comments,reviews,url,title

# Get PR diff for context
gh pr diff $PR_NUMBER
```

### Phase 2: Parse and Prepare
Extract comments into structured format:
```json
{
  "issue_id": 1,
  "reviewer": "@alice",
  "comment": "Need to sanitize user input",
  "file": "src/mimir/search.py",
  "line": 45,
  "type": "security"
}
```

### Phase 3: Launch Parallel Agents
For each issue, launch a Task agent with this prompt:

```markdown
# PR Issue Deep Investigation

## Context: All PR Issues
[Provide summary of ALL issues in the PR]

## Your Focus: Issue #[N]
**Original Comment**: @reviewer: "[exact quote]"
**File**: [file path]:[line]
**Type**: [issue type]

## Your Mission
1. **Read Files**: Examine the files mentioned in this issue
2. **Understand Context**: Look at surrounding code, related functions, imports
3. **Validate Concern**: Is the reviewer's concern technically correct?
4. **Assess Impact**: If correct, how severe is the issue?
5. **Recommendation**: Should we address this? Why or why not?

## Required Output Format
Return a structured analysis with:
- **is_valid**: true/false - Is the concern technically correct?
- **severity**: critical/high/medium/low/not-applicable
- **recommendation**: address/discuss/skip
- **reasoning**: Detailed explanation of findings
- **effort_estimate**: Time in minutes
- **additional_concerns**: Any related issues discovered

## Investigation Rules
- **Be Critical**: Don't assume reviewer is always right
- **Be Thorough**: Check related code, tests, dependencies
- **Be Specific**: Reference line numbers and exact code
- **Be Honest**: If uncertain, recommend "discuss" not "address" or "skip"
- **Consider Context**: Is this consistent with existing patterns?
```

**CRITICAL**: Launch all agents in parallel (single message with multiple Task tool calls)

### Phase 4: Synthesize Results
Collect agent responses and categorize:
- **Address**: Agent confirmed issue is valid and should be fixed
- **Discuss**: Agent is uncertain or sees tradeoffs
- **Skip**: Agent determined issue is not valid or not applicable

Present in structured format with reasoning from each agent.

## Sub-Agent Prompt Template

```markdown
# PR Issue Deep Investigation

## Context: All PR Issues in PR #[number]
[Numbered list of all issues for context]

## Your Focus: Issue #[N]
**Reviewer**: @[name]
**Comment**: "[exact quote]"
**File**: [path]:[line]
**Type**: [category]

## Your Mission
Investigate this specific issue and determine if it should be addressed.

### Steps
1. Read the file mentioned in the comment
2. Understand the surrounding code context
3. Check for related code (tests, similar patterns, imports)
4. Validate if the reviewer's concern is technically correct
5. Assess the severity and impact if concern is valid
6. Make a clear recommendation with reasoning

### Output Format
Provide your analysis in this format:

**Is Valid**: [true/false]
**Severity**: [critical/high/medium/low/not-applicable]
**Recommendation**: [address/discuss/skip]

**Reasoning**:
[Detailed explanation of your findings, referencing specific code and line numbers]

**Effort Estimate**: ~[X] minutes

**Additional Concerns**:
[Any related issues you discovered while investigating]

### Investigation Principles
- Be skeptical - verify the concern is real
- Be thorough - check related code and tests
- Be specific - cite exact code and locations
- Be honest - say "discuss" if uncertain
- Consider codebase patterns and consistency
```

## Selection Options
- `1,4` - Fix specific items by number
- `all` - Fix everything
- `all recommended` - Fix only items agents recommend addressing
- `all uncertain` - Fix items needing discussion
- `none` or `skip` - Don't fix anything

## Example Interaction

```
User: /pr-deep-review 6

Claude: Found 4 review comments in PR #6
        Launching 4 parallel expert agents to investigate...

        [agents investigate in parallel - ~1-2 minutes]

        PR #6: Deep Review Analysis
        Analyzed 4 issues with parallel expert agents

        🟢 RECOMMENDED TO ADDRESS (2 items)
        [1] Security - .github/workflows/slack-message.yml:52
            Reviewer: @alice: "Bot token should not be in plaintext"
            Agent Finding: Reviewed workflow file. Token is already using GitHub
                          Secrets (${{ secrets.SLACK_BOT_TOKEN }}), not plaintext.
                          However, validation check at line 48 could be improved.
            Recommendation: Address validation improvement, clarify to reviewer that
                          secrets are properly configured.
            Estimated Effort: ~10 min

        [2] Error Handling - .github/workflows/slack-message.yml:168
            Reviewer: @bob: "Missing error handling for empty version"
            Agent Finding: Confirmed - pyproject.toml parsing has no validation.
                          If file is malformed, grep will return empty string and
                          workflow will fail silently.
            Recommendation: Address - add validation check before version operations
            Estimated Effort: ~15 min

        🟡 UNCERTAIN / NEEDS DISCUSSION (1 item)
        [3] Testing - .github/workflows/notify-slack.yml:8
            Agent Finding: workflow_dispatch is present for manual testing, but
                          no documented test procedure. Unclear if reviewer wants
                          automated tests or just documentation.
            Recommendation: Discuss - clarify what level of testing is expected

        🔴 RECOMMENDED TO SKIP (1 item)
        [4] Documentation - README.md
            Agent Finding: README already documents required secrets in the
                          "Setup" section (lines 34-42). Reviewer may have missed it.
            Recommendation: Skip - documentation already exists, just point
                          reviewer to existing section

        → Which should I fix? Reply with numbers (e.g. "1,4", "all recommended", or "none"):

[STOPS AND WAITS HERE]

User: all recommended

Claude: Creating tasks for 2 recommended fixes...
        [uses TodoWrite to create tasks]
        [begins implementation]
```

## Performance Considerations
- Agents run in parallel (should complete in ~1-2 minutes total)
- Each agent has access to Read, Grep, Glob tools
- Agents can discover related files independently
- Single message with multiple Task tool calls for true parallelism

## Agent Configuration
- **Agent Type**: `subagent_type: "general-purpose"`
- **Tools Available**: Read, Grep, Glob, Bash (for inspection only)
- **Parallel Execution**: All agents launched in single message

## Error Handling
- If any agent fails, continue with others and note the failure
- Present partial results if some agents succeed
- If all agents fail, fall back to `/pr-analyze-comments` approach

## When to Use This vs pr-analyze-comments
- **pr-analyze-comments**: Fast, trust reviewers, implement suggestions directly
- **pr-deep-review**: Validate feedback first, especially for:
  - Large PRs with many comments
  - Conflicting reviewer opinions
  - Comments that might be outdated
  - When you're unsure if suggestions are correct
  - Complex technical concerns requiring investigation

## Safety Considerations
- **Read-Only Investigation**: Agents only read files, no modifications
- **User Control**: Always requires explicit selection before fixing
- **Parallel Safety**: Agents work independently on separate issues
- **GitHub CLI**: Uses official `gh` tool for all operations

## Integration with Mímir Workflow
- **GT Integration**: Works with current branch PRs
- **Task Tracking**: Creates TodoWrite tasks for selected items
- **AI Validation**: Uses sub-agents for intelligent analysis

## Related Commands
- `/pr-analyze-comments` - Quick analysis without deep investigation
- `/commit` - Commit fixes after implementation
- `/pr-request-review` - Request re-review after fixes

## Notes
- Uses GitHub CLI (`gh`) for all GitHub operations
- Launches parallel Task agents for deep investigation
- Always waits for user selection before implementing fixes
- Provides AI-validated recommendations with reasoning
- More thorough but slower than `/pr-analyze-comments`
