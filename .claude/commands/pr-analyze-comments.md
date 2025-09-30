# Analyze PR Comments

Analyzes PR review comments and presents numbered action items that you can easily select to fix.

## Usage
```
/pr-analyze-comments
/pr-analyze-comments <pr_number>
```

If no PR number is provided, uses the current branch's PR.

## Prerequisites

- `gh` - GitHub CLI for PR operations
- `git` - Version control operations

## Purpose
Gets PR review comments and presents them as numbered action items so you can say "fix 1,5,9" etc.

## What This Command Does

1. Gets PR number (from argument or current branch)
2. Collects all review comments from GitHub
3. Analyzes what needs to be done
4. Presents as numbered list with priority markers
5. **STOPS AND WAITS** for user selection

## Implementation

### Phase 1: Determine PR Number
```bash
# If no PR number provided, get current branch's PR
if [[ -z "$PR_NUMBER" ]]; then
  BRANCH=$(git branch --show-current)
  PR_NUMBER=$(gh pr list --head "$BRANCH" --json number --jq '.[0].number')
fi
```

### Phase 2: Collect PR Data
```bash
# Get PR basic info
gh pr view $PR_NUMBER --json number,title,state,author,url

# Get all review comments
gh pr view $PR_NUMBER --json comments,reviews

# Get PR diff for context
gh pr diff $PR_NUMBER
```

### Phase 3: Analyze and Categorize
Parse comments and categorize by:
- **Priority**: Critical/Blocking, Important, Nice-to-have
- **Type**: Bug fix, Performance, Security, Style, Documentation, Testing
- **Effort**: Estimate time to fix (~5min, ~15min, ~30min, etc.)
- **Files**: Which files need changes

Look for keywords to determine priority:
- **Critical/Blocking**: "blocking", "breaks", "critical", "must fix", "security"
- **Important**: "should", "need to", "important", "bug"
- **Nice-to-have**: "could", "might", "consider", "nit", "suggestion"

### Phase 4: Present Formatted List
```
PR #[number]: [title]
Status: X outstanding issues, Y comments total

🔴 CRITICAL/BLOCKING (X items) - Must fix before merge
[1] Issue Type - file.ext:line
    @reviewer: "exact quote from comment"
    Context: explanation why this matters
    Fix: specific implementation steps
    Effort: ~X min | Files: path/to/files

🟡 IMPORTANT (X items) - Should fix
[N] Issue Type - file.ext:line
    @reviewer: "exact quote"
    Context: explanation
    Fix: implementation steps
    Effort: ~X min | Files: paths

🔵 NICE TO HAVE (X items) - Optional improvements
[N] Issue Type - file.ext:line
    @reviewer: "exact quote"
    Fix: implementation steps
    Effort: ~X min | Files: paths

Total estimated effort: ~X minutes
→ Which should I fix? Reply with numbers (e.g. "1,4", "all critical", or "all"):
```

### Phase 5: STOP AND WAIT
**⚠️ CRITICAL: DO NOT proceed past this point**
- Display the formatted list
- Show the prompt asking which items to fix
- **WAIT for user response**
- **DO NOT automatically select or start fixing**

### Phase 6: USER SELECTION HANDLING (ONLY after user responds)
**ONLY AFTER user provides selection** (e.g. "1,4", "all critical"):
1. Parse their selection
2. Create TodoWrite tasks for selected items
3. Begin implementation with task tracking

If user says "none" or "skip", acknowledge and stop.

## Selection Options
- `1,4` - Fix specific items by number
- `all` - Fix everything
- `all critical` - Fix all critical/blocking items
- `all important` - Fix all important items
- `1-3` - Fix range of items 1 through 3
- `none` or `skip` - Don't fix anything

## Priority Markers
- 🔴 Critical/Blocking - Must fix before merge
- 🟡 Important/Should fix - Should address
- 🔵 Nice-to-have - Optional improvements

## Example Interaction Flow

```
User: /pr-analyze-comments 6

Claude: [collects and analyzes PR data]

PR #6: Add Slack notifications for releases-dev channel
Status: 4 outstanding issues, 6 comments total

🔴 CRITICAL/BLOCKING (1 item) - Must fix before merge
[1] Security - .github/workflows/slack-message.yml:52
    @alice: "Bot token should not be in plaintext"
    Context: Security best practice - secrets should use GitHub Secrets
    Fix: Move SLACK_BOT_TOKEN to GitHub Secrets if not already there
    Effort: ~5 min | Files: .github/workflows/slack-message.yml

🟡 IMPORTANT (2 items) - Should fix
[2] Error Handling - .github/workflows/slack-message.yml:168
    @bob: "Missing error handling for empty version"
    Context: Could fail if pyproject.toml is malformed
    Fix: Add validation check before accessing version
    Effort: ~10 min | Files: .github/workflows/slack-message.yml

[3] Testing - .github/workflows/notify-slack.yml:8
    @carol: "Should test workflow_dispatch trigger"
    Context: Manual testing capability not verified
    Fix: Document testing steps or add workflow test
    Effort: ~15 min | Files: docs or test workflow

🔵 NICE TO HAVE (1 item) - Optional improvements
[4] Documentation - README.md
    @dave: "Could document Slack setup in README"
    Fix: Add section explaining Slack secret configuration
    Effort: ~20 min | Files: README.md

Total estimated effort: ~50 minutes
→ Which should I fix? Reply with numbers (e.g. "1,4", "all critical", or "all"):

[STOPS AND WAITS HERE]

User: all critical

Claude: Creating task for item 1...
[uses TodoWrite to create tasks]
[begins implementation]
```

## Comment Analysis Guidelines

### Identifying Issue Type
- **Security**: "token", "secret", "authentication", "authorization", "XSS", "injection"
- **Bug**: "error", "bug", "broken", "fails", "incorrect", "wrong"
- **Performance**: "slow", "performance", "optimize", "cache", "bottleneck"
- **Testing**: "test", "coverage", "spec", "should test"
- **Documentation**: "docs", "readme", "comment", "documentation"
- **Style**: "formatting", "style", "naming", "convention", "nit"

### Estimating Effort
- **~5 min**: Simple changes (variable rename, add comment, fix typo)
- **~10 min**: Small fixes (add validation, simple error handling)
- **~15 min**: Medium changes (refactor function, add test case)
- **~30 min**: Larger changes (new feature addition, significant refactor)
- **~1 hour+**: Complex changes (architecture changes, multiple files)

## Error Handling

### No PR Found
```
Error: No PR found for current branch
Run: /pr-analyze-comments <PR_NUMBER>
```

### No Comments
```
No review comments found on PR #6
PR appears to be ready for merge!
```

### GitHub API Errors
If `gh` commands fail:
```
Error: GitHub authentication required
Run: gh auth login
```

## Safety Considerations

- **Read-Only Analysis**: No modifications until user confirms
- **GitHub CLI**: Uses official `gh` tool for all operations
- **User Control**: Always requires explicit selection before fixing
- **Task Tracking**: Uses TodoWrite for visibility

## Integration with Mímir Workflow

- **GT Integration**: Works with current branch PRs
- **Task Tracking**: Creates TodoWrite tasks for selected items
- **Structured Approach**: Methodical fixes with progress tracking

## Related Commands
- `/pr-deep-review` - Deep AI validation of issues before fixing
- `/commit` - Commit fixes after implementation
- `/pr-request-review` - Request re-review after fixes

## Notes
- Uses GitHub CLI (`gh`) for all GitHub operations
- Parses and categorizes comments locally (no external tools needed)
- Always waits for user selection before implementing fixes
- Creates TodoWrite tasks for transparency and tracking
