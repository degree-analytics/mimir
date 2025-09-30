# Request Code Review Command

Generate comprehensive PR review requests with detailed analysis for consistent, high-quality GitHub comments that guide reviewers effectively.

## Usage
```
/pr-request-review
/pr-request-review <PR_NUMBER>
```

If no PR number is provided, uses the current branch's PR.

## Prerequisites

- `gh` - GitHub CLI for PR operations
- `git` - Version control operations

## Purpose
Automatically analyze a PR and post a structured review request comment that helps reviewers understand:
- What changed and why
- Where to focus their attention
- Key areas of concern or complexity
- Context needed for effective review

## Implementation

### Phase 1: Determine PR Number
```bash
# If no PR number provided, get current branch's PR
if [[ -z "$PR_NUMBER" ]]; then
  BRANCH=$(git branch --show-current)
  PR_NUMBER=$(gh pr list --head "$BRANCH" --json number --jq '.[0].number')
fi
```

### Phase 2: Validate and Collect PR Data
```bash
# 1. Validate PR exists and get basic info
gh pr view $PR_NUMBER --json number,title,state,author,headRefName,baseRefName,url

# 2. Get PR commits for change analysis
gh pr view $PR_NUMBER --json commits

# 3. Get PR diff for file changes
gh pr diff $PR_NUMBER
```

### Phase 3: Analyze Changes

Analyze the PR to understand:
- Primary changes made (from commit messages and diff)
- Files modified and their significance
- Test coverage changes
- Risk areas that need review attention

### Phase 4: Generate Review Request Comment

Create a structured comment in `.build/tmp/pr-review-request-<number>.md`:

**CRITICAL**: Comment MUST start with reviewer tags to trigger automation!

```markdown
@claude, @codex — please review

## Review Request

**PR Summary**: [1-2 sentence overview of what this PR does]

### Changes Made
[Summary of key changes from commits and diff analysis]
- Area 1: What changed and why
- Area 2: What changed and why
- Area 3: What changed and why

### Review Focus Areas
Please pay special attention to:

1. **[Category]** - [specific files or areas]
   - [Why this area needs careful review]
   - [Specific concerns or questions]

2. **[Category]** - [specific files or areas]
   - [Why this area needs careful review]

### Test Coverage
- [X] Unit tests added/updated
- [X] Integration tests added/updated
- [ ] Manual testing performed

### Risk Assessment
**Risk Level**: [Low/Medium/High]
- [Key risks or concerns identified]
- [Mitigations in place]

### Context
[Any additional context reviewers should know]
- Related PRs or issues
- Design decisions
- Known limitations or follow-ups

---
Ready for review!
```

### Phase 5: Post Comment
```bash
# Ensure directory exists
mkdir -p .build/tmp

# Post the comment to GitHub
gh pr comment $PR_NUMBER --body-file .build/tmp/pr-review-request-$PR_NUMBER.md

# Verify posting succeeded
echo "✓ Posted review request to PR #$PR_NUMBER"
gh pr view $PR_NUMBER --json url --jq .url
```

## Review Focus Categories

Common areas to highlight:
- **Core Logic** - Indexing, search algorithms, ranking
- **API Changes** - CLI interface, configuration options
- **Performance** - Query speed, indexing performance
- **Testing** - Test coverage and quality
- **Documentation** - README, docs updates, docstrings
- **Configuration** - YAML schema, default values
- **Error Handling** - Edge cases and failure modes

## Smart Analysis Guidelines

### Identify Key Changes
From recent commits:
- What functionality was added/modified?
- Were there any refactorings or structural changes?
- What files changed most significantly?

### Assess Complexity
- **Low**: Documentation, minor fixes, simple additions
- **Medium**: New features, moderate refactors, CLI changes
- **High**: Core algorithms, breaking changes, architectural shifts

### Determine Risk Areas
- Changes to search/indexing algorithms
- CLI interface modifications
- Configuration schema changes
- Changes without adequate test coverage
- Large refactorings affecting many files

## Example Workflow

```bash
# From current branch
/pr-request-review

# Or specify PR number
/pr-request-review 6
```

Expected output:
```
Analyzing PR #6...
- Collected PR metadata ✓
- Analyzed 3 recent commits ✓
- Reviewed file changes (5 files modified) ✓

Generated review request comment in .build/tmp/pr-review-request-6.md
Posted comment to PR #6 ✓

View PR: https://github.com/degree-analytics/mimir/pull/6
```

## Error Handling

### PR Not Found
```
Error: PR not found for current branch
Run: /pr-request-review <PR_NUMBER>
```

### GitHub Authentication
If `gh` commands fail:
```
Error: GitHub authentication required
Run: gh auth login
```

### No PR for Branch
```
Error: No PR found for branch "feature-branch"
Create a PR first with: gt submit
```

## Safety Considerations

- **File Location**: Always uses `.build/tmp/` (project standard)
- **Read-Only Analysis**: No modifications to PR or repository code
- **GitHub CLI**: Uses official `gh` tool for all GitHub operations
- **Markdown Safety**: Properly escapes code blocks and special characters

## Integration with Mímir Workflow

- **GT Integration**: Works with GT workflow branches
- **Structured Output**: Consistent comment format for all PRs
- **Reviewer-Focused**: Provides actionable guidance, not just description
- **Context-Aware**: Understands mimir's architecture and patterns

## Related Commands
- `/commit` - Create commits before requesting review

## Notes
- Uses GitHub CLI (`gh`) for all GitHub operations
- Generates structured, reviewer-friendly comments
- Safe file handling in `.build/tmp/` directory
- Works with both explicit PR numbers and current branch
- Can be customized per project's review culture
