# Mímir Commit

Create well-structured commits using GT workflow with intelligent validation and testing.

## Usage
```
/commit
/commit --no-verify
/commit --submit
```

## Prerequisites

- `git` - Version control operations
- `gt` - Graphite CLI for stack management
- `just` - Task runner for validation commands

## Implementation

### Step 1: Verify Current State
```bash
# Check current repository state
git status

# Verify we're on the correct branch
git branch --show-current
```

### Step 2: Stage Files
```bash
# Stage all modified and new files (standard workflow)
git add -A

# Alternative: Stage only specific files if needed
# git add path/to/specific/files

# Verify staging
git status
```

### Step 3: Pre-Commit Validation (unless --no-verify)
```bash
# Run linting
just lint

# Run tests
just test
```

If either fails:
- Stop the commit process
- Report the specific failures
- Guide user to fix issues before proceeding

### Step 4: Review Changes
```bash
# Show detailed diff of staged changes
git diff --staged

# Check commit history for message style
git log --oneline -5
```

### Step 5: Generate Commit Message
Analyze the staged changes and create a conventional commit message:
- **Format**: `type(scope): description` or `type: description`
- **Types**: feat, fix, docs, chore, test, refactor, ci
- **Keep it concise**: Single line preferred
- **Follow existing patterns**: Match the style of recent commits
- **No trailing periods** in subject line

Examples:
- `feat: add semantic search support`
- `fix: handle empty query results correctly`
- `docs: update README with installation steps`
- `test: add indexing test coverage`
- `chore: bump version to 1.4.0`

### Step 6: Create Commit with GT
```bash
# Use gt modify to add commit to current branch
gt modify -m "$(cat <<'EOF'
<generated commit message>
EOF
)"

# Verify commit succeeded
git log -1 --oneline
```

**CRITICAL**:
- **ALWAYS use `gt modify`** - adds commit to current branch
- **NEVER use `gt create`** unless explicitly starting a new feature
- This follows the GT workflow and repository standards

### Step 7: Optional Stack Submission (if --submit flag)
```bash
# Push the entire stack to remote
gt submit
```

## Command Flags

### --no-verify
Skips linting and testing validation:
```bash
/commit --no-verify
```
Use when:
- Emergency fixes
- Documentation-only changes
- You've already validated locally
- CI will catch issues anyway

### --submit
Commits AND pushes the stack to remote:
```bash
/commit --submit
```
Equivalent to:
```bash
gt modify -m "message"
gt submit
```

## Commit Standards

### Conventional Commit Format
```
type(scope): description

[optional body]
```

### Common Types
- **feat**: New feature (triggers minor version bump)
- **fix**: Bug fix (triggers patch version bump)
- **docs**: Documentation only
- **test**: Adding or updating tests
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **chore**: Maintenance tasks, dependencies
- **ci**: CI/CD changes
- **BREAKING**: Breaking changes (triggers major version bump)

### Best Practices
- **Atomic commits**: One logical change per commit
- **Clear descriptions**: Explain what and why, not how
- **Present tense**: "add feature" not "added feature"
- **Reference issues**: Include PR or issue numbers when relevant
- **Keep it short**: Aim for <50 chars in subject line

### Release Triggers
In mimir, certain commit types trigger automatic releases:
- `feat:` → Minor version bump (1.4.0 → 1.5.0)
- `fix:` → Patch version bump (1.4.0 → 1.4.1)
- `BREAKING` in message → Major version bump (1.4.0 → 2.0.0)

## Error Handling

### No Staged Changes
If `git add -A` finds nothing:
- Check if there are any modifications: `git status`
- Inform user: "No changes to commit"

### Linting Failures
- Display specific linting errors
- Stop commit process
- Guide: "Fix linting issues with `just lint` before committing"

### Test Failures
- Display failed test output
- Stop commit process
- Guide: "Fix failing tests before committing"

### Large Changesets
If staged changes are >500 lines:
- Warn about large changeset
- Confirm intent: "This is a large commit. Continue? (y/n)"
- Suggest breaking into smaller commits if appropriate

## Example Workflows

### Standard Commit
```bash
User: /commit

Claude:
1. Checking repository state... ✓
2. Staging all changes... ✓ (8 files changed)
3. Running linting... ✓
4. Running tests... ✓ (42 passed)
5. Reviewing changes...
   - Modified: src/mimir/search.py
   - Modified: tests/unit/test_search.py
6. Generated message: "feat: add fuzzy matching to search queries"
7. Creating commit with gt modify... ✓ [a3f8b21]

Commit created successfully!
Note: This will trigger a minor version release (1.4.0 → 1.5.0)
```

### Skip Validation
```bash
User: /commit --no-verify

Claude:
1. Checking repository state... ✓
2. Staging all changes... ✓ (2 files changed)
3. Skipping validation (--no-verify)
4. Reviewing changes...
   - Modified: README.md
   - Modified: docs/setup.md
5. Generated message: "docs: update installation instructions"
6. Creating commit with gt modify... ✓ [daa4c07]

Commit created successfully!
```

### Commit and Submit
```bash
User: /commit --submit

Claude:
1-7. [standard commit workflow]
8. Submitting stack to remote... ✓

Commit created and stack submitted!
PR updated: https://github.com/degree-analytics/mimir/pull/6
```

## Integration with Mímir Patterns

- **Justfile First**: Uses `just lint` and `just test` for validation
- **GT Exclusive**: Uses only GT commands for commits (`gt modify`)
- **Safety First**: Validates before committing (unless --no-verify)
- **Conventional Commits**: Follows standardized commit message format
- **Release Awareness**: Knows which commits trigger version bumps

## Related Commands
- `/pr-request-review` - Request review after committing

## Notes
- **Always use `gt modify`** for normal commits (not `gt create`)
- **Validation by default** - ensures code quality
- **Smart message generation** - follows project conventions
- **Optional stack submission** - use --submit to push immediately
- **Release-aware** - will note when commit will trigger a version bump
