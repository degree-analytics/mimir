---
purpose: "{Document CLI commands, flags, and behavior}"
audience: "{Engineers using the CLI}"
owner: "{CLI maintainer}"
review: "{YYYY-MM-DD (Quarterly)}"
status: draft
---

# {CLI Name} Reference

## When to Use This

- Look up supported commands
- Confirm flags and expected output

## Prerequisites

- `{cli}` installed (`pip install -e .`)
- Access to `docs/` content for indexing

## Commands

### {command}

```bash
{cli} {command} {options}
```

Options:

- `--flag` – {description} (default: `{value}`)

**Verification**: `{cli} {command}` returns `{expected}`.

### {additional_command}

```bash
{cli} {command}
```

## Configuration

```yaml
cache_dir: .cache/mimir
vector_search:
  enabled: false
```

## References

- `[Docs Index](docs/index.md)`
- `[Spacewalker Docs Search Guide](../spacewalker/docs/workflows/documentation-search-guide.md)`
