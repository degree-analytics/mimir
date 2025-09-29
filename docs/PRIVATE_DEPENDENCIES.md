---
purpose: "Explain how CI installs the private llm-cli-tools-core dependency"
audience: "Platform engineers and repo maintainers"
owner: "Platform Eng"
review: "2025-10-01 (Quarterly)"
status: "Active"
---

# Accessing Private Git Dependencies in CI

## When to Use This

- Pipelines need `llm-cli-tools-core` or other private GitHub repositories
- New repositories adopt the shared GitHub App workflow

## Prerequisites

- GitHub App `llm-cli-tools-core-readonly` installed on the consuming
  repository
- Organization-level access to create Actions secrets and variables
- `just` available to run dispatcher commands (see `CLAUDE.md`)

## Secrets & Variables

| Name | Type | Scope | Purpose |
| --- | --- | --- | --- |
| `APP_ID` | Actions Variable | Organization | GitHub App ID (2026778). |
| `APP_PRIVATE_KEY` | Actions Secret | Organization | PEM from App settings |

## Steps

1. **Generate a private key**
   - Visit the app settings:
  <https://github.com/organizations/degree-analytics/settings/apps/llm-cli-tools-core-readonly>
   - Under **Private keys**, click **Generate a private key** and download the
  PEM file.
   - Confirm the app is installed on each repository that needs
  `llm-cli-tools-core` (e.g. `mimir`, `spacewalker`).

2. **Store credentials as organization secrets**
   - Navigate to *Settings → Secrets and variables → Actions*.
   - Add variable `APP_ID` with value `2026778`.
   - Add secret `APP_PRIVATE_KEY` containing the PEM content.
   - Grant visibility to each repository that installs the package.

3. **Acquire an installation token in CI**

   ```yaml
   - name: Generate GitHub App token
     id: llm_cli_app
     uses: tibdex/github-app-token@v2
     with:
       app_id: ${{ vars.APP_ID }}
       private_key: ${{ secrets.APP_PRIVATE_KEY }}
   ```

4. **Install the dependency via HTTPS**

   ```yaml
   - name: Install dependencies
     run: |
       TOKEN="${{ steps.llm_cli_app.outputs.token }}"
       git config --global \
         url."https://x-access-token:${TOKEN}@github.com/" \
         .insteadOf https://github.com/
       pip install "git+https://github.com/degree-analytics/llm-cli-tools-core@v0.1.4"
   ```

5. **Reuse across repositories**

   - Install the GitHub App on each repo, expose the two credentials, run the
     token step, reuse the HTTPS install block.
   - No per-repo SSH keys are required under this workflow.

## Verification

- `just docs check` (once spell/link linting lands) confirms docs stay up to
  date.
- GitHub Action logs show `tibdex/github-app-token@v2` succeeded and the pip
  installation exited `0`.
- Optional local test:

  ```bash
  just docs check
  ```

  confirms markdown linting and `mimir index` run before merging doc
  changes.

## Appendix: Legacy SSH Deploy-Key Workflow (Disabled)

1. Generate a read-only ED25519 keypair:

   ```bash
   ssh-keygen -t ed25519 -C "llm-cli-tools-core-ci" -f llm-cli-tools-core-ci
   ```

2. Add the **public** key as a deploy key on the repository (read-only).
3. Store the **private** key as secret `LLM_CLI_TOOLS_CORE_SSH_KEY`;
   optionally add `SSH_KNOWN_HOSTS`.
4. In CI, create `~/.ssh/id_ed25519` from the secret, ensure `known_hosts`
   contains `github.com`, start `ssh-agent`, and run:

   ```bash
   pip install git+ssh://git@github.com/degree-analytics/llm-cli-tools-core@v0.1.4
   ```

## References

- `docs/index.md` — documentation taxonomy and lifecycle.
- `../spacewalker/docs/workflows/github-actions-best-practices.md` — broader
  CI guardrails.
