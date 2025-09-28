# Accessing Private Git Dependencies in CI

Mímir (and downstream repos) install the shared `llm-cli-tools-core` package as a private
Git dependency. We now rely on the organization GitHub App
[`llm-cli-tools-core-readonly`](https://github.com/organizations/degree-analytics/settings/apps/llm-cli-tools-core-readonly)
for read access. This section documents how to configure CI with that app. The original
SSH/deploy-key workflow is still included in the appendix in case deploy keys are re-enabled
in the future.

## GitHub App Workflow (recommended)

### 1. Generate a private key

1. Visit the app settings:
   https://github.com/organizations/degree-analytics/settings/apps/llm-cli-tools-core-readonly
2. Under **Private keys**, click **Generate a private key** and download the PEM file.
3. Note the **App ID** and ensure the app is installed on every repository that must consume
   `llm-cli-tools-core` (e.g. `mimir`, `spacewalker`).

### 2. Store credentials as organization secrets

In the `degree-analytics` organization, open *Settings → Secrets and variables → Actions*
and add (already configured):

- Org **variable** `APP_ID` – set to `2026778` (the GitHub App ID).
- Org **secret** `APP_PRIVATE_KEY` – the full PEM private key generated in step 1.

Grant visibility to any repository that installs the package.

### 3. Acquire an installation token in CI

Use [`tibdex/github-app-token`](https://github.com/tibdex/github-app-token) to mint a scoped
token during the workflow run:

```yaml
      - name: Generate GitHub App token
        id: llm_cli_app
        uses: tibdex/github-app-token@v1
        with:
          app_id: ${{ vars.APP_ID }}
          private_key: ${{ secrets.APP_PRIVATE_KEY }}
```

### 4. Install the dependency via HTTPS

Once the token is available, pip can fetch the repo using the `x-access-token` format:

```yaml
      - name: Install dependencies
        run: |
          git config --global url."https://x-access-token:${{ steps.llm_cli_app.outputs.token }}@github.com/".insteadOf https://github.com/
          pip install "git+https://github.com/degree-analytics/llm-cli-tools-core@v0.1.2"
```

Export the token to an environment variable if multiple commands need it within the job.

### 5. Reuse across repositories

Any repository can adopt the same pattern—install the GitHub App on that repo, expose the
two secrets, run the token step, and install via HTTPS. No per-repo SSH keys are required.

---

## Appendix: SSH deploy-key workflow (currently disabled)

If deploy keys are ever re-enabled on `llm-cli-tools-core`, the legacy SSH approach works as
follows:

1. Generate a read-only ED25519 keypair (no passphrase):

   ```bash
   ssh-keygen -t ed25519 -C "llm-cli-tools-core-ci" -f llm-cli-tools-core-ci
   ```

2. Add the **public** key as a deploy key on the repo (read-only).
3. Store the **private** key as an org secret `LLM_CLI_TOOLS_CORE_SSH_KEY` and, optionally,
   `SSH_KNOWN_HOSTS` with the `github.com` host key.
4. In CI, create `~/.ssh/id_ed25519` from the secret, ensure `known_hosts` contains
   `github.com`, start `ssh-agent`, and run:

   ```bash
   pip install git+ssh://git@github.com/degree-analytics/llm-cli-tools-core@v0.1.2
   ```

This keeps the workflow ready if we ever move back to deploy keys.
