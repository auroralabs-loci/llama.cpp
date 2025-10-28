## Mirroring project — purpose and required configuration

This repository uses an `overlay` branch to store repository-specific overlays (primarily the `.github` workflows and workflow configuration) that should be applied on top of the upstream default branch.

The `overlay` branch is not a fork of upstream code; instead it contains the project-specific additions and workflow files that are applied to `main` after the repository is kept in sync with an upstream repo. This lets you keep a faithful copy of upstream while maintaining your own CI/workflow customizations.

### How the sync workflows work

Two GitHub Actions workflows coordinate the mirroring and overlay application:

- `sync-upstream.yml` (named "Sync Upstream")
	- Fetches the default branch and latest commit SHA from the upstream repository (configured via GitHub variable - `UPSTREAM_REPO`).
	- Rebuilds `main` from the upstream commit and then restores the `.github` directory from the `overlay` branch so your custom workflows and settings are preserved on `main`.
	- Updates a lightweight tag (default name `upstream-baseline`) pointing to the upstream SHA used as the baseline.
	- Important env/vars/secrets observed in the workflow:
		- `GH_TOKEN` is obtained from `secrets.MIRROR_REPOS_WRITE_PAT` (used to push updates back to the repo).
		- `vars.UPSTREAM_REPO` — repository variable specifying the upstream repo (owner/repo).

- `sync-upstream-prs.yml` (named "Mirror Upstream PRs")
	- Runs after `Sync Upstream` and mirrors a small number of recent open PRs from the upstream repo into this repository as pull requests against the mirrored baseline branch (the default branch `main` in this repo).
	- It only considers PRs opened/updated within a configurable lookback window and limits the number processed (defaults in organization variables: `MAX_UPSTREAM_PRS=1`, `UPSTREAM_PR_LOOKBACK_DAYS=2`).
	- For each selected upstream PR it creates/updates an origin branch named like `upstream-PR<N>-branch_<owner>-<headref>` and opens a mirrored PR in this repo (or updates the branch if it already exists).
	- It will also close mirrored PRs here if the upstream PR has been closed/changed base.
	- Important env/vars/secrets observed in the workflow:
		- `GH_TOKEN` is obtained from `secrets.MIRROR_REPOS_WRITE_PAT` (used for pushing mirrored branches and calling gh APIs).
		- `vars.UPSTREAM_REPO` — upstream repository variable.
		- `UPSTREAM_BASELINE_TAG` — tag used as the baseline reference; the workflow resolves this tag to avoid opening PRs when the baseline moved.

### Repository secrets and variables you should add/check if your GitHub repo has access to

To make the overlay sync and Loci workflows run correctly, add the following to the repository (or organization) settings:

- Repository variables (`Settings → Variables`):
	- `UPSTREAM_REPO` — required. Format: `owner/repo` (example: `example/upstream-repo`). Used by the sync workflows to know which upstream repo to mirror.
	- `LOCI_DEV_BACKEND_URL` — required by Loci Action. The URL of your Loci backend API (example: `https://dev.api.loci-dev.net/`).

- Repository secrets (`Settings → Secrets and variables → Actions → Secrets`):
	- `MIRROR_REPOS_WRITE_PAT` — required. A personal access token (PAT) with repo write permissions for this repository so the workflows can push branches/tags and open/close PRs. The workflows use it as `GH_TOKEN`.
	- `REPO_READ_ONLY_PAT` — recommended/required by the Loci action when generating GH summaries. A token with read access to the repository (less privileged than the write PAT) — used as `LOCI_GITHUB_TOKEN` in the Loci action.
	- `LOCI_API_KEY` — required. The API key for your Loci application (project-scoped or account-scoped depending on your Loci setup).

Notes and recommended permissions:
	- `MIRROR_REPOS_WRITE_PAT` must be able to push branches and tags and create PRs; it is used for both the mirror updates and PR mirroring.
	- `REPO_READ_ONLY_PAT` can be a least-privileged token that allows reading repo metadata for the summary step.
	- Keep `LOCI_DEV_AGENTIC_API_KEY` secret and rotate it per your security policy.

### Quick checklist / contract

- Inputs
	- Upstream repository: `vars.UPSTREAM_REPO` (owner/repo)
	- Secrets: `secrets.MIRROR_REPOS_WRITE_PAT`, `secrets.REPO_READ_ONLY_PAT`, `secrets.LOCI_DEV_AGENTIC_API_KEY`
	- Loci back-end: `vars.LOCI_DEV_BACKEND_URL` and `LOCI_PROJECT` (env in workflow or set as a variable)

- Outputs / side-effects
	- Pushes/updates the `main` branch and the `upstream-baseline` tag.
	- Creates/updates mirrored PR branches and PRs in this repository.
	- Uploads artifacts to Loci and posts summaries when configured.

### Next steps / recommendations

- Add the repository variables and secrets listed above before enabling the workflows.
- Create a Loci project matching `LOCI_PROJECT` and obtain an API (per company) key to use as `LOCI_DEV_AGENTIC_API_KEY`. Configure `LOCI_DEV_BACKEND_URL` to point to your Loci backend.
- If you want a safer rollout, run the workflows via `workflow_dispatch` first and verify behavior in a test repository.
