#!/usr/bin/env bash
set -euo pipefail

# Disables the workflows that syncing main brings in from upstream, keeping the ones on overlay.
#
# Required environment variables:
#   GH_TOKEN, GITHUB_REPOSITORY

overlay_workflows=$(git ls-tree --name-only refs/remotes/origin/overlay .github/workflows/)

gh api "repos/${GITHUB_REPOSITORY}/actions/workflows" --paginate \
  --jq '.workflows[] | select(.state == "active" and (.path | startswith(".github/workflows/"))) | "\(.id) \(.path)"' |
while read -r id path; do
  if grep -qxF "$path" <<<"$overlay_workflows"; then
    continue
  fi
  echo "Disabling ${path}"
  gh workflow disable "$id" --repo "$GITHUB_REPOSITORY" || echo "::warning::Could not disable ${path}"
done
