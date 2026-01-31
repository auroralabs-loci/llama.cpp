#!/usr/bin/env bash
set -euo pipefail

# Syncs origin default branch to upstream and creates loci/main-* overlay branch
#
# Required environment variables:
#   UPSTREAM_REPO          - upstream repo (e.g., "openssl/openssl")
#   UPSTREAM_SHA           - upstream commit SHA to sync to
#   GH_TOKEN               - GitHub token for API calls
#   GITHUB_ACTOR           - actor for git config

git remote add upstream "https://github.com/${UPSTREAM_REPO}.git" 2>/dev/null || true
git fetch upstream --prune --tags

# --- Sync origin default branch to upstream (with loci-analysis.yml overlay) ---

git config user.name "${GITHUB_ACTOR}"
git config user.email "${GITHUB_ACTOR}@users.noreply.github.com"

git fetch origin "main:refs/remotes/origin/main" || true
git fetch origin "overlay:refs/remotes/origin/overlay" || true

SCRIPT_DIR="$(dirname "$0")"
SYNC_LOCI_MAIN_SCRIPT="/tmp/sync-loci-main.sh"
cp "$SCRIPT_DIR/sync-loci-main.sh" "$SYNC_LOCI_MAIN_SCRIPT"

#--------------------------------------------------#
git checkout -B main "${UPSTREAM_SHA}"
#--------------------------------------------------#

git restore --source "refs/remotes/origin/overlay" -- .github/workflows/loci-analysis.yml || true
if [ -n "$(git status --porcelain)" ]; then
  git add -A
  git commit -m "Add loci-analysis workflow from overlay"
fi

origin_sha=$(git rev-parse "refs/remotes/origin/main" 2>/dev/null || true)
current_sha=$(git rev-parse HEAD)

if [ "$origin_sha" = "$current_sha" ]; then
  echo "Origin main already up-to-date. Skipping sync."
else
  echo "Updating origin main to: ${current_sha} (upstream: ${UPSTREAM_SHA} + loci-analysis.yml)."
  git push origin "main:refs/heads/main" --force
fi

#--------------------------------------------------#
git checkout origin/overlay -- .
#--------------------------------------------------#

# --- Create/update loci/main-* branch with overlay ---

if loci_main_branch=$(bash "$SYNC_LOCI_MAIN_SCRIPT" "$UPSTREAM_SHA"); then
  echo "Branch ${loci_main_branch} already up-to-date."
else
  echo "Branch ${loci_main_branch} created/updated."
fi
