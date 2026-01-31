#!/usr/bin/env bash
set -euo pipefail

# Ensures loci/main-{sha} branch exists with up-to-date loci-analysis.yml
#
# Usage: sync-loci-main.sh <base_sha>
#
# Required environment variables:
#   GITHUB_ACTOR - actor for git config
#
# Assumes:
#   - refs/remotes/origin/overlay is already fetched
#   - git user config is already set
#
# Output: prints the branch name to stdout
# Exit codes:
#   0 = branch was already up-to-date (no changes made)
#   1 = branch was created or updated

base_sha="$1"
short_sha="${base_sha:0:7}"
loci_main_branch="loci/main-${short_sha}"

if git ls-remote --exit-code --heads origin "refs/heads/${loci_main_branch}" &>/dev/null; then
  git fetch origin "${loci_main_branch}:refs/remotes/origin/${loci_main_branch}" 2>/dev/null || true

  overlay_hash=$(git rev-parse "refs/remotes/origin/overlay:.github/workflows/loci-analysis.yml" 2>/dev/null || true)
  branch_hash=$(git rev-parse "refs/remotes/origin/${loci_main_branch}:.github/workflows/loci-analysis.yml" 2>/dev/null || true)

  if [ "$overlay_hash" = "$branch_hash" ]; then
    echo "$loci_main_branch"
    exit 0 # branch is up-to-date
  fi

  echo "Branch ${loci_main_branch} exists but loci-analysis.yml changed. Updating." >&2
else
  echo "Creating ${loci_main_branch} from commit: ${base_sha}." >&2
fi

#--------------------------------------------------#
git checkout -B "${loci_main_branch}" "${base_sha}"
#--------------------------------------------------#

git restore --source refs/remotes/origin/overlay -- .github/workflows/loci-analysis.yml || true
if [ -n "$(git status --porcelain)" ]; then
  git add -A
  git commit -m "Add loci-analysis workflow from overlay"
fi
git push origin "${loci_main_branch}" --force

#--------------------------------------------------#
git checkout origin/overlay -- .
#--------------------------------------------------#

echo "$loci_main_branch"
exit 1 # branch was updated/created
