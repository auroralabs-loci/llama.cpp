#!/usr/bin/env bash
set -euo pipefail

# Creates/updates mirror branches and PRs from pulls.ndjson
#
# Required environment variables:
#   UPSTREAM_REPO    - upstream repo (e.g., "openssl/openssl")
#   GH_TOKEN         - GitHub token for API calls
#   GITHUB_REPOSITORY - current repo

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/utils.sh"

git remote add upstream "https://github.com/${UPSTREAM_REPO}.git" 2>/dev/null || true

while read -r line; do
  [ -z "$line" ] && continue

  num=$(jq -r .pull_number <<<"$line")
  title=$(jq -r .title <<<"$line")
  body=$(jq -r .body <<<"$line")
  pull_head_sha=$(jq -r .pull_head_sha <<<"$line")
  loci_pr_branch=$(jq -r .loci_pr_branch <<<"$line")
  loci_main_branch=$(jq -r .loci_main_branch <<<"$line")
  use_loci_base=$(jq -r '.use_loci_base // 0' <<<"$line")

  # Target loci/main-* only when base_sha was explicitly provided; otherwise target main
  if [ "$use_loci_base" -eq 1 ]; then
    target_base="$loci_main_branch"
  else
    target_base="main"
  fi

  echo "::group::PR #${num}: ${loci_pr_branch} -> ${target_base}"

  echo "Updating ${loci_pr_branch} to ${pull_head_sha}."
  if git show-ref --verify --quiet "refs/remotes/upstream/pr/${num}"; then
    git branch --no-track -f "${loci_pr_branch}" "refs/remotes/upstream/pr/${num}"
  else
    git fetch upstream "refs/pull/${num}/head:refs/heads/${loci_pr_branch}" || \
    git fetch upstream "${pull_head_sha}:refs/heads/${loci_pr_branch}"
  fi
  git push origin "refs/heads/${loci_pr_branch}:refs/heads/${loci_pr_branch}" --force

  upsert_mirror_pr "$loci_pr_branch" "$target_base" "$num"

  echo "::endgroup::"
done < pulls.ndjson
