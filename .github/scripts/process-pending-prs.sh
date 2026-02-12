#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/utils.sh"

# List pending-pr branches, rename to pr, create mirror PRs
pending_branches=$(git ls-remote --heads origin 'refs/heads/loci/pending-pr-*' | awk '{print $2}' | sed 's|refs/heads/||')
if [ -z "$pending_branches" ]; then
  echo "No pending PR branches found."
  exit 0
fi

while read -r pending_branch; do
  [[ "$pending_branch" =~ ^loci/pending-pr-([0-9]+)-(.+)$ ]] || continue
  num="${BASH_REMATCH[1]}"
  rest="${BASH_REMATCH[2]}"
  pr_branch="loci/pr-${num}-${rest}"

  echo "::group::Promoting ${pending_branch} → ${pr_branch}"

  # Rename: push as loci/pr-*, delete loci/pending-pr-*
  git fetch origin "${pending_branch}:refs/heads/${pending_branch}"
  git push origin "refs/heads/${pending_branch}:refs/heads/${pr_branch}" --force
  git push origin --delete "${pending_branch}"

  # Create mirror PR (lib fetches upstream metadata internally)
  upsert_mirror_pr "$pr_branch" "main" "$num"

  echo "::endgroup::"
done <<<"$pending_branches"
