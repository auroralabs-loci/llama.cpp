#!/usr/bin/env bash
# Shared functionality for mirror PR operations. Source this file, don't execute it.
#
# Requires env: UPSTREAM_REPO, GITHUB_REPOSITORY, GH_TOKEN

# upsert_mirror_pr <pr_branch> <target_base> <upstream_pr_number>
#
# Checks if a mirror PR already exists for the given branch+base.
# If not, fetches upstream PR metadata and creates the mirror PR.
upsert_mirror_pr() {
  local pr_branch="$1" target_base="$2" num="$3"

  local existing
  existing=$(gh pr list --repo "$GITHUB_REPOSITORY" --state open \
    --head "$pr_branch" --base "$target_base" \
    --json number --jq '.[0].number // empty' 2>/dev/null || true)

  if [ -n "${existing}" ]; then
    echo "Mirrored PR #${existing} already exists. Branch updated."
    return 0
  fi

  # Fetch upstream PR metadata
  local pr_data title body
  pr_data=$(gh api "repos/${UPSTREAM_REPO}/pulls/${num}" 2>/dev/null || echo '{}')
  title=$(jq -r '.title // "Unknown"' <<<"$pr_data")
  body=$(jq -r '.body // ""' <<<"$pr_data")

  echo "Creating mirrored PR targeting ${target_base}."
  local PR_BODY
  PR_BODY=$(printf '> [!NOTE]\n> Source pull request: [%s#%s](https://github.com/%s/pull/%s)\n\n%s' \
    "$UPSTREAM_REPO" "$num" "$UPSTREAM_REPO" "$num" "$body")

  gh pr create --repo "$GITHUB_REPOSITORY" \
    --head "${pr_branch}" \
    --base "$target_base" \
    --title "UPSTREAM PR #${num}: ${title}" \
    --body "$PR_BODY"
}
