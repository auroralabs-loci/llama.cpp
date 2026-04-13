#!/usr/bin/env bash
# Checks a mirror PR for an insightful loci summary and notifies the original PR author.
#
# Usage:
#   notify-insightful-prs.sh <mirror_pr_number> <sticky_comment_id>
#
# Requires env: GITHUB_REPOSITORY, GH_TOKEN

set -euo pipefail

NOTIFICATION_MARKER='<!-- loci-notification -->'

mirror_pr_number="${1:?mirror PR number required}"
sticky_comment_id="${2:?sticky comment ID required}"

pr_body=$(gh pr view "$mirror_pr_number" --repo "$GITHUB_REPOSITORY" --json body --jq '.body // ""')

source_pr_path=$(echo "$pr_body" \
  | grep -o '<!-- loci-source-pr: [^>]* -->' \
  | head -1 \
  | sed 's/<!-- loci-source-pr: //;s/ -->//')

if [ -z "$source_pr_path" ]; then
  echo "PR #${mirror_pr_number}: No loci-source-pr marker found, skipping."
  exit 0
fi

source_repo=$(echo "$source_pr_path" | cut -d/ -f1-2)
source_pr_number=$(echo "$source_pr_path" | cut -d/ -f4)
sticky_comment_body=$(gh api "repos/${GITHUB_REPOSITORY}/issues/comments/${sticky_comment_id}" --jq '.body')

echo "PR comment body: ${sticky_comment_body}."

if ! echo "$sticky_comment_body" | grep -qE '!\[(Control Flow Graph|Flame Graph):'; then
  echo "PR #${mirror_pr_number}: Summary has no CFG/Flame Graph images, not insightful. Skipping."
  exit 0
fi

echo "PR #${mirror_pr_number}: Insightful summary found. Source: ${source_repo}#${source_pr_number}"
# comment_url="https://github.com/${GITHUB_REPOSITORY}/pull/${mirror_pr_number}#issuecomment-${sticky_comment_id}"
# notification_body="🔎 Loci found performance insights on this PR. [View Latest Summary →](${comment_url}) ${NOTIFICATION_MARKER}"
# existing_comment_id=$(gh api --paginate "repos/${source_repo}/issues/${source_pr_number}/comments" \
#   | jq -r "[.[] | select(.body | contains(\"${NOTIFICATION_MARKER}\"))] | last | .id // empty" \
#   2>/dev/null || true)
#
# if [ -n "$existing_comment_id" ]; then
#   echo "Updating existing notification comment #${existing_comment_id} on ${source_repo}#${source_pr_number}."
#   gh api "repos/${source_repo}/issues/comments/${existing_comment_id}" \
#     --method PATCH \
#     --field body="$notification_body"
# else
#   echo "Posting new notification comment on ${source_repo}#${source_pr_number}."
#   gh api "repos/${source_repo}/issues/${source_pr_number}/comments" \
#     --method POST \
#     --field body="$notification_body"
# fi
