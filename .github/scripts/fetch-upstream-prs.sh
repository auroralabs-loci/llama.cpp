#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"

# Fetches and processes upstream PRs, outputting selected PRs to pulls.ndjson
#
# Required environment variables:
#   UPSTREAM_REPO         - upstream repo (e.g., "openssl/openssl")
#   UPSTREAM_DEFAULT      - upstream default branch name
#   GH_TOKEN              - GitHub token for API calls
#   GITHUB_ACTOR          - actor for git config
#   GITHUB_OUTPUT         - path to output file
#
# Optional environment variables (for scheduled mode):
#   UPSTREAM_PR_LOOKBACK_DAYS - how far back to look for PRs
#   MAX_UPSTREAM_PRS          - max PRs to process
#
# Optional environment variables (for manual mode):
#   PR_URL                - specific PR URL to mirror
#   BASE_SHA              - upstream main commit SHA to use as base (overrides merge-base)

git remote add upstream "https://github.com/${UPSTREAM_REPO}.git" 2>/dev/null || true
git fetch upstream "${UPSTREAM_DEFAULT}:refs/remotes/upstream/${UPSTREAM_DEFAULT}"
git fetch origin overlay:refs/remotes/origin/overlay || true

git config user.name "${GITHUB_ACTOR}"
git config user.email "${GITHUB_ACTOR}@users.noreply.github.com"

# If BASE_SHA is provided without PR_URL, just create the loci/main branch and exit
if [ -n "${BASE_SHA:-}" ] && [ -z "${PR_URL:-}" ]; then
  echo "Base SHA mode: creating loci/main branch for ${BASE_SHA}"
  if loci_main_branch=$(bash "$SCRIPT_DIR/sync-loci-main.sh" "$BASE_SHA"); then
    echo "Branch ${loci_main_branch} already exists and is up-to-date."
  else
    echo "Created/updated ${loci_main_branch}."
  fi
  echo "prs_to_sync=no" >> "$GITHUB_OUTPUT"
  exit 0
fi

> pulls.ndjson
selected_pulls_count=0
manual_mode=0

if [ -n "${PR_URL:-}" ]; then
  manual_mode=1
  echo "Manual mode: processing PR from URL: ${PR_URL}"

  if [[ ! "$PR_URL" =~ ^https://github.com/([^/]+/[^/]+)/pull/([0-9]+)$ ]]; then
    echo "::error::Invalid PR URL format. Expected: https://github.com/owner/repo/pull/123"
    exit 1
  fi

  pr_repo="${BASH_REMATCH[1]}"
  manual_pr_num="${BASH_REMATCH[2]}"

  if [ "$pr_repo" != "$UPSTREAM_REPO" ]; then
    echo "::error::PR repo (${pr_repo}) does not match UPSTREAM_REPO (${UPSTREAM_REPO})"
    exit 1
  fi

  pulls=$(gh api "repos/${UPSTREAM_REPO}/pulls/${manual_pr_num}" | jq -s '.')
else
  lookback_days="${UPSTREAM_PR_LOOKBACK_DAYS:-7}"
  cutoff=$(date -u -d "${lookback_days} days ago" +%Y-%m-%dT%H:%M:%SZ)
  max_pulls="${MAX_UPSTREAM_PRS:-10}"
  per_page=20
  page=1

  echo "Searching for ${max_pulls} valid pull requests targeting ${UPSTREAM_DEFAULT}, updated since ${cutoff}."
fi

while true; do
  if [ "$manual_mode" -eq 0 ]; then
    pulls=$(gh api "repos/${UPSTREAM_REPO}/pulls?state=open&base=${UPSTREAM_DEFAULT}&sort=updated&direction=desc&per_page=${per_page}&page=${page}" 2>/dev/null || echo "[]")
    page_pulls_count=$(echo "$pulls" | jq 'length')

    if [ "$page_pulls_count" -eq 0 ]; then
      echo "Pull requests exhausted on page ${page}. Stopping."
      break
    fi
    echo "Processing page ${page} (${page_pulls_count} pull requests)"
  fi

  while read -r pr; do
    pull_num=$(jq -r '.number' <<<"$pr")
    pull_head_sha=$(jq -r '.head.sha' <<<"$pr")
    pull_head_ref=$(jq -r '.head.ref' <<<"$pr")

    # Skip cutoff check in manual mode
    if [ "$manual_mode" -eq 0 ]; then
      updated_at=$(jq -r '.updated_at' <<<"$pr")
      created_at=$(jq -r '.created_at' <<<"$pr")
      if [[ "$updated_at" < "$cutoff" && "$created_at" < "$cutoff" ]]; then
        continue
      fi
    fi

    # Sanitize branch name: replace / with -, truncate to 50 chars
    sanitized_branch=$(echo "${pull_head_ref}" | tr '/' '-' | cut -c1-50)
    loci_pr_branch="loci/pr-${pull_num}-${sanitized_branch}"

    # Fetch pull request head for merge-base computation
    git fetch upstream "refs/pull/${pull_num}/head:refs/remotes/upstream/pr/${pull_num}" 2>/dev/null || \
    git fetch upstream "${pull_head_sha}:refs/remotes/upstream/pr/${pull_num}" 2>/dev/null || true

    # Determine merge-base: use BASE_SHA if provided, otherwise compute it
    if [ -n "${BASE_SHA:-}" ]; then
      merge_base="${BASE_SHA}"
      echo "  PR #${pull_num}: using provided BASE_SHA as merge-base: ${merge_base}"
    else
      merge_base=$(git merge-base "${pull_head_sha}" "refs/remotes/upstream/${UPSTREAM_DEFAULT}" 2>/dev/null || true)
      if [ -z "${merge_base}" ]; then
        echo "  PR #${pull_num}: could not compute merge-base. Skipping."
        if [ "$manual_mode" -eq 1 ]; then
          echo "::error::Could not compute merge-base for manually specified PR"
          exit 1
        fi
        continue
      fi
    fi

    short_merge_base="${merge_base:0:7}"

    # Create or update base branch if needed (must happen before conflict check when using loci base)
    if loci_main_branch=$(bash "$SCRIPT_DIR/sync-loci-main.sh" "$merge_base"); then
      : # Branch already up-to-date
    else
      # Branch was created/updated - in scheduled mode, skip PR until next run
      if [ "$manual_mode" -eq 0 ]; then
        echo "  PR #${pull_num}: created/updated ${loci_main_branch}. Skipping PR until next run."
        continue
      else
        echo "  PR #${pull_num}: created/updated ${loci_main_branch}. Continuing with PR."
      fi
    fi

    # Check for merge conflicts - against loci/main-* when BASE_SHA provided, otherwise upstream default
    if [ -n "${BASE_SHA:-}" ]; then
      conflict_target="refs/heads/${loci_main_branch}"
      conflict_target_name="$loci_main_branch"
    else
      conflict_target="refs/remotes/upstream/${UPSTREAM_DEFAULT}"
      conflict_target_name="upstream ${UPSTREAM_DEFAULT}"
    fi

    if ! git merge-tree --write-tree "${merge_base}" "${pull_head_sha}" "${conflict_target}" &>/dev/null; then
      echo "  PR #${pull_num}: has conflicts with ${conflict_target_name}. Skipping."
      if [ "$manual_mode" -eq 1 ]; then
        echo "::error::PR has merge conflicts with ${conflict_target_name}"
        exit 1
      fi
      continue
    fi

    origin_sha=$(git ls-remote --heads origin "refs/heads/${loci_pr_branch}" | cut -f1 || true)
    if [ -n "${origin_sha}" ] && [ "${origin_sha}" = "${pull_head_sha}" ]; then
      echo "  PR #${pull_num}: already up-to-date."
      # In manual mode, still add it (user explicitly requested); in scheduled mode, skip
      if [ "$manual_mode" -eq 0 ]; then
        echo "  Skipping."
        continue
      else
        echo "  Adding anyway (manual mode)."
      fi
    fi

    # Determine if we should target loci/main-* (only when base_sha explicitly provided)
    if [ -n "${BASE_SHA:-}" ]; then
      use_loci_base=1
    else
      use_loci_base=0
    fi

    # Select pull request
    jq -c \
      --arg pull_number "$pull_num" \
      --arg pull_head_sha "$pull_head_sha" \
      --arg loci_pr_branch "$loci_pr_branch" \
      --arg short_merge_base "$short_merge_base" \
      --arg loci_main_branch "$loci_main_branch" \
      --argjson use_loci_base "$use_loci_base" \
      '{
        pull_number: $pull_number,
        title: .title,
        body: (.body // ""),
        pull_head_sha: $pull_head_sha,
        loci_pr_branch: $loci_pr_branch,
        short_merge_base: $short_merge_base,
        loci_main_branch: $loci_main_branch,
        use_loci_base: $use_loci_base
      }' <<<"$pr" >> pulls.ndjson

    selected_pulls_count=$((selected_pulls_count + 1))
    echo "  PR #${pull_num}: added (${selected_pulls_count})."

    if [ "$manual_mode" -eq 0 ] && [ "$selected_pulls_count" -ge "$max_pulls" ]; then
      echo "Quota of ${max_pulls} reached, stopping."
      break 2
    fi
  done < <(echo "$pulls" | jq -c '.[]' 2>/dev/null)

  # In manual mode, we only process one PR, so break after first iteration
  if [ "$manual_mode" -eq 1 ]; then
    break
  fi

  page=$((page + 1))
done

if [ "$selected_pulls_count" -eq 0 ]; then
  echo "prs_to_sync=no" >> "$GITHUB_OUTPUT"
  echo "No valid upstream PRs to sync"
else
  echo "prs_to_sync=yes" >> "$GITHUB_OUTPUT"
  echo "Selected ${selected_pulls_count} upstream PRs to process"
fi