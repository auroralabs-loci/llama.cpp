#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/utils.sh"

# fetch-upstream-prs.sh for upstreams that review on Gerrit: selects open changes into /tmp/pulls.ndjson.
#
# Required environment variables:
#   UPSTREAM_REPO, UPSTREAM_DEFAULT, UPSTREAM_GERRIT_URL, GH_TOKEN, GITHUB_ACTOR, GITHUB_OUTPUT, GITHUB_REPOSITORY
#
# Optional environment variables:
#   UPSTREAM_GERRIT_PROJECT   - Gerrit project (default: UPSTREAM_REPO)
#   UPSTREAM_GERRIT_QUERY     - extra query terms (default: label:Verified=+1)
#   UPSTREAM_PR_LOOKBACK_DAYS, MAX_UPSTREAM_PRS, PR_URL, BASE_SHA - as in fetch-upstream-prs.sh

project="$(gerrit_project)"
extra_query="${UPSTREAM_GERRIT_QUERY:-label:Verified=+1}"

git remote add upstream "https://github.com/${UPSTREAM_REPO}.git" 2>/dev/null || true
git remote add gerrit "${UPSTREAM_GERRIT_URL}/${project}" 2>/dev/null || true
git fetch upstream "${UPSTREAM_DEFAULT}:refs/remotes/upstream/${UPSTREAM_DEFAULT}"
git fetch origin overlay:refs/remotes/origin/overlay || true

git config user.name "${GITHUB_ACTOR}"
git config user.email "${GITHUB_ACTOR}@users.noreply.github.com"

# Gerrit rebases a change when it submits it, so merges are detected by status, not by commit.
mirrored=$(gh pr list -R "$GITHUB_REPOSITORY" --state open --limit 200 --json number,body \
  --jq '.[] | "\(.number) \(.body | capture("<!-- loci-source-gerrit: [^ ]*/\\+/(?<n>[0-9]+) -->").n // empty)"' 2>/dev/null || true)
if [ -n "$mirrored" ]; then
  query=$(awk '$2 {printf "%schange:%s", (n++ ? "+OR+" : ""), $2}' <<<"$mirrored")
  gerrit_api "changes/?q=${query}&n=200" | jq -r '.[] | select(.status != "NEW") | "\(._number) \(.status)"' |
  while read -r change status; do
    pr=$(awk -v c="$change" '$2 == c {print $1}' <<<"$mirrored")
    echo "Change ${change} is ${status}: closing mirror PR #${pr}."
    gh pr close "$pr" -R "$GITHUB_REPOSITORY" --delete-branch \
      --comment "Upstream change ${change} was ${status,,}." || echo "::warning::Could not close #${pr}"
  done
fi

if [ -n "${BASE_SHA:-}" ] && [ -z "${PR_URL:-}" ]; then
  echo "Base SHA mode: creating loci/main branch for ${BASE_SHA}"
  bash "$SCRIPT_DIR/sync-loci-main.sh" "$BASE_SHA" || true
  echo "prs_to_sync=no" >> "$GITHUB_OUTPUT"
  exit 0
fi

> /tmp/pulls.ndjson
selected=0
new_bases=" "
manual_mode=0
max_pulls="${MAX_UPSTREAM_PRS:-2}"
fields="o=CURRENT_REVISION&o=CURRENT_COMMIT"

if [ -n "${PR_URL:-}" ]; then
  manual_mode=1
  if [[ ! "$PR_URL" =~ ^${UPSTREAM_GERRIT_URL}/c/${project}/\+/([0-9]+)(/[0-9]+)?/?$ ]]; then
    echo "::error::Invalid change URL. Expected: ${UPSTREAM_GERRIT_URL}/c/${project}/+/12345"
    exit 1
  fi
  query="change:${BASH_REMATCH[1]}"
else
  today=$(date -u +%Y-%m-%d)
  opened_today=$(gh pr list -R "$GITHUB_REPOSITORY" --state all --search "created:>=${today}" --limit 200 --json number --jq length)
  pending_count=$(git ls-remote --heads origin 'refs/heads/loci/pending-pr-*' | wc -l)
  max_pulls=$((max_pulls - opened_today - pending_count))
  [ "$max_pulls" -ge 0 ] || max_pulls=0
  echo "Daily quota: ${MAX_UPSTREAM_PRS:-2}, already used today: ${opened_today} opened + ${pending_count} pending."
  query="project:${project} status:open branch:${UPSTREAM_DEFAULT} -is:wip -age:${UPSTREAM_PR_LOOKBACK_DAYS:-7}d ${extra_query}"
  echo "Searching for ${max_pulls} changes: ${query}"
fi
query=$(jq -rn --arg q "$query" '$q | @uri')

start=0
while [ "$manual_mode" -eq 1 ] || [ "$selected" -lt "$max_pulls" ]; do
  changes=$(gerrit_api "changes/?q=${query}&n=25&S=${start}&${fields}")
  [ "$(jq length <<<"$changes")" -gt 0 ] || { echo "Changes exhausted."; break; }

  while read -r change; do
    if [ "$manual_mode" -eq 0 ] && [ "$selected" -ge "$max_pulls" ]; then
      echo "Quota of ${max_pulls} reached, stopping."
      break 2
    fi

    num=$(jq -r ._number <<<"$change")
    head=$(jq -r .current_revision <<<"$change")
    ref=$(jq -r '.revisions[.current_revision].ref' <<<"$change")
    parent=$(jq -r '.revisions[.current_revision].commit.parents[0].commit' <<<"$change")
    # Stable across patchsets, since the subject can change; process-pending-prs.sh needs the trailing part.
    loci_pr_branch="loci/pr-${num}-gerrit"

    git fetch --no-tags gerrit "${ref}:refs/remotes/upstream/pr/${num}"

    # Stacked on another open change: comparing against master would mix in the changes below it.
    if ! git merge-base --is-ancestor "$parent" "refs/remotes/upstream/${UPSTREAM_DEFAULT}"; then
      if [ "$manual_mode" -eq 1 ]; then
        echo "::notice::Change ${num} is stacked on an open change; its analysis includes the changes below it."
      else
        echo "  Change ${num}: stacked on an open change. Skipping."
        continue
      fi
    fi

    merge_base="${BASE_SHA:-$(git merge-base "$head" "refs/remotes/upstream/${UPSTREAM_DEFAULT}")}"
    echo "  Change ${num}: patchset $(jq -r '.revisions[.current_revision]._number' <<<"$change"), merge-base ${merge_base}"

    if loci_main_branch=$(bash "$SCRIPT_DIR/sync-loci-main.sh" "$merge_base"); then
      case "$new_bases" in *" ${merge_base} "*) base_is_new=1 ;; *) base_is_new=0 ;; esac
    else
      new_bases="${new_bases}${merge_base} "
      base_is_new=1
    fi

    # Base analysis was only just triggered: a PR run now would fail with "version does not exist"
    if [ "$base_is_new" -eq 1 ] && [ "$manual_mode" -eq 0 ]; then
      pending_branch="loci/pending-pr-${num}-gerrit"
      echo "  Change ${num}: ${loci_main_branch} just triggered. Pushing pending branch ${pending_branch}."
      git push origin "refs/remotes/upstream/pr/${num}:refs/heads/${pending_branch}" --force
      selected=$((selected + 1))
      continue
    fi

    if [ -n "${BASE_SHA:-}" ]; then
      conflict_target="refs/heads/${loci_main_branch}"
      use_loci_base=1
    else
      conflict_target="refs/remotes/upstream/${UPSTREAM_DEFAULT}"
      use_loci_base=0
    fi
    if ! git merge-tree --write-tree --merge-base "$merge_base" "$head" "$conflict_target" &>/dev/null; then
      echo "  Change ${num}: conflicts with ${conflict_target#refs/*/}. Skipping."
      [ "$manual_mode" -eq 0 ] || { echo "::error::Change has merge conflicts"; exit 1; }
      continue
    fi

    if [ "$(git ls-remote --heads origin "refs/heads/${loci_pr_branch}" | cut -f1)" = "$head" ] && [ "$manual_mode" -eq 0 ]; then
      echo "  Change ${num}: already up-to-date. Skipping."
      continue
    fi

    jq -nc \
      --arg pull_number "$num" \
      --arg pull_head_sha "$head" \
      --arg loci_pr_branch "$loci_pr_branch" \
      --arg short_merge_base "${merge_base:0:7}" \
      --arg loci_main_branch "$loci_main_branch" \
      --argjson use_loci_base "$use_loci_base" \
      '$ARGS.named' >> /tmp/pulls.ndjson
    selected=$((selected + 1))
    echo "  Change ${num}: added (${selected})."
  done < <(jq -c '.[]' <<<"$changes")

  [ "$manual_mode" -eq 0 ] && [ "$(jq '.[-1]._more_changes // false' <<<"$changes")" = true ] || break
  start=$((start + 25))
done

if [ "$selected" -eq 0 ]; then
  latest=$(git rev-parse "refs/remotes/upstream/${UPSTREAM_DEFAULT}")
  echo "No changes selected. Syncing latest upstream ${UPSTREAM_DEFAULT} (${latest})."
  bash "$SCRIPT_DIR/sync-loci-main.sh" "$latest" || true
  echo "prs_to_sync=no" >> "$GITHUB_OUTPUT"
else
  echo "prs_to_sync=yes" >> "$GITHUB_OUTPUT"
  echo "Selected ${selected} upstream changes to process"
fi
