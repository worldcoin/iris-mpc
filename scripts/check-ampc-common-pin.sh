#!/usr/bin/env bash
#
# Fails if this repo pins an ampc-common revision that is not reachable from
# ampc-common's main branch. Three states it catches:
#
#   1. The pin is a commit on an unmerged ampc-common branch, so main here would
#      build against a dependency that does not exist upstream.
#   2. The pin went stale while the PR waited. ampc-common squash-merges, so the
#      pre-merge commit is genuinely not reachable from main afterwards, and
#      restoring such a pin downgrades this repo's dependency.
#   3. Cargo.toml moved to a new rev but Cargo.lock was not regenerated. The
#      builds run cargo without --locked, so they resolve the manifest and
#      succeed while the lockfile still holds a valid old pin.
#
# Both files are read for that reason. Ancestry is checked against the full
# commit hash the lockfile records after '#', because Cargo permits abbreviated
# revs and a remote cannot resolve an abbreviated object id.
#
# Runs standalone: bash scripts/check-ampc-common-pin.sh [Cargo.lock] [Cargo.toml]

set -euo pipefail

LOCK="${1:-Cargo.lock}"
MANIFEST="${2:-Cargo.toml}"
UPSTREAM="https://github.com/worldcoin/ampc-common.git"

fail=0
full_hashes=()

lock_sources=$(grep -o 'git+[^"]*ampc-common[^"]*' "$LOCK" | sort -u || true)
if [[ -z "$lock_sources" ]]; then
  echo "No ampc-common dependency in $LOCK — nothing to check."
  exit 0
fi

while IFS= read -r src; do
  if [[ "$src" =~ \#([0-9a-f]{40})$ ]]; then
    full_hashes+=("${BASH_REMATCH[1]}")
  else
    echo "::error::$LOCK has an ampc-common source with no resolved commit — a branch or tag pin, not a rev: $src"
    fail=1
  fi
done <<<"$lock_sources"

# Cargo.toml is the file cargo actually resolves when --locked is absent, so a
# manifest rev the lockfile does not know about means the lockfile is stale.
while IFS= read -r line; do
  if [[ "$line" =~ rev[[:space:]]*=[[:space:]]*\"([0-9a-f]{7,40})\" ]]; then
    manifest_rev="${BASH_REMATCH[1]}"
    matched=0
    for full in "${full_hashes[@]:-}"; do
      [[ "$full" == "$manifest_rev"* ]] && matched=1
    done
    if [[ $matched -eq 0 ]]; then
      echo "::error::$MANIFEST pins ampc-common at $manifest_rev but $LOCK does not resolve to it."
      echo "         Regenerate the lockfile — the builds do not use --locked, so they"
      echo "         would silently resolve the manifest pin instead."
      fail=1
    fi
  else
    echo "::error::$MANIFEST has an ampc-common dependency that is not pinned to a rev: $line"
    fail=1
  fi
done < <(grep -E 'ampc-common' "$MANIFEST" | grep -E '\bgit\b' || true)

unique=$(printf '%s\n' "${full_hashes[@]:-}" | sort -u | grep -v '^$' || true)
[[ -n "$unique" ]] || exit 1

if [[ $(wc -l <<<"$unique") -gt 1 ]]; then
  echo "::error::ampc-common crates resolve to different commits:"
  printf '  %s\n' $unique
  fail=1
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git init --quiet "$work"
git -C "$work" remote add origin "$UPSTREAM"
git -C "$work" fetch --quiet --filter=blob:none origin main:refs/remotes/origin/main

while IFS= read -r rev; do
  if ! git -C "$work" fetch --quiet --filter=blob:none origin "$rev" 2>/dev/null; then
    echo "::error::ampc-common commit $rev does not exist upstream."
    fail=1
    continue
  fi
  if git -C "$work" merge-base --is-ancestor "$rev" refs/remotes/origin/main; then
    echo "ok: ampc-common $rev is on main"
  else
    echo "::error::ampc-common commit $rev is not an ancestor of main."
    echo "         Either the ampc-common PR has not merged yet, or it was"
    echo "         squash-merged and this pin points at the pre-merge commit."
    echo "         Re-pin to the commit that landed on ampc-common main."
    fail=1
  fi
done <<<"$unique"

exit "$fail"
