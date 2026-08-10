#!/usr/bin/env bash
#
# Fails if this repo pins an ampc-common revision that is not reachable from
# ampc-common's main branch. Two states it catches:
#
#   1. The pin is a commit on an unmerged ampc-common branch, so main here would
#      build against a dependency that does not exist upstream.
#   2. The pin went stale while the PR waited. ampc-common squash-merges, so the
#      pre-merge commit is genuinely not reachable from main afterwards, and
#      restoring such a pin downgrades this repo's dependency.
#
# Reads Cargo.lock rather than Cargo.toml: the lockfile carries the resolved
# source for every ampc-common crate in one place, and a branch or tag pin shows
# up there as a source without a rev.
#
# Runs standalone: bash scripts/check-ampc-common-pin.sh [path/to/Cargo.lock]

set -euo pipefail

LOCK="${1:-Cargo.lock}"
UPSTREAM="https://github.com/worldcoin/ampc-common.git"

sources=$(grep -o 'git+[^"]*ampc-common[^"]*' "$LOCK" | sort -u || true)
if [[ -z "$sources" ]]; then
  echo "No ampc-common dependency in $LOCK — nothing to check."
  exit 0
fi

fail=0
revs=()
while IFS= read -r src; do
  if [[ "$src" =~ \?rev=([0-9a-f]{7,40}) ]]; then
    revs+=("${BASH_REMATCH[1]}")
  else
    echo "::error::ampc-common is not pinned to a revision: $src"
    fail=1
  fi
done <<<"$sources"

unique_revs=$(printf '%s\n' "${revs[@]:-}" | sort -u | grep -v '^$' || true)
if [[ -z "$unique_revs" ]]; then
  exit 1
fi

if [[ $(wc -l <<<"$unique_revs") -gt 1 ]]; then
  echo "::error::ampc-common crates disagree on a revision:"
  printf '  %s\n' $unique_revs
  fail=1
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git init --quiet "$work"
git -C "$work" remote add origin "$UPSTREAM"
git -C "$work" fetch --quiet --filter=blob:none origin main:refs/remotes/origin/main

while IFS= read -r rev; do
  if ! git -C "$work" fetch --quiet --filter=blob:none origin "$rev" 2>/dev/null; then
    echo "::error::ampc-common revision $rev does not exist upstream."
    fail=1
    continue
  fi
  if git -C "$work" merge-base --is-ancestor "$rev" refs/remotes/origin/main; then
    echo "ok: ampc-common $rev is on main"
  else
    echo "::error::ampc-common revision $rev is not an ancestor of main."
    echo "         Either the ampc-common PR has not merged yet, or it was"
    echo "         squash-merged and this pin now points at the pre-merge commit."
    echo "         Re-pin to the commit that landed on ampc-common main."
    fail=1
  fi
done <<<"$unique_revs"

exit "$fail"
