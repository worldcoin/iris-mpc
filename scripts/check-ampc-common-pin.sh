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
    # Cargo records the resolved hash after '#' even for branch/tag pins, so
    # this only fires on a malformed lockfile entry; real branch pins are
    # caught by the manifest walk below (no rev key).
    echo "::error::$LOCK has an ampc-common source with no resolved commit hash: $src"
    fail=1
  fi
done <<<"$lock_sources"

# Cargo.toml is the file cargo actually resolves when --locked is absent, so a
# manifest rev the lockfile does not know about means the lockfile is stale.
# A dependency comes in three shapes — an inline table on one line, a
# [dependencies.<name>] table with git/rev keys on separate lines, and dotted
# keys (<name>.git = "…") — so this walks the file instead of grepping for git
# and rev on the same line. Only the root manifest is walked; a workspace
# member pinning a divergent rev still trips the single-commit check on the
# lockfile below. Because the lockfile already proved an ampc-common dependency
# exists, finding none here is itself a failure — a parse miss must be loud,
# never a pass.
manifest_deps_checked=0
check_manifest_pin() {
  local context="$1" manifest_rev="$2"
  manifest_deps_checked=$((manifest_deps_checked + 1))
  if [[ -z "$manifest_rev" ]]; then
    echo "::error::$MANIFEST has an ampc-common dependency that is not pinned to a rev: $context"
    fail=1
    return
  fi
  if [[ ! "$manifest_rev" =~ ^[0-9a-fA-F]{4,40}$ ]]; then
    echo "::error::$MANIFEST pins ampc-common with a rev that is not a commit hash: $context"
    fail=1
    return
  fi
  manifest_rev=$(tr '[:upper:]' '[:lower:]' <<<"$manifest_rev")
  local matched=0 full
  for full in "${full_hashes[@]:-}"; do
    [[ "$full" == "$manifest_rev"* ]] && matched=1
  done
  if [[ $matched -eq 0 ]]; then
    echo "::error::$MANIFEST pins ampc-common at $manifest_rev but $LOCK does not resolve to it."
    echo "         Regenerate the lockfile — the builds do not use --locked, so they"
    echo "         would silently resolve the manifest pin instead."
    fail=1
  fi
}

table_header="(top of file)"
table_git=""
table_rev=""
dotted_name=""
flush_table() {
  if [[ "$table_git" == *ampc-common* ]]; then
    check_manifest_pin "$table_header" "$table_rev"
  fi
  table_git=""
  table_rev=""
  dotted_name=""
}

# TOML strings may be basic or literal, so both quote styles are accepted.
re_key="^[[:space:]]*(([A-Za-z0-9_-]+)\\.)?(git|rev)[[:space:]]*=[[:space:]]*[\"']([^\"']*)[\"']"
re_inline_git="[{].*git[[:space:]]*="
re_inline_rev="rev[[:space:]]*=[[:space:]]*[\"']([^\"']*)[\"']"

while IFS= read -r line; do
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  if [[ "$line" =~ ^[[:space:]]*\[ ]]; then
    flush_table
    table_header="$line"
    continue
  fi
  # Inline table, only when it closes on the same line (TOML forbids multiline
  # inline tables); an unclosed one falls through to key accumulation. Only
  # the part inside the braces is parsed: a trailing comment could otherwise
  # smuggle in a rev the check accepts while cargo follows an unpinned branch,
  # or drag an unrelated dependency into the ampc-common check.
  if [[ "$line" == *\}* ]]; then
    inline="${line%%\}*}"
    if [[ "$inline" == *ampc-common* && "$inline" =~ $re_inline_git ]]; then
      inline_rev=""
      [[ "$inline" =~ $re_inline_rev ]] && inline_rev="${BASH_REMATCH[1]}"
      check_manifest_pin "$line" "$inline_rev"
      continue
    fi
  fi
  if [[ "$line" =~ $re_key ]]; then
    dep_name="${BASH_REMATCH[2]}"
    dep_key="${BASH_REMATCH[3]}"
    dep_val="${BASH_REMATCH[4]}"
    # Dotted keys carry the dependency name in the prefix — a prefix change is
    # a new entry. Two dotted deps interleaved line-by-line would still
    # conflate, but the found-nothing check keeps a full miss loud.
    if [[ -n "$dep_name" && "$dep_name" != "$dotted_name" ]]; then
      flush_table
      dotted_name="$dep_name"
    fi
    if [[ "$dep_key" == "git" ]]; then
      table_git="$dep_val"
    else
      table_rev="$dep_val"
    fi
  fi
done <"$MANIFEST"
flush_table

if [[ $manifest_deps_checked -eq 0 ]]; then
  echo "::error::$LOCK resolves ampc-common but no ampc-common git dependency was found in $MANIFEST."
  echo "         Either the two files disagree, or the dependency is written in a form"
  echo "         this check does not parse — see scripts/check-ampc-common-pin.sh."
  fail=1
fi

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
