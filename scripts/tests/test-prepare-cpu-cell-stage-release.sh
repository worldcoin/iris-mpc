#!/usr/bin/env bash
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
prepare=$repo/scripts/prepare-cpu-cell-stage-release.sh
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
git init -q
stage=deploy/stage
mkdir -p "$stage"
cp "$repo/$stage"/common-values-{ampc-hnsw,cpu-cell}.yaml "$stage/"
a=00000000-0000-4000-8000-000000000001
b=00000000-0000-4000-8000-000000000002
former=00000000-0000-4000-8000-000000000003
image="ghcr.io/worldcoin/iris-mpc-linear-scan-amd64:$(printf '1%.0s' {1..40})@sha256:$(printf '2%.0s' {1..64})"
for party in 0 1 2; do
    dir=$stage/ampc-hnsw-$party-stage
    mkdir -p "$dir"
    cp "$repo/$dir"/values-ampc-hnsw{,-a,-b}.yaml "$dir/"
    for slot in a b; do
        file=$dir/values-ampc-hnsw-$slot.yaml
        if [[ $slot == a ]]; then
            count=1 id=$a parent=$b
        else
            count=0 id=$b parent=$former
        fi
        count=$count id=$id parent=$parent yq -i '
            .env.SMPC__STARTUP_MODE="candidate" | .replicaCount=env(count) |
            .env.SMPC__CPU_CELL__DEPLOYMENT_ID=strenv(id) |
            .env.SMPC__CPU_CELL__HANDOFF__SOURCE_DEPLOYMENT_ID=strenv(parent) |
            .env.SMPC__CPU_CELL__HANDOFF__ID="00000000-0000-4000-8000-000000000004" |
            .startupProbe.failureThreshold=421' "$file"
        if [[ $slot == a ]]; then
            id=$id yq -i '.initContainer.env += [{"name":"DNS_DEPLOYMENT_ID","value":strenv(id)}]' "$file"
        fi
    done
done
git add deploy

reject() {
    git diff --binary > "$tmp/before"
    if "$prepare" "$@" > "$tmp/output" 2>&1; then echo "Expected refusal: $*" >&2; exit 1; fi
    git diff --binary > "$tmp/after"
    cmp "$tmp/before" "$tmp/after"
}
reject ghcr.io/worldcoin/iris-mpc-linear-scan-amd64:latest
last=$stage/ampc-hnsw-2-stage/values-ampc-hnsw-b.yaml
yq -i '.replicaCount=1' "$last"
reject "$image"
git restore -- "$last"
yq -i '.env.SMPC__STARTUP_MODE="active"' "$last"
reject "$image"
git restore -- "$last"
yq -i '(.initContainer.env[]|select(.name=="CELL_SLOT").value)="a"' "$last"
reject "$image"
git restore -- "$last"

"$prepare" "$image"
[[ $(git diff --name-only | wc -l) -eq 3 ]]
git diff --exit-code -- ':(glob)deploy/stage/*/values-ampc-hnsw-a.yaml'
b_new=$(yq -r '.env.SMPC__CPU_CELL__DEPLOYMENT_ID' "$last")
handoff=$(yq -r '.env.SMPC__CPU_CELL__HANDOFF__ID' "$last")
[[ $b_new != "$b" && $b_new != "$a" && $handoff != "$b_new" ]]
for party in 0 1 2; do
    file=$stage/ampc-hnsw-$party-stage/values-ampc-hnsw-b.yaml
    yq -o=json "$file" | jq -e --arg id "$b_new" --arg parent "$a" --arg handoff "$handoff" --arg image "$image" '
        .replicaCount==1 and .startupProbe.failureThreshold==421 and .image==$image and
        .env.SMPC__CPU_CELL__DEPLOYMENT_ID==$id and
        .env.SMPC__CPU_CELL__HANDOFF__SOURCE_DEPLOYMENT_ID==$parent and
        ([.initContainer.env[]|select(.name=="DNS_DEPLOYMENT_ID").value]==[$id]) and
        ([.initContainer.env[]|select(.name=="DNS_PREVIOUS_DEPLOYMENT_ID")]==[]) and
        .env.SMPC__CPU_CELL__HANDOFF__ID==$handoff' >/dev/null
done
git diff > "$tmp/prepared"
"$prepare" "$image"
git diff > "$tmp/retry"
cmp "$tmp/prepared" "$tmp/retry"
reject "${image%?}3"

# A circular history cannot identify the prepared rollout, even for the same image.
for file in "$stage"/ampc-hnsw-*-stage/values-ampc-hnsw-a.yaml; do
    id=$b_new yq -i '.env.SMPC__CPU_CELL__HANDOFF__SOURCE_DEPLOYMENT_ID=strenv(id)' "$file"
done
reject "$image"
git restore -- ':(glob)deploy/stage/*/values-ampc-hnsw-a.yaml'

for party in 0 1 2; do
    yq -i '.replicaCount=0' "$stage/ampc-hnsw-$party-stage/values-ampc-hnsw-a.yaml"
done
git add deploy
"$prepare" "$image"
[[ $(git diff --name-only | wc -l) -eq 3 ]]
git diff --exit-code -- ':(glob)deploy/stage/*/values-ampc-hnsw-b.yaml'
for party in 0 1 2; do
    file=$stage/ampc-hnsw-$party-stage/values-ampc-hnsw-a.yaml
    yq -o=json "$file" | jq -e --arg id "$b_new" --arg old "$a" '.replicaCount==1 and .env.SMPC__CPU_CELL__HANDOFF__SOURCE_DEPLOYMENT_ID==$id and
        ([.initContainer.env[]|select(.name=="DNS_DEPLOYMENT_ID").value]==[.env.SMPC__CPU_CELL__DEPLOYMENT_ID]) and
        ([.initContainer.env[]|select(.name=="DNS_PREVIOUS_DEPLOYMENT_ID").value]==[$old])' >/dev/null
done
echo "Stage release preparation checks passed"
