#!/usr/bin/env bash
set -euo pipefail

die() { echo "$*" >&2; exit 1; }
[[ $# == 1 ]] || die "Usage: $0 <ghcr.io/worldcoin/iris-mpc-linear-scan-amd64:COMMIT@sha256:DIGEST>"
image=$1
[[ $image =~ ^ghcr\.io/worldcoin/iris-mpc-linear-scan-amd64:[0-9a-f]{40}@sha256:[0-9a-f]{64}$ ]] ||
    die "Expected a stage linux/amd64 image pinned by Git SHA and digest"

root=$(git rev-parse --show-toplevel)
cd "$root"
stage=deploy/stage
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Read the same four layers as Argo. The lines compared across parties omit
# local identities only after checking them against the physical slot.
for slot in a b; do
    for party in 0 1 2; do
        dir=$stage/ampc-hnsw-$party-stage
        values=$dir/values-ampc-hnsw-$slot.yaml
        # shellcheck disable=SC2016 # $layer belongs to yq.
        yq ea -o=json '. as $layer ireduce ({}; . * $layer)' \
            "$stage/common-values-ampc-hnsw.yaml" "$dir/values-ampc-hnsw.yaml" \
            "$stage/common-values-cpu-cell.yaml" "$values" |
            jq -er --arg party "$party" --arg slot "$slot" '
                def uuid: type=="string" and test("^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$") and .!="00000000-0000-0000-0000-000000000000";
                def init_value($name): [.initContainer.env[] | select(.name==$name).value];
                .env as $e | $e.SMPC__CPU_CELL__DEPLOYMENT_ID as $id |
                $e.SMPC__CPU_CELL__HANDOFF__SOURCE_DEPLOYMENT_ID as $parent |
                $e.SMPC__CPU_CELL__HANDOFF__ID as $handoff |
                init_value("DNS_DEPLOYMENT_ID") as $owner |
                select(($id|uuid) and ($parent|uuid) and ($handoff|uuid) and ($id!=$parent)
                    and (.replicaCount==0 or .replicaCount==1)
                    and $e.SMPC__STARTUP_MODE=="candidate"
                    and $e.SMPC__SERVER_COORDINATION__PARTY_ID==$party
                    and $e.SMPC__CPU_CELL__PEERS__SLOT==$slot
                    and init_value("CELL_SLOT")==[$slot] and init_value("PARTY_ID")==[$party]
                    and ($owner==[$id] or ($owner==[] and init_value("DNS_PREVIOUS_DEPLOYMENT_ID")==[])))
                | [$id,$parent,$handoff,.replicaCount,.image,($owner[0]//"legacy")] | @tsv' > "$tmp/$slot-$party" ||
            die "Party $party slot $slot: invalid candidate identity or deployment configuration"
        cmp -s "$tmp/$slot-0" "$tmp/$slot-$party" || die "Party $party disagrees with party 0 about slot $slot"
    done
done

read -r a a_parent _ < "$tmp/a-0"
read -r b b_parent _ < "$tmp/b-0"
if [[ $a_parent == "$b" && $b_parent != "$a" ]]; then
    latest=a; previous=b
elif [[ $b_parent == "$a" && $a_parent != "$b" ]]; then
    latest=b; previous=a
else
    die "Expected exactly one slot to name the other deployment as its handoff source"
fi
read -r source_id _ last_handoff latest_count latest_image _ < "$tmp/$latest-0"
read -r _ _ _ previous_count _ previous_owner < "$tmp/$previous-0"
[[ $latest_count == 1 ]] || die "Expected one replica for the most recently configured slot $latest"
if [[ $previous_count == 1 ]]; then
    [[ $latest_image == "$image" ]] || die "Both slots are enabled and the configured rollout uses another image"
    echo "Already prepared $previous -> $latest: deployment $source_id, handoff $last_handoff"
    exit 0
fi
source=$latest
destination=$previous

deployment=$(uuidgen | tr '[:upper:]' '[:lower:]')
handoff=$(uuidgen | tr '[:upper:]' '[:lower:]')
dns_env=$(jq -cn --arg current "$deployment" --arg previous "$previous_owner" '
    [{name:"DNS_DEPLOYMENT_ID",value:$current}] +
    (if $previous=="legacy" then [] else [{name:"DNS_PREVIOUS_DEPLOYMENT_ID",value:$previous}] end)')
export image source_id deployment handoff dns_env
for party in 0 1 2; do
    values=$stage/ampc-hnsw-$party-stage/values-ampc-hnsw-$destination.yaml
    yq '.replicaCount = 1 | .image = strenv(image) |
        .env.SMPC__CPU_CELL__DEPLOYMENT_ID = strenv(deployment) |
        .env.SMPC__CPU_CELL__HANDOFF__ID = strenv(handoff) |
        .env.SMPC__CPU_CELL__HANDOFF__SOURCE_DEPLOYMENT_ID = strenv(source_id) |
        .initContainer.env |= (map(select(.name != "DNS_DEPLOYMENT_ID" and .name != "DNS_PREVIOUS_DEPLOYMENT_ID")) + env(dns_env))' "$values" > "$tmp/updated"
    diff -u -L "a/$values" -L "b/$values" "$values" "$tmp/updated" >> "$tmp/change.patch" || [[ $? == 1 ]]
done
git apply "$tmp/change.patch"
echo "Prepared $source -> $destination: deployment $deployment, handoff $handoff"
