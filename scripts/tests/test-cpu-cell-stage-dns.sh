#!/usr/bin/env bash
set -euo pipefail

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export PATH="$tmp:$PATH" STATE="$tmp/state" PARTY_ID=1 CELL_SLOT=a POD_IP=10.0.0.1
unset DNS_DEPLOYMENT_ID DNS_PREVIOUS_DEPLOYMENT_ID FAIL_CLAIM
root=$(cd "$(dirname "$0")/../.." && pwd)
yq -r '.initContainer.configMap."init.sh"' "$root/deploy/stage/common-values-cpu-cell.yaml" > "$tmp/init"
# macOS lacks GNU timeout. No network operations take place in this fixture.
printf '%s\n' '#!/bin/sh' 'shift 3; exec "$@"' > "$tmp/timeout"
cat > "$tmp/aws" <<'AWS'
#!/usr/bin/env bash
set -euo pipefail
[[ $1 == route53 ]]
case "$2" in
    list-hosted-zones-by-name) echo /hostedzone/ZONE; exit 0;;
    wait) [[ $3 == resource-record-sets-changed && $5 == /change/CHANGE ]]; exit;;
    change-resource-record-sets) ;;
    *) exit 1;;
esac
shift 2
while [[ $1 != --change-batch ]]; do shift; done
batch=$2
echo "$batch" >> "$STATE.calls"
if [[ -n ${FAIL_CLAIM:-} ]]; then echo 'AccessDenied' >&2; exit 1; fi
# Apply the entire batch to a copy. Exact deletes and duplicate creates conflict.
if ! updated=$(jq -ec --argjson batch "$batch" '
    reduce $batch.Changes[] as $c (.;
        $c.ResourceRecordSet as $r | $r.Type as $t |
        if $r.Name!="pod-a.1.stage.hnsw.worldcoin.dev" or $r.TTL!=5 or ($r.ResourceRecords|length)!=1 then error("unexpected record")
        elif $c.Action=="DELETE" and .[$t]==$r then del(.[$t])
        elif $c.Action=="CREATE" and .[$t]==null then .[$t]=$r
        elif $c.Action=="UPSERT" and $t=="A" then .[$t]=$r
        else error("conflict") end)' "$STATE"); then
    echo 'An error occurred (InvalidChangeBatch) when calling the ChangeResourceRecordSets operation: conflict' >&2
    exit 1
fi
echo "$updated" > "$STATE"
echo /change/CHANGE
AWS
chmod +x "$tmp"/{aws,timeout}
reset_zone() { echo '{}' > "$STATE"; : > "$STATE.calls"; }
register() { bash "$tmp/init"; }
reject() {
    local before
    before=$(cat "$STATE")
    if register > "$tmp/error" 2>&1; then echo 'Unexpected registration success' >&2; exit 1; fi
    [[ $(cat "$STATE") == "$before" ]] || { echo 'Failed registration changed DNS' >&2; exit 1; }
}
check_owner() {
    jq -e --arg id "\"$DNS_DEPLOYMENT_ID\"" --arg ip "$POD_IP" \
        '.TXT.ResourceRecords[0].Value==$id and .A.ResourceRecords[0].Value==$ip' "$STATE" >/dev/null
}
first=11111111-1111-4111-8111-111111111111
second=22222222-2222-4222-8222-222222222222
third=33333333-3333-4333-8333-333333333333
reset_zone
register
jq -e '.TXT==null and .A.ResourceRecords[0].Value=="10.0.0.1"' "$STATE" >/dev/null
reject # Legacy CREATE alone never overwrites a collision.
export DNS_DEPLOYMENT_ID=$first
reject # An unmarked legacy A cannot be adopted.

reset_zone
export DNS_PREVIOUS_DEPLOYMENT_ID=$third
reject # Managed claims require their predecessor even when the slot is empty.
unset DNS_PREVIOUS_DEPLOYMENT_ID
register
check_owner
register # Same generation, same IP retries after an uncertain response.
check_owner
POD_IP=10.0.0.2
reject # Same generation, different IP cannot displace the first pod.
export DNS_PREVIOUS_DEPLOYMENT_ID=$first DNS_DEPLOYMENT_ID=$second
register
check_owner
register # Replacement replay after the predecessor marker has gone.
check_owner
export DNS_PREVIOUS_DEPLOYMENT_ID=$second DNS_DEPLOYMENT_ID=$third POD_IP=10.0.0.1
register
check_owner
export DNS_PREVIOUS_DEPLOYMENT_ID=$first DNS_DEPLOYMENT_ID=$second
reject # A late claim for a retired deployment, even when its IP matches today's pod.
POD_IP=10.0.0.9 reject

: > "$STATE.calls"
FAIL_CLAIM=1 reject
[[ $(wc -l < "$STATE.calls") -eq 1 ]]
grep -q AccessDenied "$tmp/error"
unset DNS_DEPLOYMENT_ID
reject # Predecessor alone.
export DNS_DEPLOYMENT_ID=$first
reject # Identical UUIDs.
DNS_DEPLOYMENT_ID=00000000-0000-0000-0000-000000000000 reject
DNS_PREVIOUS_DEPLOYMENT_ID=bad-id reject
echo 'CPU cell stage DNS registration tests passed'
