#!/usr/bin/env bash
# POP-4051 local repro: force one party into the unbounded poll_exact_messages
# wait (the silent batch-loop stall) against the local 3-party hawk stack.
#
# Mechanism (deterministic, no log-races):
#   1. All parties idle, queues empty.
#   2. Pause party 1. Party 0's next sync round will block retrying peer-state
#      fetches against the paused party — a seconds-wide window.
#   3. Send one uniqueness request (unique --rng-seed to defeat FIFO
#      content-based deduplication). Copies land on all three queues.
#   4. Party 0 reads its own count (1) and latches messages_to_poll=1 in its
#      peer-visible sticky state, then blocks on party 1's sync endpoint.
#   5. Steal party 0's queue copy (external ReceiveMessage, long visibility).
#   6. Unpause party 1 (total pause < 8s, under the 10s sync timeout).
#   7. Party 0 completes the sync round with target=1 latched and an EMPTY
#      queue -> waits in poll_exact_messages. Parties 1/2 poll their copies,
#      form the batch, and wait (loudly) in the batch-entries SHA sync.
#
# Verdicts:
#   STALL_REPRODUCED  party 0 entered poll (target 1) and did not finish within STALL_WAIT
#   ABORT_OBSERVED    fix fired: "aborting empty batch poll attempt" (EXPECT_ABORT=1)
#   RECOVERED         after releasing the stolen copy, party 0 finished the batch
#   NO_REPRO          timing lost — retried automatically
#
# Usage:
#   Baseline (fix disabled -> silent stall):
#     docker compose -f docker-compose.hawk.yaml up -d
#     ./scripts/tools/pop4051-repro.sh
#   Fix validation (bounded abort):
#     ABORT_SECS=10 docker compose -f docker-compose.hawk.yaml -f docker-compose.pop4051-repro.yaml up -d
#     EXPECT_ABORT=1 ./scripts/tools/pop4051-repro.sh
set -uo pipefail

export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_REGION=us-east-1
ENDPOINT=(--endpoint-url http://localhost:4566)
Q0="http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/smpcv2-0-dev.fifo"
COMPOSE="docker compose -f docker-compose.hawk.yaml"
P0=hawk_participant_0
P1=hawk_participant_1
HOLD_SECS="${HOLD_SECS:-600}"      # visibility timeout while holding party 0's stolen copy
STALL_WAIT="${STALL_WAIT:-25}"     # seconds of silence in poll that count as the stall
EXPECT_ABORT="${EXPECT_ABORT:-0}"  # 1 = expect the fix's abort instead of the silent stall
ATTEMPTS="${ATTEMPTS:-4}"
RECEIPT=""
CLIENT_PID=""

say() { printf '\n== [%s] %s ==\n' "$(date -u +%H:%M:%S)" "$*"; }
p_logs() { $COMPOSE logs --no-log-prefix --since "$1" "$2" 2>/dev/null; }

cleanup() {
  [ -n "$CLIENT_PID" ] && kill "$CLIENT_PID" 2>/dev/null
  docker unpause "$($COMPOSE ps -q $P1)" >/dev/null 2>&1
  release_message
}
trap cleanup EXIT

release_message() {
  [ -z "$RECEIPT" ] && return 0
  aws "${ENDPOINT[@]}" sqs change-message-visibility --queue-url "$Q0" \
    --receipt-handle "$RECEIPT" --visibility-timeout 0 >/dev/null 2>&1 || true
  RECEIPT=""
}

steal_message() {
  RECEIPT=""
  for _ in $(seq 1 8); do
    local out
    out=$(aws "${ENDPOINT[@]}" sqs receive-message --queue-url "$Q0" \
      --wait-time-seconds 1 --visibility-timeout "$HOLD_SECS" \
      --max-number-of-messages 1 --query 'Messages[0].ReceiptHandle' --output text 2>/dev/null)
    if [ -n "$out" ] && [ "$out" != "None" ]; then RECEIPT="$out"; return 0; fi
  done
  return 1
}

send_request() {
  # Unique rng seed per send: FIFO topic/queues use content-based deduplication,
  # so identical client payloads within 5 min are silently dropped.
  AWS_ENDPOINT_URL="http://127.0.0.1:4566" ./target/release/client \
    --request-topic-arn arn:aws:sns:us-east-1:000000000000:iris-mpc-input.fifo \
    --requests-bucket-name wf-smpcv2-dev-sns-requests \
    --public-key-base-url "http://localhost:4566/wf-dev-public-keys" \
    --response-queue-url http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/iris-mpc-results-us-east-1.fifo \
    --region us-east-1 \
    --n-batches 1 --batch-size 1 \
    --rng-seed "$(( (RANDOM << 15) ^ RANDOM ^ $(date +%s) ))" \
    >"/tmp/pop4051-client-$1.log" 2>&1 &
  CLIENT_PID=$!
}

end_client() { [ -n "$CLIENT_PID" ] && kill "$CLIENT_PID" 2>/dev/null; wait "$CLIENT_PID" 2>/dev/null; CLIENT_PID=""; }

attempt() {
  local n="$1" t0
  t0=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  say "attempt $n: pausing $P1 to widen party 0's sync window"
  docker pause "$($COMPOSE ps -q $P1)" >/dev/null || return 2
  local pause_started; pause_started=$(date +%s)

  say "sending one uniqueness request (unique rng seed)"
  send_request "$n"

  # Wait for party 0 to READ its queue count and latch the non-zero target in
  # its peer-visible sticky state — it logs this at info. Only then steal: with
  # party 1 paused, party 0 is now blocked in peer-sync, so the steal cannot
  # lose a race against its ReceiveMessage.
  say "waiting for party 0 to latch messages_to_poll=1"
  local latched=0
  for _ in $(seq 1 14); do
    if p_logs "$t0" $P0 | grep -q "messages_to_poll=1"; then latched=1; break; fi
    sleep 0.5
  done
  if [ "$latched" -ne 1 ]; then
    say "party 0 did not latch within window (was mid-round at pause); NO_REPRO"
    docker unpause "$($COMPOSE ps -q $P1)" >/dev/null
    end_client; return 1
  fi

  say "party 0 latched; stealing its queue copy (visibility ${HOLD_SECS}s)"
  if ! steal_message; then
    say "steal failed — party 0 grabbed its copy first; NO_REPRO"
    docker unpause "$($COMPOSE ps -q $P1)" >/dev/null
    end_client; return 1
  fi

  local pause_elapsed=$(( $(date +%s) - pause_started ))
  say "stole party 0's copy after ${pause_elapsed}s of pause; unpausing $P1"
  docker unpause "$($COMPOSE ps -q $P1)" >/dev/null
  if [ "$pause_elapsed" -ge 8 ]; then
    say "pause window ${pause_elapsed}s >= 8s — peers may have torn down; treating as NO_REPRO"
    release_message; end_client; return 1
  fi

  say "party 0 should now be in poll_exact_messages with an empty queue; observing ${STALL_WAIT}s"
  sleep "$STALL_WAIT"
  local p0; p0=$(p_logs "$t0" $P0)

  if ! echo "$p0" | grep -q "Polling SQS for up to 1"; then
    say "party 0 never entered the poll (agreement round missed); NO_REPRO"
    release_message; end_client; return 1
  fi

  if [ "$EXPECT_ABORT" = "1" ]; then
    if echo "$p0" | grep -q "aborting empty batch poll attempt"; then
      say "VERDICT: ABORT_OBSERVED — the fix bounded the wait"
    else
      say "VERDICT: FIX_MISS — party 0 in poll but no abort (ABORT_SECS set? < STALL_WAIT?)"
      echo "$p0" | tail -5; release_message; end_client; return 2
    fi
  else
    if echo "$p0" | grep -q "Finished polling SQS. Processed 1"; then
      say "party 0 finished its poll — steal lost the race; NO_REPRO"
      release_message; end_client; return 1
    fi
    say "VERDICT: STALL_REPRODUCED — party 0 silent in poll_exact_messages (target latched, queue empty)"
    echo "-- party 0 last lines:"; echo "$p0" | tail -3
    echo "-- peer 1 (formed batch, waiting in entries-sync):"
    p_logs "$t0" $P1 | grep -E "Finished polling|Batch sync entries|sync .*fail" | tail -3
  fi

  say "releasing party 0's copy (visibility -> 0); watching recovery"
  release_message
  for _ in $(seq 1 45); do
    if p_logs "$t0" $P0 | grep -q "Finished polling SQS. Processed 1"; then
      say "VERDICT: RECOVERED — party 0 consumed the released copy and completed the batch"
      end_client; return 0
    fi
    sleep 2
  done
  say "VERDICT: NO_RECOVERY within 90s — inspect logs manually"
  end_client; return 2
}

say "preflight: stack health + client binary"
$COMPOSE ps --format '{{.Service}} {{.Health}}' | grep -v healthy | grep -v '^$' && { echo "unhealthy services above — aborting"; exit 1; }
[ -x ./target/release/client ] || cargo build --release -q -p iris-mpc-bins --bin client || exit 1

for i in $(seq 1 "$ATTEMPTS"); do
  attempt "$i"; rc=$?
  [ "$rc" -eq 0 ] && { say "SUCCESS"; exit 0; }
  [ "$rc" -eq 2 ] && { say "FAILED (hard)"; exit 2; }
  say "retrying after cleanup"; sleep 8
done
say "no repro in $ATTEMPTS attempts"
exit 1
