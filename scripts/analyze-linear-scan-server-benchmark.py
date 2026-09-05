#!/usr/bin/env python3
"""Summarize production linear-scan cascade logs from all three parties.

The normal and mirror searches run concurrently.  For one logical request the
cluster therefore performs the sum of both orientations' comparisons, without
summing the work performed redundantly by the three MPC parties.

Cascade timestamps describe the search interval only; they cannot establish
client ingress or result-delivery times. Client results provide independent
monotonic SNS-publish-to-all-three-responses timings for true end-to-end rates.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


SUMMARY_MARKER = "LINEAR_SCAN_CASCADE_SUMMARY"
FIELD_RE = re.compile(r"([a-z_]+)=(\"[^\"]*\"|\S+)")
PARTY_RE = re.compile(r"(?:server|party)[-_]?([0-2])")
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="discard this many requests from each party and orientation",
    )
    parser.add_argument(
        "--minimum-cps",
        type=float,
        help="exit unsuccessfully when median cluster comparisons/s is lower",
    )
    parser.add_argument("--client-results", type=Path, help="service-client results with monotonic end-to-end timings")
    parser.add_argument("--json", type=Path, help="also write the summary as JSON")
    return parser.parse_args()


def party_from_path(path: Path) -> int:
    match = PARTY_RE.search(path.name)
    if match is None:
        raise ValueError(f"cannot infer party id from log filename: {path}")
    return int(match.group(1))


def parse_fields(line: str) -> dict[str, str]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        record = None
    if isinstance(record, dict) and record.get("message") == SUMMARY_MARKER:
        return {name: str(value) for name, value in record.items()}

    fields = {}
    for name, value in FIELD_RE.findall(line.split(SUMMARY_MARKER, 1)[1]):
        fields[name] = value.strip('"').rstrip(",")
    if match := TIMESTAMP_RE.search(line):
        fields["timestamp"] = match.group(0)
    return fields


def parse_timestamp(value: str) -> float:
    # Python's ISO parser accepts microseconds; tracing emits nanoseconds.
    value = re.sub(r"(\.\d{6})\d+", r"\1", value).replace("Z", "+00:00")
    return dt.datetime.fromisoformat(value).timestamp()


def load_cascades(paths: list[Path]) -> dict[int, dict[str, list[dict[str, str]]]]:
    cascades: dict[int, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in paths:
        party = party_from_path(path)
        with path.open(encoding="utf-8", errors="replace") as log:
            for line in log:
                if SUMMARY_MARKER not in line:
                    continue
                fields = parse_fields(line)
                orientation = fields.get("orientation")
                if orientation not in {"normal", "mirror"}:
                    raise ValueError(f"missing/invalid orientation in {path}: {line.rstrip()}")
                for required in ("total_comparisons", "elapsed_seconds"):
                    if required not in fields:
                        raise ValueError(f"missing {required} in {path}: {line.rstrip()}")
                cascades[party][orientation].append(fields)
    return cascades


def summarize(
    cascades: dict[int, dict[str, list[dict[str, str]]]], warmups: int
) -> dict[str, object]:
    if set(cascades) != {0, 1, 2}:
        raise ValueError(f"expected logs for parties 0, 1, and 2; got {sorted(cascades)}")

    orientations = set(cascades[0])
    if not orientations or any(set(cascades[party]) != orientations for party in range(3)):
        raise ValueError("parties have different orientation sets")

    retained: dict[int, dict[str, list[dict[str, str]]]] = defaultdict(dict)
    sample_counts = set()
    for party in range(3):
        for orientation in orientations:
            rows = cascades[party][orientation][warmups:]
            retained[party][orientation] = rows
            sample_counts.add(len(rows))
    if len(sample_counts) != 1:
        raise ValueError(f"party/orientation sample counts differ: {sorted(sample_counts)}")
    samples = sample_counts.pop()
    if samples == 0:
        raise ValueError("no samples remain after warm-up removal")

    synchronized_rates = []
    synchronized_durations = []
    end_to_end_rates = []
    end_to_end_durations = []
    cascade_start_skews = []
    comparisons_per_request = []
    for sample in range(samples):
        total_comparisons = 0
        slowest_duration = 0.0
        party_starts: list[float] = []
        completion_times: list[float] = []
        timestamps_available = True
        for orientation in sorted(orientations):
            party_comparisons = {
                int(retained[party][orientation][sample]["total_comparisons"])
                for party in range(3)
            }
            if len(party_comparisons) != 1:
                raise ValueError(
                    f"comparison disagreement in sample {sample}, {orientation}: "
                    f"{sorted(party_comparisons)}"
                )
            total_comparisons += party_comparisons.pop()
            slowest_duration = max(
                slowest_duration,
                *(
                    float(retained[party][orientation][sample]["elapsed_seconds"])
                    for party in range(3)
                ),
            )
        for party in range(3):
            starts = []
            for orientation in orientations:
                row = retained[party][orientation][sample]
                timestamp = row.get("timestamp")
                if timestamp is None:
                    timestamps_available = False
                    break
                end = parse_timestamp(timestamp)
                completion_times.append(end)
                starts.append(end - float(row["elapsed_seconds"]))
            if not timestamps_available:
                break
            # Both orientations are spawned for the same request.  The first
            # cascade to enter is the best observable party cascade-start timestamp.
            party_starts.append(min(starts))

        if timestamps_available:
            earliest_cascade_start = min(party_starts)
            latest_cascade_start = max(party_starts)
            completion = max(completion_times)
            cascade_start_skew = latest_cascade_start - earliest_cascade_start
            end_to_end_duration = completion - earliest_cascade_start
            synchronized_duration = completion - latest_cascade_start
        else:
            cascade_start_skew = 0.0
            end_to_end_duration = slowest_duration
            synchronized_duration = slowest_duration

        comparisons_per_request.append(total_comparisons)
        cascade_start_skews.append(cascade_start_skew)
        end_to_end_durations.append(end_to_end_duration)
        synchronized_durations.append(synchronized_duration)
        end_to_end_rates.append(total_comparisons / end_to_end_duration)
        synchronized_rates.append(total_comparisons / synchronized_duration)

    result: dict[str, object] = {
        "samples": samples,
        "warmup_requests": warmups,
        "orientations": sorted(orientations),
        "comparisons_per_request": comparisons_per_request,
        "cascade_start_skew_seconds": cascade_start_skews,
        "median_cascade_start_skew_seconds": statistics.median(cascade_start_skews),
        "max_cascade_start_skew_seconds": max(cascade_start_skews),
        "cascade_elapsed_seconds": end_to_end_durations,
        "cascade_comparisons_per_second": end_to_end_rates,
        "median_cascade_comparisons_per_second": statistics.median(end_to_end_rates),
        "synchronized_elapsed_seconds": synchronized_durations,
        "synchronized_comparisons_per_second": synchronized_rates,
        "median_comparisons_per_second": statistics.median(synchronized_rates),
        "min_comparisons_per_second": min(synchronized_rates),
        "max_comparisons_per_second": max(synchronized_rates),
    }
    return result


def main() -> int:
    args = parse_args()
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be non-negative")
    cascades = load_cascades(args.logs)
    result = summarize(cascades, args.warmup_requests)
    print(
        "REAL_SERVER_BENCH_RESULT "
        f"samples={result['samples']} "
        f"orientations={','.join(result['orientations'])} "
        f"median_synchronized_comparisons_per_second="
        f"{result['median_comparisons_per_second']:.3f} "
        f"min_synchronized_comparisons_per_second="
        f"{result['min_comparisons_per_second']:.3f} "
        f"max_synchronized_comparisons_per_second="
        f"{result['max_comparisons_per_second']:.3f} "
        f"median_cascade_comparisons_per_second="
        f"{result['median_cascade_comparisons_per_second']:.3f} "
        f"median_cascade_start_skew_seconds={result['median_cascade_start_skew_seconds']:.6f} "
        f"max_cascade_start_skew_seconds={result['max_cascade_start_skew_seconds']:.6f}"
    )
    if args.client_results is not None:
        all_records = json.loads(args.client_results.read_text(encoding="utf-8"))["records"]
        all_records.sort(key=lambda row: (row["batch_index"], row["batch_item_index"]))
        records = all_records[args.warmup_requests:]
        if len(records) != result["samples"]:
            raise ValueError("client and cascade sample counts differ")
        durations = [float(row["publish_to_all_responses_seconds"]) for row in records]
        if any(value <= 0 for value in durations):
            raise ValueError("client durations must be positive")
        rates = [count / duration for count, duration in zip(result["comparisons_per_request"], durations)]
        result.update(
            client_end_to_end_elapsed_seconds=durations,
            client_end_to_end_comparisons_per_second=rates,
            median_client_end_to_end_comparisons_per_second=statistics.median(rates),
            median_client_end_to_end_seconds=statistics.median(durations),
            client_prepare_to_all_responses_seconds=[row["prepare_to_all_responses_seconds"] for row in records],
            client_timing_scope="SNS publish start to all three result responses; includes queueing and persistence",
        )
        print("REAL_SERVER_CLIENT_E2E_RESULT "
              f"samples={len(durations)} median_seconds={statistics.median(durations):.6f} "
              f"median_comparisons_per_second={statistics.median(rates):.3f}")
        if len(all_records) > 1 and len({row["batch_index"] for row in all_records}) == 1:
            # A pipelined batch shares one monotonic publish-start timestamp.
            # Work/individual response latency is not throughput under queueing:
            # measure the completed work between the warm-up and final response.
            warmup_end = max(
                (float(row["publish_to_all_responses_seconds"])
                 for row in all_records[:args.warmup_requests]), default=0.0
            )
            if any(duration <= warmup_end for duration in durations):
                raise ValueError("retained responses overlap the warm-up completion boundary")
            window = max(durations) - warmup_end
            throughput = sum(result["comparisons_per_request"]) / window
            all_comparisons = sum(summarize(cascades, 0)["comparisons_per_request"])
            publish_duration = max(float(row["publish_to_all_responses_seconds"]) for row in all_records)
            prepare_duration = max(float(row["prepare_to_all_responses_seconds"]) for row in all_records)
            result.update(
                client_queued_completion_window_seconds=window,
                client_queued_comparisons_per_second=throughput,
                client_queued_timing_scope=(
                    "all-party completion of the warm-up through all-party completion "
                    "of the final queued request; includes persistence and delivery"
                ),
                client_entire_burst_comparisons=all_comparisons,
                client_entire_burst_publish_seconds=publish_duration,
                client_entire_burst_publish_comparisons_per_second=all_comparisons / publish_duration,
                client_entire_burst_prepare_seconds=prepare_duration,
                client_entire_burst_prepare_comparisons_per_second=all_comparisons / prepare_duration,
            )
            print("REAL_SERVER_CLIENT_QUEUED_RESULT "
                  f"samples={len(durations)} seconds={window:.6f} "
                  f"comparisons_per_second={throughput:.3f}")
            print("REAL_SERVER_CLIENT_ENTIRE_BURST_RESULT "
                  f"requests={len(all_records)} publish_seconds={publish_duration:.6f} "
                  f"publish_comparisons_per_second={all_comparisons / publish_duration:.3f} "
                  f"prepare_seconds={prepare_duration:.6f} "
                  f"prepare_comparisons_per_second={all_comparisons / prepare_duration:.3f}")
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if (
        args.minimum_cps is not None
        and result["median_comparisons_per_second"] < args.minimum_cps
    ):
        print(
            "REAL_SERVER_BENCH_BELOW_TARGET "
            f"actual={result['median_comparisons_per_second']:.3f} "
            f"minimum={args.minimum_cps:.3f}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
