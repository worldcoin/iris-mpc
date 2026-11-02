#!/usr/bin/env python3
"""Generate a stage-only, one-off ingestion reaper Job without submitting it."""

import argparse
import copy
from datetime import datetime, timezone
import json
import subprocess
import sys

ACCOUNTS = {0: "024848486749", 1: "024848486818", 2: "024848486770"}
NAMESPACE = "ampc-hnsw"
CRONJOB = "ampc-hnsw-modifications-retention"
RULE = [{"table": "ingested_requests", "ts_column": "received_at",
         "retention": "0 seconds", "batch_size": 2000, "batch_pause_ms": 250}]


def make_job(cronjob, party, mode, suffix):
    if cronjob["metadata"]["namespace"] != NAMESPACE:
        raise ValueError("Expected the ampc-hnsw retention namespace")
    spec = copy.deepcopy(cronjob["spec"]["jobTemplate"]["spec"])
    containers = spec["template"]["spec"]["containers"]
    if len(containers) != 1:
        raise ValueError("Expected one reaper container")
    container = containers[0]
    env = {item["name"]: item for item in container.get("env", [])}
    if env.get("RETENTION__DB_SCHEMA", {}).get("value") != f"SMPC_stage_{party}":
        raise ValueError("Unexpected party database schema; inspect before continuing")
    expected_secret = {"name": "application", "key": "DATABASE_AURORA_HNSW_URL"}
    if env.get("RETENTION__DB_URL", {}).get("valueFrom", {}).get("secretKeyRef") != expected_secret:
        raise ValueError("Expected the existing HNSW database secret reference")
    if not container["image"].startswith("ghcr.io/worldcoin/retention-reaper:") or "@sha256:" not in container["image"]:
        raise ValueError("Expected an immutable retention-reaper image")
    replacements = {
        "RETENTION__ENABLED": "true",
        "RETENTION__DRY_RUN": str(mode == "dry-run").lower(),
        "RETENTION_JOBS": json.dumps(RULE, separators=(",", ":")),
    }
    for name, value in replacements.items():
        env[name] = {"name": name, "value": value}
    container["env"] = list(env.values())
    spec["backoffLimit"] = 0
    spec["activeDeadlineSeconds"] = 1800
    spec["ttlSecondsAfterFinished"] = 86400
    spec["template"]["spec"]["restartPolicy"] = "Never"
    # Preserve pod labels: the existing SecurityGroupPolicy selects this template.
    spec["template"].setdefault("metadata", {}).get("annotations", {}).pop(
        "argocd.argoproj.io/tracking-id", None
    )
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {
            "name": f"iris-ingestion-{mode}-{party}-{suffix}", "namespace": NAMESPACE,
            "labels": {"pop.worldcoin.org/purpose": "iris-ingestion-reset"},
        },
        "spec": spec,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--party", type=int, choices=ACCOUNTS, required=True)
    parser.add_argument("--mode", choices=("dry-run", "delete"), default="dry-run")
    args = parser.parse_args()
    context = f"arn:aws:eks:eu-central-1:{ACCOUNTS[args.party]}:cluster/ampc-hnsw-{args.party}-stage"
    cronjob = json.loads(subprocess.check_output([
        "kubectl", "--context", context, "--request-timeout=30s", "-n", NAMESPACE,
        "get", "cronjob", CRONJOB, "-o", "json",
    ], text=True))
    job = make_job(cronjob, args.party, args.mode,
                   datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
    print(f"Generated {args.mode} Job for {context}; no Job was submitted.", file=sys.stderr)
    print(json.dumps(job, indent=2))


if __name__ == "__main__":
    main()
