import copy
import importlib.util
import json
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "stage-messaging-cutover" / "ingestion-reset.py"
spec = importlib.util.spec_from_file_location("ingestion_reset", SCRIPT)
reset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reset)


class StageIngestionResetTest(unittest.TestCase):
    def setUp(self):
        self.cronjob = {
            "metadata": {"namespace": "ampc-hnsw"},
            "spec": {"jobTemplate": {"spec": {"template": {
                "metadata": {"labels": {"app.kubernetes.io/name": "retention"}},
                "spec": {
                    "serviceAccountName": "retention",
                    "containers": [{
                        "name": "reaper",
                        "image": "ghcr.io/worldcoin/retention-reaper:v1@sha256:" + "a" * 64,
                        "env": [
                            {"name": "RETENTION__DB_SCHEMA", "value": "SMPC_stage_1"},
                            {"name": "RETENTION__DB_URL", "valueFrom": {"secretKeyRef": {
                                "name": "application", "key": "DATABASE_AURORA_HNSW_URL"}}},
                            {"name": "RETENTION_JOBS", "value": '[{"table":"modifications"}]'},
                        ],
                    }],
                },
            }}}},
        }

    def test_dry_run_preserves_networking_and_does_not_mutate_cronjob(self):
        before = copy.deepcopy(self.cronjob)
        job = reset.make_job(self.cronjob, 1, "dry-run", "test")
        self.assertEqual(self.cronjob, before)
        self.assertEqual(job["spec"]["template"]["metadata"]["labels"],
                         before["spec"]["jobTemplate"]["spec"]["template"]["metadata"]["labels"])
        self.assertEqual(job["spec"]["template"]["spec"]["serviceAccountName"], "retention")
        self.assertEqual(self.env(job)["RETENTION__DRY_RUN"]["value"], "true")
        self.assertNotIn("schedule", job["spec"])

    def test_delete_replaces_all_rules_and_disables_retries(self):
        job = reset.make_job(self.cronjob, 1, "delete", "test")
        self.assertEqual(self.env(job)["RETENTION__DRY_RUN"]["value"], "false")
        self.assertEqual(json.loads(self.env(job)["RETENTION_JOBS"]["value"]), reset.RULE)
        self.assertEqual(job["spec"]["backoffLimit"], 0)
        self.assertEqual(job["spec"]["template"]["spec"]["restartPolicy"], "Never")

    def test_rejects_wrong_party_or_database_secret(self):
        with self.assertRaises(ValueError):
            reset.make_job(self.cronjob, 0, "delete", "test")
        env = self.cronjob["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]["env"]
        env[1]["valueFrom"]["secretKeyRef"]["key"] = "OTHER_DATABASE"
        with self.assertRaises(ValueError):
            reset.make_job(self.cronjob, 1, "delete", "test")

    def test_rejects_extra_containers(self):
        self.cronjob["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"].append({})
        with self.assertRaises(ValueError):
            reset.make_job(self.cronjob, 1, "delete", "test")

    @staticmethod
    def env(job):
        return {x["name"]: x for x in job["spec"]["template"]["spec"]["containers"][0]["env"]}


if __name__ == "__main__":
    unittest.main()
