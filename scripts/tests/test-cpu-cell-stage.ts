// Local-only validation: Helm rendering and DNS init execution with AWS stubbed, networking disabled.
import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, readFileSync, rmdirSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

process.chdir(fileURLToPath(new URL("../../", import.meta.url)));
const chart = process.env.COMMON_APP_CHART;
assert.ok(chart, "set COMMON_APP_CHART to worldcoin-foundation/common-app 2.7.4 or later");
const read = (path: string) => Bun.YAML.parse(readFileSync(path, "utf8")) as Record<string, any>;
const base = read("deploy/stage/common-values-ampc-hnsw.yaml");
const cells = read("deploy/stage/common-values-cpu-cell.yaml");
const execute = (cmd: string[]) => Bun.spawnSync({ cmd, timeout: 30_000 });
let dnsScript = "";
let dnsImage = "";
for (let party = 0; party < 3; party++) {
  const partyPath = `deploy/stage/ampc-hnsw-${party}-stage/values-ampc-hnsw.yaml`;
  const partyValues = read(partyPath);
  for (const slot of ["a", "b"]) {
    const name = `ampc-hnsw-${slot}`;
    const rendered = execute(["helm", "template", name, chart, "--namespace", "ampc-hnsw",
      "-f", "deploy/stage/common-values-ampc-hnsw.yaml", "-f", partyPath,
      "-f", "deploy/stage/common-values-cpu-cell.yaml", "-f", `deploy/stage/ampc-hnsw-${party}-stage/values-${name}.yaml`]);
    assert.equal(rendered.exitCode, 0, rendered.stderr.toString());
    const objects = rendered.stdout.toString().split(/^---\s*$/m).map(text => Bun.YAML.parse(text) as any).filter(Boolean);
    const deployment = objects.find(object => object.kind === "Deployment");
    assert.equal(deployment.metadata.name, name);
    assert.equal(deployment.spec.replicas, 0);
    assert.equal(deployment.spec.strategy.type, "Recreate");
    assert.deepEqual(deployment.spec.selector.matchLabels, {"app.kubernetes.io/name": name});
    const pod = deployment.spec.template.spec;
    const container = pod.containers[0];
    assert.equal(container.image, cells.image);
    assert.match(container.image, /^ghcr\.io\/worldcoin\/iris-mpc-linear-scan-amd64:[a-f0-9]{40}@sha256:[a-f0-9]{64}$/);
    assert.equal(pod.serviceAccountName, "ampc-hnsw");
    assert.deepEqual(container.resources, base.resources);
    assert.deepEqual(pod.nodeSelector, base.nodeSelector);
    assert.deepEqual(pod.tolerations, base.tolerations);
    assert.equal(objects.some(object => ["Service", "ServiceAccount"].includes(object.kind)), false);
    const env = Object.fromEntries(container.env.map((item: any) => [item.name, item.value ?? item.valueFrom]));
    assert.equal(env.SMPC__STARTUP_MODE, "candidate");
    assert.equal(env.SMPC__SERVER_COORDINATION__PARTY_ID, party.toString());
    assert.equal(Object.keys(env).some(key => key.startsWith("SMPC__CPU_CELL__")), false, "handoff must stay disabled");
    for (const key of ["SMPC__DATABASE__URL", "SMPC__REQUESTS_QUEUE_URL", "SMPC__RESULTS_TOPIC_ARN"])
      assert.deepEqual(env[key], base.env[key].valueFrom);
    for (const key of ["SMPC__TLS__PRIVATE_KEY", "SMPC__TLS__LEAF_CERT", "SMPC__TLS__ROOT_CERTS"])
      assert.equal(env[key], base.env[key]);
    const hosts = JSON.parse(env.SMPC__SERVER_COORDINATION__NODE_HOSTNAMES);
    assert.deepEqual(hosts, [0, 1, 2].map(peer => peer === party ? "0.0.0.0" : `pod-${slot}.${peer}.stage.hnsw.worldcoin.dev`));
    for (const [probe, endpoint] of [["livenessProbe", "health"], ["readinessProbe", "ready"], ["startupProbe", "startup"]]) {
      assert.equal(container[probe].httpGet, undefined);
      assert.equal(container[probe].exec.command.at(-1), `http://127.0.0.1:3090/${endpoint}`);
    }
    const tls = pod.volumes.find((volume: any) => volume.secret?.items?.some((item: any) => item.key === "certificate.crt"));
    assert.equal(tls.secret.secretName, "application");
    const policy = objects.find(object => object.kind === "SecurityGroupPolicy");
    assert.deepEqual(policy.spec.securityGroups.groupIds, partyValues.attachSecurityGroupPolicy.groupIds);
    assert.deepEqual(policy.spec.podSelector.matchLabels, deployment.spec.selector.matchLabels);
    const init = pod.initContainers[0];
    assert.equal(init.name, "cpu-cell-dns");
    const initEnv = Object.fromEntries(init.env.map((item: any) => [item.name, item.value ?? item.valueFrom]));
    assert.equal(initEnv.PARTY_ID, party.toString());
    assert.equal(initEnv.CELL_SLOT, slot);
    assert.deepEqual(initEnv.POD_IP, {fieldRef: {fieldPath: "status.podIP"}});
    const script = objects.find(object => object.kind === "ConfigMap").data["init.sh"];
    if (dnsScript) assert.equal(script, dnsScript);
    dnsScript = script;
    if (dnsImage) assert.equal(init.image, dnsImage);
    dnsImage = init.image;
  }
}
console.log("PASS: six inert slot renders; selectors, inherited DB/TLS/SA/SG, probes and resources");
assert.match(dnsScript, /timeout --signal=TERM --kill-after=10s 180s/);
const fixture = mkdtempSync(join(tmpdir(), "cpu-cell-dns-"));
chmodSync(fixture, 0o755);
writeFileSync(join(fixture, "init.sh"), dnsScript);
writeFileSync(join(fixture, "aws"), `#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> /fixture/aws.log
[[ "$AWS_MAX_ATTEMPTS" == 1 ]]
case "$1:$2" in
  route53:list-hosted-zones-by-name)
    if [[ "$MOCK" == no-zone ]]; then echo None; else echo /hostedzone/Zfixture; fi ;;
  route53:change-resource-record-sets)
    if [[ "$MOCK" == collision ]]; then exit 17; fi
    if [[ "$MOCK" == no-change-id ]]; then echo None; else echo /change/Cfixture; fi ;;
  route53:wait)
    [[ "$3" == resource-record-sets-changed ]]
    if [[ "$MOCK" == wait-error ]]; then exit 42; fi ;;
  *) exit 99 ;;
esac
`);
chmodSync(join(fixture, "aws"), 0o755);
try {
  for (const scenario of ["success", "collision", "no-zone", "no-change-id", "wait-error", "invalid-party"]) {
    writeFileSync(join(fixture, "aws.log"), "");
    chmodSync(join(fixture, "aws.log"), 0o666);
    const result = execute(["docker", "run", "--rm", "--network", "none", "--read-only", "--user", "65534:65534",
      "--mount", `type=bind,source=${fixture},target=/fixture`, "--entrypoint", "/bin/bash",
      "-e", "PATH=/fixture:/usr/local/bin:/usr/bin:/bin", "-e", `MOCK=${scenario}`,
      "-e", `PARTY_ID=${scenario === "invalid-party" ? "9" : "1"}`, "-e", "CELL_SLOT=b", "-e", "POD_IP=192.0.2.4",
      dnsImage, "/fixture/init.sh"]);
    assert.equal(result.exitCode === 0, scenario === "success", `${scenario}: ${result.stderr}`);
    const calls = readFileSync(join(fixture, "aws.log"), "utf8");
    assert.equal(calls.includes(" wait resource-record-sets-changed"), ["success", "wait-error"].includes(scenario), scenario);
    assert.equal(calls.includes("UPSERT"), false);
    assert.equal(calls.includes("DELETE"), false);
    if (scenario === "success") {
      assert.ok(calls.includes("--id /change/Cfixture"));
      const batch = JSON.parse(calls.match(/\{"Changes":.*\}/)![0]);
      assert.deepEqual(batch.Changes, [{Action: "CREATE", ResourceRecordSet: {
        Name: "pod-b.1.stage.hnsw.worldcoin.dev", Type: "A", TTL: 5, ResourceRecords: [{Value: "192.0.2.4"}],
      }}]);
    }
  }
} finally {
  for (const name of ["aws", "init.sh", "aws.log"]) unlinkSync(join(fixture, name));
  rmdirSync(fixture);
}
console.log("PASS: exact init image, stub AWS success/wait/error/collision cases; no AWS networking");
for (const slot of ["a", "b"]) {
  console.log(`${slot.toUpperCase()} serving / handoff source: ${[0, 1, 2].map(party => `http://pod-${slot}.${party}.stage.hnsw.worldcoin.dev:3000/handoff-status`).join(", ")}`);
  console.log(`${slot.toUpperCase()} candidate operators: ${[0, 1, 2].map(party => `http://pod-${slot}.${party}.stage.hnsw.worldcoin.dev:3090/handoff-status`).join(", ")}`);
}
console.log("LIVE PREFLIGHT STILL REQUIRED: certificate SAN/mTLS and HTTP trust boundary, SG reachability, DNS resolution, fenced retirement, capacity, compatible image and migrations.");
