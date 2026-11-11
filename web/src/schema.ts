// SPDX-License-Identifier: Apache-2.0

export type SuiteName =
  | "expert_id"
  | "expert_ood"
  | "bc_id"
  | "bc_ood"
  | "dagger_id"
  | "dagger_ood"
  | "mutant_shuffle"
  | "mutant_zero";

export interface SuiteMetrics {
  n: number;
  successes: number;
  collisions: number;
  invalids: number;
  timeouts: number;
  success_rate: number;
  collision_rate: number;
  mean_time_success_s: number | null;
  expected: "pass" | "fail";
  observed: "pass" | "fail";
  pass: boolean;
}

export interface DaggerGateAudit {
  ood_successes_clause_can_fail: boolean;
  ood_collisions_clause_can_fail: boolean;
  binding_clauses: string[];
  id_clause: string;
  original_margin_met: boolean;
  original_margin_rule: string;
}

export interface ReportV1 {
  schema_version: number;
  sample: boolean;
  git_sha: string;
  run_id: string;
  honesty: string;
  dagger_selected_round: number;
  dagger_gate_audit?: DaggerGateAudit;
  open_loop_mse: Record<string, number>;
  suites: Record<SuiteName, SuiteMetrics>;
  episodes: EpisodeRecord[];
  trajectories: Trajectory[];
}

export interface EpisodeRecord {
  suite: string;
  seed: number;
  status: string;
  successes?: number;
}

export interface Trajectory {
  suite: string;
  seed: number;
  status: string;
  start: [number, number, number];
  goal: [number, number];
  xy: [number, number][];
  obstacles: [number, number, number, number][];
  border: number;
}

export const SUITE_ORDER: SuiteName[] = [
  "expert_id",
  "expert_ood",
  "bc_id",
  "bc_ood",
  "dagger_id",
  "dagger_ood",
  "mutant_shuffle",
  "mutant_zero",
];

export function gateAuditText(report: ReportV1): string {
  const a = report.dagger_gate_audit;
  if (!a) {
    return "";
  }
  const bc = report.suites.bc_ood;
  const parts: string[] = [];
  if (!a.ood_successes_clause_can_fail && !a.ood_collisions_clause_can_fail) {
    parts.push(
      `Gate audit: BC fails all ${bc.n} OOD maps with ${bc.collisions} collisions, so the two OOD no-regression inequalities are reported but cannot fail on this run.`,
    );
  } else {
    parts.push(`Gate audit: binding DAgger clauses on this run: ${a.binding_clauses.join("; ")}.`);
  }
  parts.push(`The binding in-distribution clause is DAgger ${a.id_clause} BC.`);
  parts.push(
    a.original_margin_met
      ? "The original OOD margin (DAgger 0.25 above BC) was met."
      : "The original OOD margin (DAgger 0.25 above BC) was missed.",
  );
  return parts.join(" ");
}

export function isReportV1(data: unknown): data is ReportV1 {
  if (typeof data !== "object" || data === null) {
    return false;
  }
  const r = data as ReportV1;
  return r.schema_version === 1 && typeof r.sample === "boolean" && typeof r.honesty === "string";
}
