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

export interface ReportV1 {
  schema_version: number;
  sample: boolean;
  git_sha: string;
  run_id: string;
  honesty: string;
  dagger_selected_round: number;
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

export function isReportV1(data: unknown): data is ReportV1 {
  if (typeof data !== "object" || data === null) {
    return false;
  }
  const r = data as ReportV1;
  return r.schema_version === 1 && typeof r.sample === "boolean" && typeof r.honesty === "string";
}
