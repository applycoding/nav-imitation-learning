// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { gateAuditText, isReportV1, type ReportV1 } from "./schema.ts";

function reportWith(bcOodSuccesses: number, bcOodCollisions: number, canFail: [boolean, boolean]): ReportV1 {
  const suite = { n: 8, successes: bcOodSuccesses, collisions: bcOodCollisions };
  return {
    suites: { bc_ood: suite },
    dagger_gate_audit: {
      ood_successes_clause_can_fail: canFail[0],
      ood_collisions_clause_can_fail: canFail[1],
      binding_clauses: ["dagger_id_successes >= bc_id_successes"],
      id_clause: "4/8 >= 3/8",
      original_margin_met: false,
      original_margin_rule: "retired",
    },
  } as unknown as ReportV1;
}

describe("schema", () => {
  it("accepts schema v1", () => {
    expect(
      isReportV1({
        schema_version: 1,
        sample: true,
        honesty: "Simulation only.",
      }),
    ).toBe(true);
  });

  it("rejects a non-v1 object", () => {
    expect(isReportV1({ schema_version: 2, sample: false, honesty: "x" })).toBe(false);
    expect(isReportV1(null)).toBe(false);
  });

  it("states that OOD clauses cannot fail at the BC floor", () => {
    const text = gateAuditText(reportWith(0, 8, [false, false]));
    expect(text).toContain("cannot fail on this run");
    expect(text).toContain("4/8 >= 3/8");
    expect(text).toContain("was missed");
  });

  it("lists binding clauses when BC is off the floor", () => {
    const text = gateAuditText(reportWith(1, 7, [true, true]));
    expect(text).not.toContain("cannot fail");
    expect(text).toContain("binding DAgger clauses");
  });
});
