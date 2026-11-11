// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { isReportV1 } from "./schema.ts";

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
});
