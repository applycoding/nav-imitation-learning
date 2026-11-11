// SPDX-License-Identifier: Apache-2.0

import { expect, test } from "@playwright/test";

const HONESTY = [
  "Simulation only",
  "Synthetic maps and a scripted expert",
  "Recorded CI output; this page does not train a policy.",
];

test.describe("dashboard", () => {
  test("shows honesty strings and suite row for locked seed", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        errors.push(msg.text());
      }
    });
    await page.goto("./");
    await expect(page).toHaveTitle(/simulation only/i);
    for (const line of HONESTY) {
      await expect(page.getByText(line, { exact: false })).toBeVisible();
    }
    await expect(page.getByText("does not demonstrate a DAgger advantage", { exact: false })).toBeVisible();
    const res = await page.request.get("./report.json");
    expect(res.ok()).toBe(true);
    const report = (await res.json()) as {
      git_sha: string;
      suites: Record<string, { successes: number; n: number; collisions: number; timeouts: number }>;
      episodes: { suite: string; seed: number; status: string }[];
    };
    await expect(page.locator("#run-info")).toContainText(report.git_sha);
    const audit = (report as unknown as { dagger_gate_audit: { id_clause: string } }).dagger_gate_audit;
    await expect(page.locator("#gate-audit")).toContainText(audit.id_clause);
    await expect(page.locator("#gate-audit")).toContainText(/Gate audit:/);
    for (const [name, s] of Object.entries(report.suites)) {
      const cells = page.locator(`tr[data-suite="${name}"] td`);
      await expect(cells).toHaveText([
        name,
        String(s.successes),
        String(s.n),
        String(s.collisions),
        String(s.timeouts),
        /^[0-9.]+$/,
        /^(true|false)$/,
      ]);
    }
    const locked = report.episodes.find((e) => e.suite === "bc_id" && e.seed === 21001);
    expect(locked).toBeDefined();
    const bcIdSuccesses = report.episodes.filter((e) => e.suite === "bc_id" && e.status === "success").length;
    await expect(page.locator('tr[data-suite="bc_id"] td').nth(1)).toHaveText(String(bcIdSuccesses));
    const canvas = page.locator("#map");
    const buf = await canvas.evaluate((el: HTMLCanvasElement) => {
      const ctx = el.getContext("2d");
      if (!ctx) {
        return "";
      }
      const data = ctx.getImageData(0, 0, el.width, el.height).data;
      let diff = 0;
      for (let i = 0; i < data.length; i += 4) {
        if (data[i] !== 244 || data[i + 1] !== 246 || data[i + 2] !== 248) {
          diff++;
        }
      }
      return String(diff);
    });
    expect(Number(buf)).toBeGreaterThan(0);
    expect(errors).toEqual([]);
  });

  test("missing report shows error without sample fallback", async ({ page, context }) => {
    await context.route("**/report.json", (route) => route.fulfill({ status: 404, body: "missing" }));
    await page.goto("./");
    await expect(page.locator("#status")).toContainText(/report\.json missing/i);
    await expect(page.locator("#sample-note")).toBeHidden();
  });
});
