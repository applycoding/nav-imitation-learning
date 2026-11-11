// SPDX-License-Identifier: Apache-2.0

import { drawScene } from "./draw.ts";
import { isReportV1, SUITE_ORDER, type ReportV1, type Trajectory } from "./schema.ts";

const HONESTY_LINES = [
  "Simulation only",
  "Synthetic maps and a scripted expert",
  "Recorded CI output; this page does not train a policy.",
] as const;

const banner = document.querySelector<HTMLParagraphElement>("#banner");
const statusEl = document.querySelector<HTMLElement>("#status");
const sampleNote = document.querySelector<HTMLElement>("#sample-note");
const table = document.querySelector<HTMLTableElement>("#suites");
const runInfo = document.querySelector<HTMLElement>("#run-info");
const select = document.querySelector<HTMLSelectElement>("#episode-select");
const canvas = document.querySelector<HTMLCanvasElement>("#map");
const ctx = canvas?.getContext("2d");

function showError(message: string): void {
  if (!statusEl) {
    return;
  }
  statusEl.hidden = false;
  statusEl.textContent = message;
}

function fillHonesty(): void {
  if (banner) {
    banner.textContent = HONESTY_LINES.map((l) => (l.endsWith(".") ? l : `${l}.`)).join(" ");
  }
}

async function loadReport(): Promise<ReportV1> {
  const res = await fetch(`${import.meta.env.BASE_URL}report.json`);
  if (!res.ok) {
    throw new Error(`report.json missing (${res.status})`);
  }
  const data: unknown = await res.json();
  if (!isReportV1(data)) {
    throw new Error("invalid report schema");
  }
  return data;
}

function renderTable(report: ReportV1): void {
  if (!table) {
    return;
  }
  table.innerHTML = "";
  const head = document.createElement("tr");
  for (const col of ["suite", "successes", "n", "collisions", "timeouts", "success_rate", "pass"]) {
    const th = document.createElement("th");
    th.textContent = col;
    head.appendChild(th);
  }
  table.appendChild(head);
  for (const name of SUITE_ORDER) {
    const s = report.suites[name];
    const row = document.createElement("tr");
    row.dataset.suite = name;
    const cells = [
      name,
      String(s.successes),
      String(s.n),
      String(s.collisions),
      String(s.timeouts),
      s.success_rate.toFixed(3),
      String(s.pass),
    ];
    for (const val of cells) {
      const td = document.createElement("td");
      td.textContent = val;
      row.appendChild(td);
    }
    table.appendChild(row);
  }
}

function fillEpisodes(trajs: Trajectory[]): void {
  if (!select) {
    return;
  }
  select.innerHTML = "";
  const sorted = [...trajs].sort((a, b) => a.seed - b.seed);
  for (const t of sorted) {
    const opt = document.createElement("option");
    opt.value = `${t.suite}:${t.seed}`;
    opt.textContent = `${t.suite} #${t.seed}`;
    select.appendChild(opt);
  }
  select.onchange = () => {
    const id = select.value;
    const found = sorted.find((t) => `${t.suite}:${t.seed}` === id);
    if (found && ctx && canvas) {
      drawScene(ctx, canvas.width, canvas.height, found);
    }
  };
  if (sorted[0] && ctx && canvas) {
    drawScene(ctx, canvas.width, canvas.height, sorted[0]);
  }
}

async function boot(): Promise<void> {
  fillHonesty();
  try {
    const report = await loadReport();
    if (report.sample) {
      if (sampleNote) {
        sampleNote.hidden = false;
        sampleNote.textContent = "Sample report (not a CI run)";
      }
    }
    if (runInfo) {
      runInfo.textContent = `git_sha ${report.git_sha}, DAgger checkpoint round ${report.dagger_selected_round} (selected on train maps)`;
    }
    renderTable(report);
    fillEpisodes(report.trajectories);
  } catch (err) {
    const msg = err instanceof Error ? err.message : "failed to load report";
    showError(msg);
  }
}

void boot();
