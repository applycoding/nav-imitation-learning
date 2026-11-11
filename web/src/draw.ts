// SPDX-License-Identifier: Apache-2.0

import type { Trajectory } from "./schema.ts";

const WORLD = 10;
const CLEAR = "#f4f6f8";
const OBSTACLE = "#4a5568";
const TRAJ = "#c53030";
const START = "#2f855a";
const GOAL = "#2b6cb0";

export function drawScene(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  traj: Trajectory,
): number {
  const scale = width / WORLD;
  ctx.fillStyle = CLEAR;
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = OBSTACLE;
  const b = traj.border;
  ctx.fillRect(0, 0, WORLD * scale, b * scale);
  ctx.fillRect(0, (WORLD - b) * scale, WORLD * scale, b * scale);
  ctx.fillRect(0, 0, b * scale, WORLD * scale);
  ctx.fillRect((WORLD - b) * scale, 0, b * scale, WORLD * scale);
  // World +y is North; canvas y grows downward, so flip y.
  const sy = (y: number): number => (WORLD - y) * scale;
  for (const [x, y, w, h] of traj.obstacles) {
    ctx.fillRect(x * scale, sy(y + h), w * scale, h * scale);
  }
  if (traj.xy.length >= 2) {
    ctx.strokeStyle = TRAJ;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(traj.xy[0][0] * scale, sy(traj.xy[0][1]));
    for (let i = 1; i < traj.xy.length; i++) {
      ctx.lineTo(traj.xy[i][0] * scale, sy(traj.xy[i][1]));
    }
    ctx.stroke();
  }
  ctx.fillStyle = START;
  ctx.beginPath();
  ctx.arc(traj.start[0] * scale, sy(traj.start[1]), 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = GOAL;
  ctx.beginPath();
  ctx.arc(traj.goal[0] * scale, sy(traj.goal[1]), 6, 0, Math.PI * 2);
  ctx.fill();
  const img = ctx.getImageData(0, 0, width, height);
  let diff = 0;
  for (let i = 0; i < img.data.length; i += 4) {
    if (img.data[i] !== 244 || img.data[i + 1] !== 246 || img.data[i + 2] !== 248) {
      diff++;
    }
  }
  return diff;
}
