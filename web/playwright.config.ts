// SPDX-License-Identifier: Apache-2.0
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "tests",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL: "http://127.0.0.1:4173/nav-imitation-learning/",
    trace: "on-first-retry",
  },
  webServer: {
    command: "npx vite preview --host 127.0.0.1 --port 4173 --base /nav-imitation-learning/",
    port: 4173,
    reuseExistingServer: !process.env.CI,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
