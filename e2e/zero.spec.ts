import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
import type { Operation } from "../src/engine/types";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page
    .getByRole("navigation", { name: "Explore", exact: true })
    .getByRole("button", { name: /^ZeRO / })
    .click();
  await expect(
    page.getByRole("heading", { name: "ZeRO Stage Explorer", exact: true }),
  ).toBeVisible();
});

test("compares independent stages and distinguishes theoretical state from live memory", async ({
  page,
}) => {
  await expect(
    page.getByRole("combobox", { name: "Run A stage", exact: true }),
  ).toHaveValue("ddp");
  await expect(
    page.getByRole("combobox", { name: "Run B stage", exact: true }),
  ).toHaveValue("zero3");
  const rows = page.locator(".zero-summary tbody tr");
  await expect(rows.locator("td:last-child")).toHaveText([
    "16",
    "10",
    "7",
    "4",
  ]);
  await expect(page.locator(".zero-summary-note")).toContainText(
    "excludes activations and temporary buffers",
  );
  await page
    .getByRole("combobox", { name: "Run A stage", exact: true })
    .selectOption("zero1");
  await page
    .getByRole("combobox", { name: "Run B stage", exact: true })
    .selectOption("zero2");
  const runs = page.getByRole("article");
  await expect(runs.nth(0).locator(".run-heading h2")).toContainText("ZeRO-1");
  await expect(runs.nth(1).locator(".run-heading h2")).toContainText("ZeRO-2");
  await expect(runs.nth(0).locator(".zero-run-ownership dd")).toHaveText([
    "replicated",
    "replicated",
    "None",
    "1/4 shard",
  ]);
  await expect(runs.nth(1).locator(".zero-run-ownership dd")).toHaveText([
    "replicated",
    "1/4 shard after RS",
    "None",
    "1/4 shard",
  ]);
  await page
    .getByRole("combobox", { name: "Storage profile", exact: true })
    .selectOption("bf16-master");
  await expect(rows.locator("td:last-child")).toHaveText([
    "16",
    "7",
    "5.5",
    "4",
  ]);
  await expect(page.locator(".zero-profile-note")).toContainText(
    "without BF16 rounding",
  );
  await expect(runs.nth(0).locator(".memory-table")).toContainText(
    "FP32 master weights",
  );
  await expect(runs.nth(1).locator(".memory-table")).toContainText(
    "FP32 master weights",
  );
  await page
    .getByRole("combobox", { name: "Logical ranks", exact: true })
    .selectOption("8");
  await expect(rows.locator("td:last-child")).toHaveText([
    "16",
    "5.5",
    "3.75",
    "2",
  ]);
  await expect(
    page.getByRole("combobox", { name: "Run A stage", exact: true }),
  ).toHaveValue("zero1");
  await expect(
    page.getByRole("combobox", { name: "Run B stage", exact: true }),
  ).toHaveValue("zero2");
});

test("inspects updated replica synchronization and completes both ZeRO optimizer steps", async ({
  page,
}) => {
  await page
    .getByRole("combobox", { name: "Run A stage", exact: true })
    .selectOption("zero1");
  await page
    .getByRole("combobox", { name: "Run B stage", exact: true })
    .selectOption("zero2");
  await page
    .getByRole("combobox", { name: "Storage profile", exact: true })
    .selectOption("bf16-master");
  const runs = page.getByRole("article");
  for (let index = 0; index < 2; index++) {
    const run = runs.nth(index);
    await run.locator(".timeline-event.kind-all-gather").first().click();
    await expect(run.locator(".operation-details > h3")).toHaveText(
      "Updated parameter synchronization",
    );
    await expect(run.locator(".operation-details")).toContainText(
      "persistent model replica",
    );
    await run
      .getByRole("button", { name: "Jump to start", exact: true })
      .click();
    await expect(run.locator(".buffer-gathered")).toHaveCount(0);
    const picker = run.getByLabel("Event boundary", { exact: true });
    await picker.selectOption({
      index: (await picker.locator("option").count()) - 1,
    });
    await expect(run.locator(".phase-tag")).toHaveText("Complete");
    await expect(
      run.getByText("Matches single-process reference", { exact: true }),
    ).toBeVisible();
    await expect(run.locator(".rank-meta").first()).toContainText(
      "Full replica",
    );
  }
});

test("shares and round-trips the selected pair and profile, then cleanly leaves Explorer", async ({
  page,
  context,
}) => {
  await page
    .getByRole("combobox", { name: "Run A stage", exact: true })
    .selectOption("zero2");
  await page
    .getByRole("combobox", { name: "Run B stage", exact: true })
    .selectOption("zero1");
  await page
    .getByRole("combobox", { name: "Storage profile", exact: true })
    .selectOption("bf16-master");
  const downloading = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export JSON", exact: true }).click();
  const contents = await readFile((await (await downloading).path())!, "utf8");
  const exported = JSON.parse(contents);
  expect(exported.schemaVersion).toBe(3);
  expect(exported.inputConfiguration).toMatchObject({
    strategy: "zero2",
    precision: "bf16-master",
  });
  expect(exported.comparison.inputConfiguration).toMatchObject({
    strategy: "zero1",
    precision: "bf16-master",
  });
  expect(exported.results.numerical.maxParameterError).toBeLessThan(1e-10);
  expect(exported.comparison.results.numerical.maxParameterError).toBeLessThan(
    1e-10,
  );
  await page
    .getByRole("navigation", { name: "Explore", exact: true })
    .getByRole("button", { name: /^DDP / })
    .click();
  await expect(page.getByRole("article")).toHaveCount(1);
  await expect(page.locator(".batch-identity")).toContainText("FP32 Adam");
  await page.locator("input[type=file]").setInputFiles({
    name: "zero-trace.json",
    mimeType: "application/json",
    buffer: Buffer.from(contents),
  });
  await expect(
    page.getByRole("combobox", { name: "Run A stage", exact: true }),
  ).toHaveValue("zero2");
  await expect(
    page.getByRole("combobox", { name: "Run B stage", exact: true }),
  ).toHaveValue("zero1");
  await expect(
    page.getByRole("combobox", { name: "Storage profile", exact: true }),
  ).toHaveValue("bf16-master");
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page
    .getByRole("button", { name: "Share configuration", exact: true })
    .click();
  const shared = await page.evaluate(() => navigator.clipboard.readText());
  await page.goto(shared);
  await expect(
    page.getByRole("combobox", { name: "Run A stage", exact: true }),
  ).toHaveValue("zero2");
  await expect(
    page.getByRole("combobox", { name: "Run B stage", exact: true }),
  ).toHaveValue("zero1");
  await expect(
    page.getByRole("combobox", { name: "Storage profile", exact: true }),
  ).toHaveValue("bf16-master");
  await page.getByRole("button", { name: "Compare runs", exact: true }).click();
  await expect(page.getByRole("article")).toHaveCount(2);
  await expect(page.locator(".comparison-note")).toContainText("DDP vs. FSDP");
  await expect(page.locator(".batch-identity")).toContainText("FP32 Adam");
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("keeps ZeRO-2 reduction controls available and applies them to the exported schedule", async ({
  page,
}) => {
  for (const pair of [
    ["zero2", "zero3"],
    ["zero3", "zero2"],
    ["zero2", "zero2"],
  ]) {
    await page
      .getByRole("combobox", { name: "Run A stage", exact: true })
      .selectOption(pair[0]);
    await page
      .getByRole("combobox", { name: "Run B stage", exact: true })
      .selectOption(pair[1]);
    await page.getByRole("button", { name: "Configure", exact: true }).click();
    const dialog = page.getByRole("dialog");
    const bucket = dialog.getByRole("combobox", {
      name: "Layers per gradient bucket",
      exact: true,
    });
    const overlap = dialog.getByRole("checkbox", {
      name: "Overlap gradient reduction with backward",
      exact: true,
    });
    await expect(bucket).toBeVisible();
    await expect(overlap).toBeVisible();
    await bucket.selectOption("2");
    await overlap.uncheck();
    await dialog
      .getByRole("button", { name: "Apply settings", exact: true })
      .click();
    const downloading = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export JSON", exact: true })
      .click();
    const contents = await readFile(
      (await (await downloading).path())!,
      "utf8",
    );
    const exported = JSON.parse(contents);
    for (const run of [exported, exported.comparison]) {
      expect(run.inputConfiguration).toMatchObject({
        bucketLayers: 2,
        overlap: false,
      });
      if (run.inputConfiguration.strategy !== "zero2") continue;
      const operations: Operation[] = run.operations;
      const reductions = operations.filter(
        (operation) => operation.kind === "reduce-scatter",
      );
      expect(reductions).toHaveLength(2);
      expect(reductions.map((operation) => operation.layers.length)).toEqual([
        2, 2,
      ]);
      const backwardEnd = Math.max(
        ...operations
          .filter((operation) => operation.kind === "backward")
          .map((operation) => operation.end),
      );
      expect(
        Math.min(...reductions.map((operation) => operation.start)),
      ).toBeGreaterThanOrEqual(backwardEnd);
    }
  }
});

test("exposes gather controls and fits a narrow viewport", async ({ page }) => {
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  const dialog = page.getByRole("dialog");
  await expect(
    dialog.getByText("DDP / ZeRO-1 / ZeRO-2 gradient reduction", {
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    dialog.getByText("ZeRO-3 parameter gathering", { exact: true }),
  ).toBeVisible();
  await dialog.getByLabel("Reshard after forward", { exact: true }).uncheck();
  await dialog
    .getByRole("button", { name: "Apply settings", exact: true })
    .click();
  await expect(
    page.getByRole("article").nth(1).locator(".timeline-event.kind-all-gather"),
  ).toHaveCount(4);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("combobox", { name: "Run A stage", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("combobox", { name: "Run B stage", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("combobox", { name: "Storage profile", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(
    await page
      .locator(".zero-table-scroll")
      .evaluate((element) => element.scrollWidth > element.clientWidth),
  ).toBe(true);
  const table = page.locator(".zero-table-scroll");
  const eventPicker = page
    .getByRole("article")
    .first()
    .getByLabel("Event boundary", { exact: true });
  const eventBeforeScroll = await eventPicker.inputValue();
  await table.focus();
  await page.keyboard.press("ArrowRight");
  await expect
    .poll(() => table.evaluate((element) => element.scrollLeft))
    .toBeGreaterThan(0);
  await expect(eventPicker).toHaveValue(eventBeforeScroll);
  await page
    .getByRole("combobox", { name: "Run A stage", exact: true })
    .selectOption("zero2");
  await page
    .getByRole("combobox", { name: "Storage profile", exact: true })
    .selectOption("bf16-master");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});
