import { test, expect, type Page } from "@playwright/test";
import { readFile, mkdir } from "node:fs/promises";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Parallel Training Lab", exact: true }),
  ).toBeVisible();
});

test("switches model ownership and finishes an actual sharded optimizer step", async ({
  page,
}) => {
  await expect(page.locator(".rank-card")).toHaveCount(4);
  await expect(page.getByText("Full replica", { exact: true })).toHaveCount(4);
  const initialParameter = await page
    .locator(".numerical-inspector dd")
    .nth(1)
    .innerText();
  await page.getByRole("button", { name: "FSDP", exact: true }).click();
  await expect(page.getByText("Shard 1 of 4", { exact: true })).toBeVisible();
  await expect(page.getByText("Shard 4 of 4", { exact: true })).toBeVisible();
  await expect(page.locator(".timeline-event.kind-all-gather")).toHaveCount(8);
  await expect(page.locator(".timeline-event.kind-reduce-scatter")).toHaveCount(
    4,
  );
  const picker = page.getByLabel("Event boundary", { exact: true });
  await picker.selectOption({
    index: (await picker.locator("option").count()) - 1,
  });
  await expect(page.locator(".phase-tag")).toHaveText("Complete");
  await expect(page.locator(".numerical-inspector dd").nth(1)).not.toHaveText(
    initialParameter,
  );
  await expect(
    page.getByText("Matches single-process reference", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".buffer-gathered")).toHaveCount(0);
});

test("replays event boundaries exactly and keeps playback separate from simulated scheduling", async ({
  page,
}) => {
  const picker = page.getByLabel("Event boundary", { exact: true });
  await picker.selectOption({ index: 17 });
  const cards = await page.locator(".rank-grid").innerText();
  const arithmetic = await page.locator(".numerical-inspector").innerText();
  await page
    .getByRole("button", { name: "Previous event", exact: true })
    .click();
  await expect(picker).toHaveValue("16");
  await page.getByRole("button", { name: "Next event", exact: true }).click();
  await expect(picker).toHaveValue("17");
  await expect(page.locator(".rank-grid")).toHaveText(cards, {
    useInnerText: true,
  });
  await expect(page.locator(".numerical-inspector")).toHaveText(arithmetic, {
    useInnerText: true,
  });
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await expect(picker).toHaveValue("0");
  await page.getByLabel("Playback speed", { exact: true }).selectOption("4");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect.poll(() => picker.inputValue()).not.toBe("0");
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  const paused = await picker.inputValue();
  await page.waitForTimeout(130);
  await expect(picker).toHaveValue(paused);
  await expect(page.locator(".metrics").first()).toContainText("modeled ms");
});

const scenarios = [
  "DDP vs. FSDP",
  "Reshard vs. retain",
  "Prefetch tradeoff",
  "One slow rank",
];
for (const scenario of scenarios) {
  test(`${scenario} compares at one shared time with independent inspection`, async ({
    page,
  }) => {
    await page
      .getByRole("button", {
        name: new RegExp(scenario.replaceAll(".", "\\.")),
      })
      .click();
    const runs = page.getByRole("article");
    await expect(runs).toHaveCount(2);
    await expect(
      page
        .locator(".batch-identity")
        .getByText(
          "32 samples = 8 local × 4 data ranks × 1 accumulation step",
          {
            exact: true,
          },
        ),
    ).toBeVisible();
    const pickers = page.getByLabel("Event boundary", { exact: true });
    await runs.nth(0).locator(".timeline-event").last().click();
    await expect(runs.nth(0).locator(".operation-details > h3")).toBeVisible();
    const leftTitle = await runs
      .nth(0)
      .locator(".operation-details > h3")
      .innerText();
    await runs.nth(1).locator(".timeline-event.kind-forward").first().click();
    await expect(runs.nth(0).locator(".operation-details > h3")).toHaveText(
      leftTitle,
    );
    await expect(runs.nth(1).locator(".operation-details > h3")).toHaveText(
      "Forward compute",
    );
    await expect(pickers.nth(0)).toHaveValue("0");
    await expect(pickers.nth(1)).toHaveValue("0");
    await runs
      .nth(1)
      .getByRole("button", { name: "Jump to end", exact: true })
      .click();
    const clock = Number(
      await page
        .locator(".scrubber > div > span")
        .last()
        .locator("strong")
        .innerText(),
    );
    expect(clock).toBeGreaterThan(0);
    const selectedBoundaries = await pickers.evaluateAll((elements) =>
      elements.map(
        (element) =>
          (element as HTMLSelectElement).selectedOptions[0].textContent,
      ),
    );
    const times = selectedBoundaries.map((value) =>
      Number(value?.split(" · ")[1].replace(" ms", "")),
    );
    expect(times[0]).toBeLessThanOrEqual(clock + 0.006);
    expect(Math.abs(times[1] - clock)).toBeLessThan(0.006);
    const cursors = page.locator(".chart-cursor");
    expect(await cursors.nth(0).getAttribute("x1")).toEqual(
      await cursors.nth(1).getAttribute("x1"),
    );
    await page
      .getByRole("combobox", { name: "Step through", exact: true })
      .selectOption("0");
    await page.getByRole("button", { name: "Next event", exact: true }).click();
    await expect(
      page.getByRole("combobox", { name: "Step through", exact: true }),
    ).toHaveValue("0");
  });
}

test("a multi-layer bucket highlights every participating layer and tensor", async ({
  page,
}) => {
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await page
    .getByRole("dialog")
    .getByRole("combobox", { name: "Layers per gradient bucket", exact: true })
    .selectOption("2");
  await page
    .getByRole("button", { name: "Apply settings", exact: true })
    .click();
  await page.locator(".timeline-event.kind-all-reduce").first().click();
  await expect(page.locator(".layer-node.selected")).toHaveCount(2);
  await expect(page.locator(".tensor-selected")).toHaveCount(8);
  await expect(
    page.locator(".timeline-event.kind-backward:not(.unrelated)"),
  ).toHaveCount(8);
  await expect(page.locator(".dependency-details")).toContainText("Rank 3");
});

test("collective inspection exports logical MiB sizes to NCCL Lab", async ({
  page,
}) => {
  await page.getByRole("button", { name: "FSDP", exact: true }).click();
  await page.locator(".timeline-event.kind-all-gather").first().click();
  const link = page.getByRole("link", {
    name: "Explore this collective in NCCL Lab",
    exact: true,
  });
  const url = new URL((await link.getAttribute("href"))!);
  expect(url.hostname).toBe("nccl-lab.pages.dev");
  expect(Object.fromEntries(url.searchParams)).toMatchObject({
    op: "AllGather",
    n: "4",
    size: "8",
    alg: "Ring",
  });
  await expect(page.locator(".collective-details")).toContainText(
    "total gathered output",
  );
  await page
    .getByRole("button", { name: "Jump to start", exact: true })
    .click();
  await expect(page.locator(".buffer-gathered")).toHaveCount(4);
  await expect(page.locator(".state-chip")).toHaveText("Executing");
  await page.locator(".timeline-event.kind-reduce-scatter").first().click();
  const scatter = new URL((await link.getAttribute("href"))!);
  expect(scatter.searchParams.get("op")).toBe("ReduceScatter");
  expect(scatter.searchParams.get("size")).toBe("8");
  await expect(page.locator(".collective-details")).toContainText(
    "total input before scattering",
  );
  await page.getByRole("button", { name: "DDP", exact: true }).click();
  await page.locator(".timeline-event.kind-all-reduce").first().click();
  const reduce = new URL((await link.getAttribute("href"))!);
  expect(reduce.searchParams.get("op")).toBe("AllReduce");
  expect(reduce.searchParams.get("size")).toBe("8");
  expect(reduce.searchParams.get("red")).toBe("Average");
});

test("exports both runs, imports locally, rejects malformed data, and shares a reproducible URL", async ({
  page,
  context,
}) => {
  await page.getByRole("button", { name: /Prefetch tradeoff/ }).click();
  const downloaded = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export JSON", exact: true }).click();
  const download = await downloaded;
  const contents = await readFile((await download.path())!, "utf8");
  const exported = JSON.parse(contents);
  expect(exported.schemaVersion).toBe(2);
  expect(exported.events.length).toBeGreaterThan(50);
  expect(exported.memoryHistory.length).toBe(exported.events.length);
  expect(exported.comparison.inputConfiguration.prefetch).toBe(1);
  expect(exported.inputConfiguration.prefetch).toBe(0);
  expect(exported.results.numerical.maxParameterError).toBeLessThan(1e-10);
  await page.getByRole("button", { name: "DDP", exact: true }).click();
  await page.locator("input[type=file]").setInputFiles({
    name: "trace.json",
    mimeType: "application/json",
    buffer: Buffer.from(contents),
  });
  await expect(page.getByRole("status")).toContainText("regenerated locally");
  await expect(page.getByRole("article")).toHaveCount(2);
  await expect(page.locator(".comparison-note")).toContainText(
    "Prefetch tradeoff",
  );
  await page.locator("input[type=file]").setInputFiles({
    name: "bad.json",
    mimeType: "application/json",
    buffer: Buffer.from('{"schemaVersion":999}'),
  });
  await expect(page.getByRole("alert")).toContainText(
    "Unsupported export schema",
  );
  await expect(page.getByRole("article")).toHaveCount(2);
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page
    .getByRole("button", { name: "Share configuration", exact: true })
    .click();
  await expect(page.getByRole("status")).toContainText("link copied");
  const shared = await page.evaluate(() => navigator.clipboard.readText());
  const copied = new URL(shared);
  expect(copied.searchParams.get("v")).toBe("2");
  const savedSession = JSON.parse(copied.searchParams.get("config")!);
  expect(savedSession.config.seed).toBe(42);
  expect(savedSession.scenario).toBe("prefetch");
  await page.goto(shared);
  await expect(page.getByRole("article")).toHaveCount(2);
  await expect(page.locator(".comparison-note")).toContainText(
    "Prefetch tradeoff",
  );
});

test("configuration enforces batch validity, retained buffers, and visible OOM", async ({
  page,
}) => {
  await page.getByRole("button", { name: "FSDP", exact: true }).click();
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await dialog.getByLabel("Global batch", { exact: true }).fill("5");
  await dialog
    .getByRole("button", { name: "Apply settings", exact: true })
    .click();
  await expect(dialog.getByRole("alert")).toContainText("divide evenly");
  await expect(
    page
      .locator(".batch-identity")
      .getByText("32 samples = 8 local × 4 data ranks × 1 accumulation step", {
        exact: true,
      }),
  ).toBeVisible();
  await dialog.getByLabel("Global batch", { exact: true }).fill("32");
  await dialog.getByLabel("Reshard after forward", { exact: true }).uncheck();
  await expect(
    dialog.getByRole("combobox", { name: /^Backward prefetch/ }),
  ).toBeDisabled();
  await dialog.getByLabel("Capacity per rank · MiB", { exact: true }).fill("1");
  await dialog
    .getByRole("button", { name: "Apply settings", exact: true })
    .click();
  await expect(dialog).not.toBeVisible();
  await expect(page.locator(".timeline-event.kind-all-gather")).toHaveCount(4);
  await expect(page.getByRole("status")).toContainText("exceeded capacity");
  await expect(page.locator(".rank-memory").first()).toContainText("60.0 MiB");
  await expect(page.locator(".rank-memory").first()).toContainText("OOM");
});

test("keyboard shortcuts do not override focused inputs and one rank remains coherent", async ({
  page,
}) => {
  const picker = page.getByLabel("Event boundary", { exact: true });
  await page.locator("h1").click();
  await page.keyboard.press("ArrowRight");
  await expect(picker).toHaveValue("1");
  await page.keyboard.press("ArrowLeft");
  await expect(picker).toHaveValue("0");
  await page.keyboard.press("Space");
  await expect(
    page.getByRole("button", { name: "Pause", exact: true }),
  ).toBeVisible();
  await page.keyboard.press("Space");
  await expect(
    page.getByRole("button", { name: "Play", exact: true }),
  ).toBeVisible();
  await page.keyboard.press("r");
  await expect(picker).toHaveValue("0");
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  const input = page
    .getByRole("dialog")
    .getByLabel("Dataset & parameter seed", { exact: true });
  await input.focus();
  await page.keyboard.press("ArrowRight");
  await expect(picker).toHaveValue("0");
  await page.keyboard.press("Escape");
  await page
    .getByRole("combobox", { name: "Logical ranks", exact: true })
    .selectOption("1");
  await page.getByRole("button", { name: "FSDP", exact: true }).click();
  await expect(page.locator(".rank-card")).toHaveCount(1);
  await page.locator(".timeline-event.kind-all-gather").first().click();
  await expect(
    page.getByText(
      "One rank makes this collective a no-op. NCCL Lab requires at least two ranks.",
      { exact: true },
    ),
  ).toBeVisible();
  await expect(
    page.getByRole("link", {
      name: "Explore this collective in NCCL Lab",
      exact: true,
    }),
  ).toHaveCount(0);
  await picker.selectOption({
    index: (await picker.locator("option").count()) - 1,
  });
  await expect(page.locator(".phase-tag")).toHaveText("Complete");
});

async function assertNoPageOverflow(page: Page) {
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
}

test("desktop and narrow mobile remain usable without browser errors", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Parallel Training Lab", exact: true }),
  ).toBeVisible();
  await mkdir(".audit/screenshots", { recursive: true });
  await assertNoPageOverflow(page);
  await page.screenshot({
    path: ".audit/screenshots/app-desktop.png",
    fullPage: true,
    animations: "disabled",
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await assertNoPageOverflow(page);
  await page.screenshot({
    path: ".audit/screenshots/app-mobile.png",
    fullPage: true,
    animations: "disabled",
  });
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  const chooser = page.waitForEvent("filechooser");
  await page
    .locator(".file-actions")
    .getByRole("button", { name: "Import JSON", exact: true })
    .click();
  await (
    await chooser
  ).setFiles({
    name: "invalid.json",
    mimeType: "application/json",
    buffer: Buffer.from("{}"),
  });
  await expect(page.getByRole("alert")).toBeVisible();
  await page.getByRole("button", { name: /DDP vs\. FSDP/ }).click();
  await assertNoPageOverflow(page);
  await expect(page.getByRole("article")).toHaveCount(2);
  await expect(page.locator(".timeline-scroll").first()).toBeVisible();
  const dimensions = await page
    .locator(".timeline-scroll")
    .first()
    .evaluate((element) => ({
      width: element.clientWidth,
      scroll: element.scrollWidth,
    }));
  expect(dimensions.scroll).toBeGreaterThan(dimensions.width);
  expect(errors).toEqual([]);
});
