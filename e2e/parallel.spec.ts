import { test, expect } from "@playwright/test";
import { mkdir, readFile } from "node:fs/promises";

for (const strategy of ["TP", "PP", "EP"]) {
  test(`${strategy} completes, inspects communication, and exports its numerical result`, async ({
    page,
  }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");
    await page.getByRole("button", { name: strategy, exact: true }).click();
    await expect(page.locator(".strategy-overview")).toBeVisible();
    await expect(page.locator(".rank-card")).toHaveCount(4);
    const communication = page.locator(
      `.timeline-event.kind-${strategy === "TP" ? "all-reduce" : strategy === "PP" ? "send-recv" : "all-to-all"}`,
    );
    expect(await communication.count()).toBeGreaterThan(0);
    await communication.first().click();
    await expect(page.locator(".operation-details")).toBeVisible();
    if (strategy === "TP") {
      await expect(page.locator(".operation-details")).toContainText("SUM");
      const link = page.getByRole("link", {
        name: "Explore this collective in NCCL Lab",
      });
      expect(
        new URL((await link.getAttribute("href"))!).searchParams.get("red"),
      ).toBe("Sum");
      await page
        .getByRole("button", { name: "Jump to start", exact: true })
        .click();
      await expect(page.locator(".phase-tag")).toHaveText(
        "Forward · partial sums",
      );
    }
    const picker = page.getByLabel("Event boundary", { exact: true });
    await picker.selectOption({
      index: (await picker.locator("option").count()) - 1,
    });
    await expect(page.locator(".phase-tag")).toHaveText("Complete");
    await expect(
      page.getByText("Matches single-process reference", { exact: true }),
    ).toBeVisible();
    const values = await page.locator(".numerical-inspector").innerText();
    await page
      .getByRole("button", { name: "Previous event", exact: true })
      .click();
    await page.getByRole("button", { name: "Next event", exact: true }).click();
    await expect(page.locator(".numerical-inspector")).toHaveText(values, {
      useInnerText: true,
    });
    const download = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export JSON", exact: true })
      .click();
    const contents = JSON.parse(
      await readFile((await (await download).path())!, "utf8"),
    );
    expect(contents.inputConfiguration.strategy).toBe(strategy.toLowerCase());
    expect(contents.results.numerical.maxParameterError).toBeLessThan(1e-10);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.emulateMedia({ reducedMotion: "reduce" });
    await expect(
      page.getByRole("button", { name: "Export JSON", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Share configuration", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByLabel("All 7 comparisons", { exact: true }),
    ).toBeVisible();
    const renderedAxisFont = await page
      .locator(".memory-plot > svg")
      .evaluate((svg) => {
        const text = svg.querySelector("text")!;
        return (
          (parseFloat(getComputedStyle(text).fontSize) *
            svg.getBoundingClientRect().width) /
          (svg as SVGSVGElement).viewBox.baseVal.width
        );
      });
    expect(renderedAxisFont).toBeGreaterThanOrEqual(10);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await page.getByRole("button", { name: "Reset", exact: true }).click();
    await page
      .getByRole("button", { name: "Clear focus", exact: true })
      .click();
    await page
      .getByRole("status")
      .getByRole("button", { name: "Close", exact: true })
      .click();
    await page.locator("h1").click();
    await page.evaluate(() => window.scrollTo(0, 0));
    await mkdir(".audit/expansion/screenshots", { recursive: true });
    await page.screenshot({
      path: `.audit/expansion/screenshots/${strategy.toLowerCase()}-mobile.png`,
      fullPage: true,
    });
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({
      path: `.audit/expansion/screenshots/${strategy.toLowerCase()}-desktop.png`,
      fullPage: true,
    });
    expect(errors).toEqual([]);
  });
}

for (const scenario of [
  "FSDP vs. TP",
  "GPipe microbatches",
  "MoE distribution",
]) {
  test(`${scenario} compares a fixed workload`, async ({ page }) => {
    await page.goto("/");
    await page
      .getByRole("button", {
        name: new RegExp(scenario.replaceAll(".", "\\.")),
      })
      .click();
    await expect(page.getByRole("article")).toHaveCount(2);
    await expect(page.locator(".comparison-note")).toContainText(scenario);
    const pickers = page.getByLabel("Event boundary", { exact: true });
    for (let i = 0; i < 2; i++) {
      const picker = pickers.nth(i);
      await picker.selectOption({
        index: (await picker.locator("option").count()) - 1,
      });
      await expect(
        page.getByRole("article").nth(i).locator(".phase-tag"),
      ).toHaveText("Complete");
    }
    await page.setViewportSize({ width: 390, height: 844 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  });
}

test("pipeline control changes real microbatch events and unsupported stages are rejected", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "PP", exact: true }).click();
  const before = await page.locator(".timeline-event.kind-forward").count();
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await page
    .getByRole("dialog")
    .getByLabel("Pipeline microbatches")
    .selectOption("2");
  await page
    .getByRole("button", { name: "Apply settings", exact: true })
    .click();
  await expect(page.locator(".timeline-event.kind-forward")).toHaveCount(
    before / 2,
  );
  await expect(
    page
      .getByLabel("Logical ranks", { exact: true })
      .locator("option[value='8']"),
  ).toHaveCount(0);
});
