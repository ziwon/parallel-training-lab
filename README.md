# Parallel Training Lab

See what each GPU rank owns, computes, and waits for during one training step.

Parallel Training Lab makes distributed training inspectable: follow parameters through forward and backward passes, watch communication unlock the next operation, and see why memory rises and falls. Explore **DDP, FSDP, TP, PP, and EP** entirely in your browser. No GPU, CUDA, model downloads, or backend required.

**[Open Parallel Training Lab](https://parallel-lab.pages.dev/)** · [Run locally](#run-locally) · [Model reference](docs/model.md) · [NCCL Lab](https://nccl-lab.pages.dev/)

[![Four FSDP ranks at the start of a parameter all-gather. Each rank owns a shard and reserves a temporary full-layer buffer; the inspector explains the dependency and the timeline shows the training step.](assets/readme/fsdp-all-gather.png)](assets/readme/fsdp-all-gather.png)

_Four-rank FSDP, paused as the first all-gather starts. Gathered buffers occupy memory before their parameters are ready to compute with. Click the screenshot for the full-size view._

## Follow your first training step

1. [Open the app](https://parallel-lab.pages.dev/), select **FSDP**, and keep four ranks with a global batch of 32.
2. Select the first **AG** event and choose **Jump to start**. Each rank still owns a parameter shard, but now reserves a complete layer buffer.
3. Step forward. The layer waits for all-gather to finish; backward later produces gradients that reduce-scatter returns to their owners. Adam updates the owned parameters.
4. Open **DDP vs. FSDP** from **All 7 comparisons**. Both runs use the same training problem. Compare their ownership, communication, and memory peaks, then inspect the numerical result.

Use `Space` to play or pause, `←` / `→` to step, and `R` to reset. Scrub the timeline to inspect any simulated time. Comparison runs share a clock while their event inspectors remain independent.

## Supported strategies

| Strategy                               | What a rank owns                          | Communication                                              |
| -------------------------------------- | ----------------------------------------- | ---------------------------------------------------------- |
| **DDP** — Distributed data parallel    | A complete model and optimizer replica    | Average local gradients with AllReduce                     |
| **FSDP** — Fully sharded data parallel | Parameter, gradient, and optimizer shards | AllGather parameters; ReduceScatter gradients              |
| **TP** — Tensor parallel               | Alternating column and row matrix slices  | Sum partial activations and input gradients with AllReduce |
| **PP** — Pipeline parallel             | Consecutive whole-layer stages            | Send activations forward and gradients backward            |
| **EP** — Expert parallel               | Complete experts in a top-1 MoE           | AllToAll dispatch and combine in both directions           |

DDP, FSDP, TP, and PP share the same four-layer dense problem. EP uses a separate eight-expert model and compares against its one-rank execution. FSDP shards model state; each rank still computes complete layers on local samples. TP partitions the layer computation itself.

## What you can explore

- **Ownership and memory.** Select ranks and tensors to inspect replicas, shards, temporary buffers, activations, gradients, and Adam states. Current and peak allocations are tracked per rank; exceeding capacity produces an OOM marker.
- **Dependencies and overlap.** Follow compute and communication lanes to see when inputs become ready, why ranks wait, and when work can overlap.
- **Seven comparisons.** Try DDP versus FSDP, reshard versus retain, backward prefetch, one slow rank, FSDP versus TP, GPipe microbatches, and MoE distribution.
- **Numerical results.** Compare gradients and updated parameters with a single-process reference. Small seeded matrices perform the arithmetic; virtual tensor sizes drive memory and timing.
- **Reproducible experiments.** Share a versioned configuration and seed, or export events, memory history, assumptions, and results as JSON. Import regenerates the simulation locally from validated inputs.

## Run locally

Use Node.js 24 and npm:

```sh
git clone https://github.com/ziwon/parallel-lab.git
cd parallel-lab
npm ci
npm run dev
```

Vite prints the local URL. The app uses React and TypeScript; the development server binds to `0.0.0.0` for access from another device.

Check the engine and create a production build:

```sh
npm run check
npx playwright install chromium
npm run test:e2e
```

`npm run check` runs type checking, unit tests, and the build. Tests cover reference arithmetic, tensor ownership, dependencies, buffer lifetimes, replay, and serialization. End-to-end checks exercise strategy switching, comparisons, inspection, export, and mobile layouts.

`npm run build` writes the static application to `dist/`. Use `npm run preview` to serve that build locally.

### Deploy to Cloudflare Pages

The [live application](https://parallel-lab.pages.dev/) is deployed from this repository’s `main` branch. Pages Git integration is configured for automatic production builds and branch previews. To deploy your own copy, connect a separate Pages project to your repository with these settings:

| Setting              | Value             |
| -------------------- | ----------------- |
| Build command        | `npm run build`   |
| Output directory     | `dist`            |
| Production branch    | `main`            |
| Environment variable | `NODE_VERSION=24` |

Review branch previews before merging to `main`. The checked-in `wrangler.jsonc` records the Pages project name and output directory; change the name for your own project. Any static host can serve the build; there are no server routes or runtime secrets. See Cloudflare’s [React deployment guide](https://developers.cloudflare.com/pages/framework-guides/deploy-a-react-site/).

For a manual production deployment, authenticate Wrangler with the project’s Cloudflare account, then run:

```sh
npm run build
npx wrangler@4 pages deploy dist --project-name parallel-lab --branch main
```

Use your own project name when deploying a fork. If a push does not start a build, check the Cloudflare GitHub App’s repository access and Pages deployment settings, or trigger a deployment from the dashboard.

## Understand the model’s limits

This is an educational simulator. It does not run PyTorch or NCCL kernels, benchmark hardware, or predict production scaling.

- Each experiment models **one FP32 Adam step** with two moment states. JavaScript numbers provide representative arithmetic without emulating FP32 rounding.
- Timings come from explicit dependencies, compute budgets, and latency-plus-bandwidth costs. Each rank has one compute lane; the group shares one serialized communication lane. Ranks do not represent physical GPU wiring.
- DDP/FSDP split equal local batches; TP ranks cooperate on the same samples; PP accumulates microbatches. Comparisons preserve the global batch without rounding or dropping samples.
- FSDP is layer-wise and FSDP2-inspired. Retention and prefetch have explicit buffer lifetimes, but this is not a runtime-exact allocation model.
- PP uses **GPipe** with 1, 2, or 4 stages. Other strategies support 1, 2, 4, or 8 ranks. EP uses eight linear experts and a **fixed top-1 router**, with no dropped tokens or router training.
- Strategies run independently. Hybrid parallel meshes, mixed precision, checkpointing, CPU offload, and real profiler import are outside this release.

Read the [model reference](docs/model.md) for the formulas, memory lifetimes, hand-checkable fixtures, and links to the official mechanisms.

## Continue in NCCL Lab

Parallel Training Lab explains **why communication is needed and when training can proceed**. [NCCL Lab](https://nccl-lab.pages.dev/) shows **how a collective moves data among ranks**.

Choose **Explore this collective in NCCL Lab** in the event inspector to carry over its operation, rank count, and logical size. TP links preserve SUM reductions. EP links identify the difference between routed payloads and NCCL Lab’s uniform AllToAll example; pipeline send/receive has no equivalent link. This simulator works independently of the external site.

## Documentation

- [Model reference](docs/model.md) — numerical semantics, memory accounting, communication costs, limitations, and primary references.
- [Scenario guide](docs/scenarios.md) — what to inspect and learn in each comparison.
- [Architecture](docs/architecture.md) — the boundary between deterministic simulation and React playback.
