---
name: Parallel Training Lab
description: An inspectable scientific console for distributed training.
colors:
  bg: "#121416"
  panel: "#1b1f23"
  surface: "#1b1f23"
  surface-inset: "#15181b"
  surface-raised: "#22272c"
  field: "#1b1f23"
  text: "#e6e9ed"
  muted: "#a0aab6"
  line: "#363e47"
  lime: "#b2e879"
  cyan: "#70c5d0"
  orange: "#e1ad7d"
  purple: "#b7a0dc"
  yellow: "#d8cf89"
  pink: "#d89dbd"
  scratch: "#94aaa8"
  button: "#22272c"
  button-hover: "#2b3138"
  primary-ink: "#1b2a0d"
  primary-hover: "#c6f59b"
  nav-active: "#243027"
  rank: "#22272c"
typography:
  headline:
    fontFamily: 'Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "30px"
    fontWeight: 630
    lineHeight: 1.3
    letterSpacing: "-1.1px"
  title:
    fontFamily: 'Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "23px"
    fontWeight: 580
    letterSpacing: "-0.02em"
  body:
    fontFamily: 'Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.65
  label:
    fontFamily: 'Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "11px"
    fontWeight: 400
  metric:
    fontFamily: 'Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "23px"
    fontWeight: 450
    lineHeight: 1
    letterSpacing: "-0.8px"
  mono:
    fontFamily: '"SFMono-Regular", Consolas, "Liberation Mono", monospace'
rounded:
  event: "2px"
  cell: "3px"
  tensor: "4px"
  control: "5px"
  rank: "6px"
  panel: "7px"
  dialog: "9px"
spacing:
  compact: "7px"
  small: "8px"
  control-gap: "10px"
  grid-gap: "12px"
  inset: "16px"
  section: "20px"
  page: "30px"
components:
  button-primary:
    backgroundColor: "{colors.lime}"
    textColor: "{colors.primary-ink}"
    rounded: "{rounded.control}"
    padding: "8px 11px"
  button-secondary:
    backgroundColor: "{colors.button}"
    textColor: "{colors.text}"
    rounded: "{rounded.control}"
    padding: "8px 11px"
  field:
    backgroundColor: "{colors.field}"
    textColor: "{colors.text}"
    rounded: "{rounded.control}"
    padding: "9px 10px"
    height: "36px"
  nav-active:
    backgroundColor: "{colors.nav-active}"
    textColor: "{colors.lime}"
    rounded: "{rounded.control}"
    padding: "12px 9px"
  run-tag:
    backgroundColor: "#22272c"
    textColor: "#b6bfc9"
    rounded: "{rounded.cell}"
    padding: "4px 6px"
  rank-card:
    backgroundColor: "{colors.rank}"
    rounded: "{rounded.rank}"
  strategy-selected:
    backgroundColor: "#243027"
    textColor: "{colors.lime}"
    padding: "8px 9px"
  ownership-cell:
    backgroundColor: "#243027"
    textColor: "#cce6b6"
    rounded: "{rounded.cell}"
    padding: "7px 3px"
---

# Design System: Parallel Training Lab

## Overview

**Creative North Star: "The Inspectable Training Console"**

This is a description of the built interface: a dark, compact scientific console in the existing NCCL-compatible visual world. Quiet surfaces make ownership, execution state, and memory changes the prominent content. The system extends one shared workspace across DDP, FSDP, TP, PP, and EP; it does not establish a separate identity for each strategy.

The interface is precise and explanatory. English labels, tabular numbers, small diagrams, and coordinated inspection connect the same training step across several views. Lime marks interaction and readiness, while muted categorical accents describe data and operations. Preserve the distinction between browser-modeled time and measured performance in metric labels and explanatory copy.

**Key Characteristics:**

- Persistent desktop navigation and a dense, inspectable main workspace.
- Shared controls and visual grammar across five training strategies.
- Tonal panels, thin borders, restrained corners, and compact line icons.
- Explicit ownership labels alongside color and partition geometry.
- Coordinated rank, tensor, event, timeline, and memory selection.

Visual authority is the implemented console in `src/styles.css`, its later overrides in `src/parallel.css`, and `src/App.tsx`, `src/components/StrategyOverview.tsx`, and `src/components/SimulationView.tsx`. `PRODUCT.md` supplies the durable English, accessibility, and NCCL compatibility constraints. There is no approved image composition or quality-bar catalog card. The extension follows the existing implementation; the north-star name above summarizes that implementation.

The captured reference set is `.impeccable/review/{tp,pp,ep}-{desktop,mobile}.png`. The final scoped review reported four findings resolved and a ship disposition for the fix batch. These documents record that built state rather than claiming a new independent accessibility or performance audit.

## Colors

The shared UI palette matches NCCL Lab: cool charcoal surfaces, slate text and borders, and lime/cyan accents. Training-specific scientific categories retain their distinct colors. Frontmatter records the actual source values; preview-only tonal ramps in the sidecar are not additional application tokens.

### Primary

- **Signal Lime (`lime`):** primary actions, selected strategies, keyboard focus, successful results, readiness, and collective communication cues. Its dark companion `primary-ink` keeps filled controls readable.

### Secondary

- **Tensor Cyan (`cyan`):** parameters and forward computation. TP ownership cells use a related dark cyan fill and pale cyan text.
- **Gradient Apricot (`orange`):** gradients and backward computation. PP stage ownership uses a related warm brown fill and apricot text.

### Tertiary

- **Optimizer Lavender (`purple`):** Adam state and optimizer operations.
- **Activation Straw (`yellow`):** activation memory; point-to-point transfers use a related yellow-brown treatment.
- **Buffer Rose (`pink`):** temporary memory. EP ownership and all-to-all events use related muted violet surfaces.
- **Scratch Sage (`scratch`):** scratch memory.

### Neutral

- **Charcoal Canvas (`bg`):** page background.
- **Panel Charcoal (`panel` / `surface`) and Raised Charcoal (`surface-raised`):** grouped work areas and nested inspector surfaces.
- **Field Charcoal (`field`):** form controls use the panel surface; the sidebar and topbar use `surface-inset` (`#15181b`).
- **Soft White (`text`):** primary text; **Slate (`muted`)** for explanations, units, and secondary labels.
- **Slate Border (`line`):** dividers and low-emphasis separation. More interactive containers use brighter related strokes from source.

**The Labeled Color Rule.** Keep labels, abbreviated operation names, legends, and ownership text beside color. A hue alone never explains whether a tensor is owned, an operation is running, or memory belongs to a category.

## Typography

**Display and Body Font:** the Inter-first system sans stack in frontmatter. No separate display face is introduced.

**Label/Mono Font:** the system monospace stack for representative arithmetic, tensor identifiers, and precise chart readings. The model-flow input/output endpoints use Georgia as mathematical notation, not as a second heading system.

### Hierarchy

- **Headline:** page title using the headline token; it reduces to 26px below 1000px, 25px below 720px, and 23px below 380px.
- **Title:** the strategy explanation uses the title token, reducing to 22px below 720px. Run titles use 17px and panel headings use 11px.
- **Body:** explanatory paragraphs use the body token. Strategy explanations cap their measure at 60ch.
- **Label:** compact control labels generally use 10–11px. Supporting chart and dense inspector labels range from 8–10px; this is a deliberate density characteristic of the current interface, not a universal minimum for new prose.
- **Metric:** tabular numbers use the metric token. Units are smaller and muted, and metric captions remain separate from their values.
- **Chart:** the memory chart’s final override uses 11px monospace text with a brighter slate fill, preserving legibility on small displays.

Use sentence case for explanations and control labels. Strategy abbreviations, tensor identifiers, and established operation abbreviations keep their technical capitalization.

## Layout

The desktop shell has a fixed left sidebar and a fluid main region capped at 1920px. The sidebar is 214px wide, reduces to 190px below 1250px and 174px below 1000px, and the content offset follows it. Main padding begins at 30px, reduces through 22px and 19px, and reaches 16px horizontally on mobile. The topbar is 58px tall on desktop and 51px on mobile.

The working sequence is page heading and file actions, comparison choices, configuration, strategy ownership explanation, sticky playback, run metrics, model execution and rank inspection, execution timeline, memory chart/table, and numerical result. Comparisons reuse this sequence and its components; side-by-side run columns stack below 1000px.

The comparison cards form a horizontally scrollable row. The visible **All 7 comparisons** native select gives direct access to every preset without requiring horizontal discovery. Cards have a 187px minimum width, changing to 175px on mobile.

The strategy selector presents all five strategies in a single segmented control. Configuration uses a four-column desktop grid with the batch description beneath; below 720px, strategy spans the width, rank and batch fields share a row, and the batch description and Configure action each receive their own row.

The ownership introduction is a two-column explanation/diagram layout with a two-column rank map. Below 720px the explanation stacks above the map. TP uses alternating column/row partition labels; PP shows owned stages and muted absent layers; EP shows eight expert cells, wrapping from eight to four columns below 1250px. The execution flow wraps experts to accommodate their count.

Rank cards occupy two columns in the standard desktop view, next to an event inspector. The inspector is 285px wide by default, 265px below 1250px, and 325px above 1700px. Below 1000px it moves under the cards and the memory table moves beneath its chart. Below 720px rank cards become a single column. Playback wraps while retaining the shared time scrubber.

Below 720px the sidebar becomes a hidden off-canvas drawer with a menu trigger and backdrop. It is made visible only while open. Controls and diagrams remain within the page width, but the execution timeline intentionally retains a 680px minimum canvas inside a focusable horizontal scroller. Mobile users must scroll that region to inspect the whole time span; it is not a fully reflowed timeline.

## Elevation & Depth

Depth comes mainly from adjacent charcoal tones, one-pixel borders, and selected strokes. Rank workspaces add a subtle dotted plotting texture. Most containers remain visually flat. The sticky playback bar has a restrained shadow; dialogs use a stronger ambient shadow and blurred dark backdrop.

### Shadow Vocabulary

- **Sticky playback:** `0 5px 12px #12141640` separates controls from scrolled content.
- **Selected rank:** `0 0 0 1px #b2e87915` reinforces its brighter border.
- **Selected tensor:** `inset 0 0 0 1px #b2e87930` reinforces the selected allocation.
- **Selected timeline event:** `0 0 0 1px #edffd040` separates the focused event from dense adjacent events.
- **Dialog:** `0 20px 120px #000b`, paired with a `#080e10bd` backdrop and 4px blur.

Selection and focus are explicit. Keyboard focus uses a two-pixel lime outline with a three-pixel offset. Rank border changes take 120ms; the mobile drawer transform takes 200ms. Reduced-motion preferences remove transitions and animations and use automatic scrolling.

## Shapes

Use compact rounded rectangles with one-pixel strokes. Small timeline events are nearly square; ownership cells and run tags are slightly rounded; controls and tensor blocks are modestly softened. Rank cards, panels, and dialogs step up through the radius tokens. The strategy control has a rounded outer shell and square interior segments, with a two-pixel lime inset underline for selection.

Status dots are small circles. Icons are thin, compact Lucide line symbols, generally 12–19px in working controls. Avoid adding large decorative illustrations to this inspection surface: the diagrams and numerical traces are its visual material.

## Components

### Buttons

Primary buttons use lime fill, dark text, and semibold emphasis. Secondary buttons use charcoal fill and a thin border; hover lightens the fill and stroke. File and utility actions use transparent or icon-only treatments. Disabled controls reduce opacity to 0.35. Focus always retains the shared visible outline.

The Play button is a dedicated primary variant with a filled play icon, a 78px minimum width, and 9px by 14px padding. It shares a compact row with reset, previous/next event, and playback speed.

### Inputs / Fields

Native inputs and selects use dark fills, visible slate borders, rounded control corners, and a 36px default height. Labels sit above fields or next to compact choosers. Number fields use tabular numerals; checkbox and range accents use lime. Preserve visible labels for the strategy, ranks, batch, comparison picker, and event picker.

### Navigation

The desktop sidebar keeps Training simulator, Compare runs, Model & assumptions, and the five strategies available together. Active navigation uses a muted green fill and lime text; the selected strategy also receives a status dot. Each strategy entry carries a plain-language subtitle. Mobile uses the same navigation in a drawer.

### Chips / Tags

Run tags are compact outlined labels with uppercase text. Phase tags combine a dot with a written state; completed states turn lime. The workload indicator is a small lime text label, not a filled badge. Keep these low-emphasis markers subordinate to run titles and primary controls.

### Cards / Containers

Workspace panels use the panel tone, a fine stroke, and clipped rounded corners. Rank cards use a slightly lighter fill. Their headings select ranks; nested tensor controls select owned allocations; metadata, shard stripes, and categorical memory bars expose the same rank’s state. Selection brightens the border and unrelated content dims rather than disappearing.

### Strategy Ownership Overview

The explanation and partition map establish what is split before playback begins. DDP shows full replicas; FSDP shows state fractions; TP shows alternating column and row partitions; PP shows stage ownership and microbatch count; EP shows expert ownership and routed-token counts. Empty ownership cells retain muted labels and borders, making absence legible. Keep workload and sample descriptions strategy-specific.

### Shared Playback and Inspector

One sticky playback bar coordinates all views. Its scrubber displays event boundary and modeled milliseconds. Space toggles playback, arrow keys step events, and R resets. Comparisons add a focused-run selector. The event inspector provides an event picker, operation state, prerequisites, jump-to-start/end controls, allocation changes, and representative arithmetic. Selecting a timeline event, tensor, or rank updates the inspection context.

### Execution Timeline

One compute lane per rank and one shared communication lane expose ordering. Events use tinted fills, short operation labels, hover brightening, and stronger selected/executing borders; unrelated events dim. The visible cursor links playback to the schedule. Keep the horizontal scroller keyboard-focusable and preserve accessible labels with full operation names and times when short bars clip their text.

### Memory Inspector

A stepped stacked area chart and category table show the selected rank’s current and peak allocation. Parameter, gradient, optimizer, activation, temporary, and scratch colors match the other views. Current and peak values are separate columns; category peaks are not presented as a sum equal to total peak. A capacity line appears when relevant. The chart reduces horizontal tick count at narrow widths and keeps the category table below it.

## Do's and Don'ts

### Do:

- **Do** extend the existing charcoal, lime, and categorical-color system across all five strategies.
- **Do** preserve explicit ownership, workload, batch, rank, and operation labels beside diagrams.
- **Do** keep the comparison select available alongside the scrollable preset cards.
- **Do** coordinate focus across rank cards, tensor controls, the timeline, and the event and memory inspectors.
- **Do** retain keyboard focus, playback shortcuts, reduced-motion behavior, and the mobile drawer.
- **Do** label timing as modeled and keep browser-only educational limitations visible where results are interpreted.

### Don't:

- **Don't** assign a separate visual identity or shell to TP, PP, or EP.
- **Don't** rely on color alone to communicate ownership, operation state, or memory category.
- **Don't** imply the timeline measures GPU throughput or physical network execution.
- **Don't** compress the wide execution timeline into unreadable mobile bars; preserve its contained horizontal scrolling.
- **Don't** turn sidecar preview tonal ramps into new application colors without an implementation decision.
