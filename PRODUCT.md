# Parallel Training Lab

<!-- impeccable:product-schema 1 -->

## Platform

web

## Product Purpose

Teach distributed training through one inspectable, deterministic training step. Explain tensor ownership, computation, communication prerequisites, and temporary memory peaks. NCCL Lab explains how a collective moves data; this lab explains why training requires it.

## Users

People learning distributed training who need to follow a training step without owning a GPU. This audience and learning goals come from the project brief.

## Capabilities and Constraints

DDP, FSDP, TP, PP, and EP are executable educational strategies. The browser runs small seeded arithmetic while virtual storage drives the memory ledger. No backend, model downloads, credentials, GPU runtime, or external service is required. Use English copy, deterministic replay, shared configurations, JSON export/import, and fixed-problem comparisons. Dense and MoE workloads are distinct. Timings are modeled, never measured throughput.

## Brand Commitments

Display title: Parallel Training Lab. Preserve compatible interaction conventions with NCCL Lab. Extend the existing scientific console with coordinated ownership, execution, memory, and event inspection.

## Accessibility & Inclusion

Keyboard playback and visible focus, reduced motion, readable tensor labels, color-independent labels, desktop and narrow mobile layouts.

## Evidence on Hand

Source and tests in src/, tests/, e2e/. Official mechanisms and deliberate simplifications are cited in docs/model.md. No production performance claims.
