# Journal: Phase 07 — Optimization & Polish

**Date:** 2026-03-30
**Branch:** feat/optimization-polish
**Commit:** 5263446

## What Happened

Final phase of the real-time video analytics pipeline. Took the working prototype (phases 1-6) and made it production-ready: CI/CD, Docker, documentation, CLI hardening, security fixes.

## Key Changes

- **Pipeline bug fix:** `captureLoop()` was crashing on video EOF — treated file end as camera disconnect. Now handles gracefully, lets inference drain the queue.
- **Benchmark fix:** Bounded queue (size 8) was dropping 95%+ of frames during benchmark. Added large queue mode for benchmark, capped at 2000 to prevent OOM.
- **CLI refactor:** Replaced hardcoded `deviceIndex=2` with proper CLI args (`--device`, `--model`, `--port`, `--bind`, `--workers`). Added `parseIntArg` with try/catch to prevent `std::stoi` crashes. Unrecognized args now warn instead of silently ignoring.
- **Security:** SSE server now binds to `127.0.0.1` by default (was `0.0.0.0`). Added `--bind` flag for explicit network exposure.
- **CI:** GitHub Actions with two jobs — release build + all tests, and a separate TSAN build for race detection.
- **Docker:** Multi-stage build (builder + runtime), non-root user, Ubuntu 22.04. Fixed OpenCV package names (4.5d on 22.04, not 406t64).
- **FP16:** Quantization script works (50% model size reduction). Runtime needs `onnxruntime-gpu` — CPU provider doesn't support FP16 ops. Deferred to when CUDA-enabled ORT is installed.

## Decisions

| Decision | Rationale |
|----------|-----------|
| Skip V4L2 zero-copy | Capture latency <1ms vs inference 35ms — not the bottleneck |
| Skip clang-tidy in CI | Nice-to-have, zero warnings already |
| Defer FP16 runtime | Need onnxruntime-gpu package, current install is CPU-only |
| Cap benchmark queue at 2000 | Prevents OOM on large --frames values |
| Default bind 127.0.0.1 | Security: don't expose video feed to network by default |

## Metrics

- **Tests:** 245/245 pass (43 pipeline + 87 serializer + 103 latency + 12 inference)
- **Warnings:** 0
- **Benchmark:** ~25-33 FPS on CPU, ~35ms/frame inference (YOLOv8n 640x640)
- **Files:** 11 changed/created, 755 insertions, 26 deletions

## Project Status

All 7 phases complete. Pipeline goes from USB camera to browser dashboard with object detection, benchmarking, and production deployment support.
