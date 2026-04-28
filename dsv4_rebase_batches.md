# DSv4 Rebase Batch Anchors

merge-base: `0519b09` (2026-01-27)
latest main: `3066ba8` (2026-04-28)
total main commits: **2803**, batches: **29** (~100/batch)

| # | cumulative | SHA | date | subject |
|---|---|---|---|---|
| 1 | 100 | `81449b4bee` | 2026-01-31 | Optimize GDN decode for Qwen3 Next (#17094) |
| 2 | 200 | `25508d11c0` | 2026-02-03 | [Docker] Remove hardcoded `America/Los_Angeles` timezone, default to UTC (#18121... |
| 3 | 300 | `75997ebe8d` | 2026-02-08 | Update author information in pyproject.toml (#18453) |
| 4 | 400 | `f4417475b8` | 2026-02-12 | Build ROCm7.2 Image with latest AITER v0.1.10.post3 (#18741) |
| 5 | 500 | `5ddc84e33e` | 2026-02-16 | [AMD] MORI-EP inter kernel type switch (#18437) |
| 6 | 600 | `38a69652e6` | 2026-02-20 | [diffusion] logging: log available mem when each stage starts in debug level (#1... |
| 7 | 700 | `9bce3b040c` | 2026-02-24 | [diffusion] [NPU] Update perf baselines (#19227) |
| 8 | 800 | `bdc1e46e5a` | 2026-02-26 | [Qwen3.5] Qwen3.5-27B inference repeat bug fix (#19411) |
| 9 | 900 | `ea6ff7b01f` | 2026-03-01 | Support multi sharding group on the same dimension in dump comparator (#19601) |
| 10 | 1000 | `1135e214b3` | 2026-03-03 | [CI] support `/rerun-ut` command in slash handler (#19800) |
| 11 | 1100 | `04e364d538` | 2026-03-05 | [V32] Enhance deepseek v32 related tests (#19985) |
| 12 | 1200 | `3e8abc71ca` | 2026-03-10 | [Disagg] Skip health check enqueue when PD disagg queues have backlog (#20191) |
| 13 | 1300 | `b227e53ebf` | 2026-03-12 | feat: add banner to sgl-model-gateway (#20471) |
| 14 | 1400 | `71a54c1c42` | 2026-03-16 | update CODEOWNERS (#20733) |
| 15 | 1500 | `574572b21b` | 2026-03-19 | [BugFix] bug fix for DeepSeek eagle3 in Attn-DP mode (#20492) |
| 16 | 1600 | `2406ddfdb8` | 2026-03-22 | Add ut guide to test skills (#21130) |
| 17 | 1700 | `80389fec00` | 2026-03-25 | [AMD] Fix AMD CI: mark /sglang-checkout as git safe.directory in container (#214... |
| 18 | 1800 | `18074e25dc` | 2026-03-29 | fix: scheduler launch hang when non-current rank dies (#20287) |
| 19 | 1900 | `1b45d81e91` | 2026-03-31 | fix: only showing recent runners from ci failure analysis (#21015) |
| 20 | 2000 | `658a2813d8` | 2026-04-03 | [NPU] Update CI Dependency (#21578) |
| 21 | 2100 | `d72f58d1c1` | 2026-04-07 | [Qwen3-Specv2]: Fix flaky ci (#22194) |
| 22 | 2200 | `a64905a7b8` | 2026-04-09 | [CICD] [prefill-only] Consolidate prefill-only model E2E tests (#22405) |
| 23 | 2300 | `870a21bf39` | 2026-04-11 | [CI] Remove Slack bot from CI failure monitor (#21581) |
| 24 | 2400 | `36891ab514` | 2026-04-14 | Rename _alive_streaming_session_count; use _is_streaming helper (#22755) |
| 25 | 2500 | `6e3bbef568` | 2026-04-17 | expose num_embeddings in VocabParallelEmbeddingWithLoRA (#22547) |
| 26 | 2600 | `48daa831ea` | 2026-04-21 | [KDA] Fuse gate+cumsum and reuse chunk index for KDA (#23038) |
| 27 | 2700 | `bf98eb3ab7` | 2026-04-24 | [Intel GPU] Enable pipeline parallelism on XPU (#23472) |
| 28 | 2800 | `9ffc0cc67e` | 2026-04-28 | [NPU] Support GLM-4.5V (#22961) |
| 29 | 2803 | `3066ba8167` | 2026-04-28 | fix(hicache): add retry logic for MooncakeStore warmup (#17195) |

## Workflow

Per batch:
1. `git merge <SHA>`
2. resolve conflicts, commit
3. tick the batch below

## Progress

- [x] batch 1: `81449b4bee` (2026-01-31, +100 commits) — merge commit `6bf5a26509`
- [x] batch 2: `25508d11c0` (2026-02-03, +200 commits) — merge commit `a05bef1a46`
- [x] batch 3: `75997ebe8d` (2026-02-08, +300 commits) — merge commit `9cd53fdef8`
- [x] batch 4: `f4417475b8` (2026-02-12, +400 commits) — merge commit `bd8ff150e8`
- [ ] batch 5: `5ddc84e33e` (2026-02-16, +500 commits)
- [ ] batch 6: `38a69652e6` (2026-02-20, +600 commits)
- [ ] batch 7: `9bce3b040c` (2026-02-24, +700 commits)
- [ ] batch 8: `bdc1e46e5a` (2026-02-26, +800 commits)
- [ ] batch 9: `ea6ff7b01f` (2026-03-01, +900 commits)
- [ ] batch 10: `1135e214b3` (2026-03-03, +1000 commits)
- [ ] batch 11: `04e364d538` (2026-03-05, +1100 commits)
- [ ] batch 12: `3e8abc71ca` (2026-03-10, +1200 commits)
- [ ] batch 13: `b227e53ebf` (2026-03-12, +1300 commits)
- [ ] batch 14: `71a54c1c42` (2026-03-16, +1400 commits)
- [ ] batch 15: `574572b21b` (2026-03-19, +1500 commits)
- [ ] batch 16: `2406ddfdb8` (2026-03-22, +1600 commits)
- [ ] batch 17: `80389fec00` (2026-03-25, +1700 commits)
- [ ] batch 18: `18074e25dc` (2026-03-29, +1800 commits)
- [ ] batch 19: `1b45d81e91` (2026-03-31, +1900 commits)
- [ ] batch 20: `658a2813d8` (2026-04-03, +2000 commits)
- [ ] batch 21: `d72f58d1c1` (2026-04-07, +2100 commits)
- [ ] batch 22: `a64905a7b8` (2026-04-09, +2200 commits)
- [ ] batch 23: `870a21bf39` (2026-04-11, +2300 commits)
- [ ] batch 24: `36891ab514` (2026-04-14, +2400 commits)
- [ ] batch 25: `6e3bbef568` (2026-04-17, +2500 commits)
- [ ] batch 26: `48daa831ea` (2026-04-21, +2600 commits)
- [ ] batch 27: `bf98eb3ab7` (2026-04-24, +2700 commits)
- [ ] batch 28: `9ffc0cc67e` (2026-04-28, +2800 commits)
- [ ] batch 29: `3066ba8167` (2026-04-28, +2803 commits)
