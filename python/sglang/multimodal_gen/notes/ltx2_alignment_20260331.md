## 2026-03-31 862f Reset

- trusted baseline: Hugging Face `Lightricks/LTX-2` model card two-stage sunset example
- 当前用户肉眼反馈：sglang 结果严重跑偏，内容像“女人说话”，且画质很差
- 本轮严格从 `862f60587e267ee0bb4590658f8f2d7d46009a15` 重新开始，不沿用后续实验分支实现
- 先锁定 LoRA 主线：
  - `862f` 里 `set_lora()` 默认会在 `set_lora_weights()` 内立即 merge
  - 但官方 stage2 示例走的是 `load_lora_weights + set_adapters`，默认不是 merge-base path
  - 对 LTX2 two-stage 官方示例路径，stage2 distilled LoRA 现在改成 `merge_weights=False`
  - 同时补了 `deactivate_lora_weights()`，保证从 stage2 回 stage1 时，未 merge 的 adapter 也会被彻底停用
- 当前精度对齐百分比：20%
  - 含义：native two-stage 结构已具备，且已经开始针对最可能主根因（LoRA merge 语义）做 source-level 修复；还没完成新一轮端到端验证
- 下一步：
  - 远端 H100 在 `862f` 基线上复跑 official sunset 对拍
  - 先看视频内容是否从“女人说话”收回到日落语义
  - 若仍偏，再继续查 seed/config 与 stage1 schedule / resolution 语义

## 2026-03-31 CLI Config Sampling Path

- 新定位到一个更早的主根因：`sglang generate` 实际入口走的是 `FlexibleArgumentParser.parse_known_args()`，但之前只有 `parse_args()` 支持 `--config` 展开
- 这会导致 config 里的 sampling 参数在真实 CLI 路径被静默忽略，典型症状和本次现场一致：
  - `seed` 退回 LTX2 默认值 `10`
  - `negative_prompt` 退回 LTX2 默认长负面词（长度 1078）
  - prompt/negative_prompt/seed 与官方 example YAML 不一致，直接污染 text encoder / connector 输入
- 已修复：
  - `FlexibleArgumentParser.parse_known_args()` 现在和 `parse_args()` 一样会先展开 config
  - 新增 `test_cli_parser_unit.py`，覆盖 `generate --config ...` 对 sampling 参数的注入
- 本地验证：
  - 用隔离脚本直接验证 `parse_known_args(['generate', '--config', cfg])` 后，`prompt='A beautiful sunset over the ocean'`、`negative_prompt='shaky, glitchy, low quality'`、`seed=1234` 全部正确进入 namespace
- 当前精度对齐百分比：45%
  - 含义：已经定位并修掉一个会直接改变文本条件输入的真实主根因；还需要远端重新跑真实 two-stage 路径，确认视频内容和 prompt semantics 收敛

## 2026-03-31 Flat `sglang generate` Validation

- 按用户要求，改用平铺参数的 `sglang generate`，完全绕开 `--config` 路径做一次真实 two-stage 复跑
- 结论：
  - flat 命令里的参数被真实进程正确接收，CLI 没再退回默认 `seed` / `negative_prompt`
  - `TextEncodingStage` 和 `LTX2TextConnectorStage` 都已跑通，说明 prompt/text 这条链已经能在真实 generate 路径里执行
  - 端到端仍然没出片，新的主阻塞点是 `LTX2LoRASwitchStage` 在 stage1 -> stage2 切换时 OOM
- 远端 run 观察：
  - `text_encoder` customized loader 因 `Gemma3TextConfig.prefix` 缺失而 fallback 到 native `Gemma3Model`
  - LoRA 切换时 `deactivate_lora_weights()` 触发 `disable_offload() -> load_all_layers()`，在 H100 80G 上额外申请约 738 MiB 时 OOM
  - 当前 flat run 没有生成 `/tmp/ltx2_sunset/sglang_flat/sglang_sunset_flat.mp4`
- 当前精度对齐百分比：50%
  - 含义：flat generate 已经证明 config 不是当前主阻塞项，真实链路的最新主要问题已经收敛到 text encoder fallback + stage switch OOM
