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
