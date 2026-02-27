# GLM-5 Usage

[GLM-5](https://huggingface.co/zai-org/GLM-5) is a 744B-parameter (40B active) Mixture-of-Experts model from Zhipu AI. It adopts the DeepSeek-V3/V3.2 architecture, including DeepSeek Sparse Attention (DSA/NSA) and Multi-Token Prediction (MTP).

## Launch GLM-5 with SGLang

To serve GLM-5 on AMD GPUs (MI300X/MI325/MI35x) with TP=8:

```bash
python3 -m sglang.launch_server \
  --model zai-org/GLM-5 \
  --tp 8 \
  --trust-remote-code \
  --nsa-prefill-backend tilelang \
  --nsa-decode-backend tilelang \
  --chunked-prefill-size 131072 \
  --mem-fraction-static 0.80 \
  --watchdog-timeout 1200
```

### Configuration Tips

- **Tensor parallelism**: TP=8 is required due to the model size (744B total parameters).
- **Trust remote code**: `--trust-remote-code` is required for the `glm_moe_dsa` architecture.
- **Transformers version**: GLM-5 requires the latest transformers for the `glm_moe_dsa` architecture. Install from source if needed:

```bash
pip install git+https://github.com/huggingface/transformers.git
```

- **Watchdog timeout**: `--watchdog-timeout 1200` (20 minutes) is recommended to allow sufficient time for loading 744B parameters.
- **Chunked prefill**: `--chunked-prefill-size 131072` is the recommended configuration.
- **Memory fraction**: `--mem-fraction-static 0.80` balances KV cache memory with model weight memory.
- **Multithreaded weight loading**: For faster startup, add `--model-loader-extra-config '{"enable_multithread_load": true}'`.

### NSA Attention Backend

GLM-5 uses DeepSeek Sparse Attention (NSA), and the attention backend is automatically set to `nsa`. On AMD GPUs, use `tilelang` for both prefill and decode kernels via `--nsa-prefill-backend tilelang --nsa-decode-backend tilelang`.

For more details on available NSA kernel options, see the [DeepSeek V3.2 documentation](deepseek_v32.md).

## Multi-Token Prediction (MTP)

GLM-5 supports Multi-Token Prediction via [EAGLE speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding), inherited from the DeepSeek V3/V3.2 architecture. This can improve decoding speed at small batch sizes:

```bash
python3 -m sglang.launch_server \
  --model zai-org/GLM-5 \
  --tp 8 \
  --trust-remote-code \
  --nsa-prefill-backend tilelang \
  --nsa-decode-backend tilelang \
  --chunked-prefill-size 131072 \
  --mem-fraction-static 0.80 \
  --watchdog-timeout 1200 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

```{tip}
To enable the experimental overlap scheduler for EAGLE speculative decoding, set the environment variable `SGLANG_ENABLE_SPEC_V2=1`.
```

## Accuracy

GLM-5 achieves strong results on the GSM8K few-shot completion benchmark:

| Platform | TP | Accuracy | Threshold |
| -------- | -- | -------- | --------- |
| AMD MI325 | 8 | 96.5% | 93% |
| AMD MI35x | 8 | 97.0% | 93% |

## Platform-Specific Guides

- For Ascend NPU deployment, see [GLM-5 on Ascend NPU](../platforms/ascend_npu_glm5_examples.md).
