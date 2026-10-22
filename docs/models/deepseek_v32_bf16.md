# DeepSeek V3.2 with BF16 DSA

The `v0.30.0-dsa-bf16` branch adds an unquantized BF16 lightning indexer to
DeepSeek V3.2. Enable it with `indexer_kv_dtype="bf16"`. The default `"auto"`
continues to use the existing FP8 indexer.

The BF16 path keeps the indexer's rotated queries, keys, and paged key cache
in BF16. Dot products accumulate in FP32; the rectified, weighted scores and
top-k selection also use FP32. It uses Triton kernels instead of DeepGEMM
for index scoring. The existing sparse MLA attention consumes the selected
indices, preserving DSA rather than replacing it with dense attention.

## Start the server

Use a BF16 DeepSeek V3.2 checkpoint. Set `MODEL_PATH` to its directory and
`TP_SIZE` to the tensor parallel size for your deployment, then run:

```bash
vllm serve "$MODEL_PATH" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --attention-config '{"backend":"FLASHMLA_SPARSE","indexer_kv_dtype":"bf16"}'
```

All three precision settings matter: `--dtype` controls model computation,
`--kv-cache-dtype` controls the main MLA cache, and `indexer_kv_dtype` controls
the separate DSA indexer. An FP8 checkpoint is still detected as quantized
when `--dtype bfloat16` is supplied. To run entirely in BF16, its weights must
already have been correctly dequantized using their scales, and its config
must describe unquantized weights. Deleting `quantization_config` from an
FP8 checkpoint without converting its tensors is not a valid conversion.

## Supported configuration

- CUDA with BF16 model activations. The example uses the existing FlashMLA
  sparse backend on Hopper or Blackwell datacenter GPUs (SM90/SM100).
- Paged indexer caching, causal top-k, chunked prefill, prefix caching,
  mixed prefill/decode batches, and multiple query tokens per request.
- Tensor and pipeline parallelism retain the model's existing implementation.
  Prefill/decode context parallelism (PCP/DCP) is rejected for the BF16 indexer.
- CUDA graph padding is masked, and full graph captures reserve score columns
  up to `max_model_len`. Piecewise graphs keep the indexer as an attention
  splitting operation so each batch receives fresh metadata.

The BF16 indexer cache uses two bytes per key element, with no quantization
scales. For the usual 128-element key this is 256 bytes per token per layer,
compared with 132 bytes in the existing FP8 indexer cache. Score buffers are
chunked using `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`; top-k adds temporary storage.
This path prioritizes unquantized computation; performance is not guaranteed
to match the optimized FP8 implementation.

## Validation

After installing vLLM and test dependencies in the project virtual environment:

```bash
.venv/bin/python -m pytest tests/kernels/attention/test_bf16_indexer.py -v
.venv/bin/python -m pytest tests/v1/attention/test_bf16_indexer.py -v
.venv/bin/python -m pytest tests/model_executor/layers/test_mla_short_prefill_indexer.py -v
```

The kernel tests compare scores against an FP32 reference computed from BF16
inputs, and check causal selection, noncontiguous pages, cache insertion,
negative slots, graph padding, query chunking, and graph replay. They require
a CUDA GPU. Model-level validation must additionally compare generated
tokens/logprobs with a BF16 reference, exercise prompts longer than
`index_topk`, and run the existing `tests/evals/gsm8k` evaluation with the
converted checkpoint and the flags above.

GPU numerical tests and full-model evaluations have not been run in the
development environment for this branch; it has no CUDA device or model
checkpoint. Offline kernel compilation and static checks do not establish
model accuracy or serving performance.

Local checks used Python 3.12 and PyTorch 2.13.0 (CPU): the metadata suite
passed 5 tests; the model regression suite passed 12 tests and skipped 5
GPU cases. These suites were run with `--confcutdir` set to their respective
test directories, without the repository-wide integration fixtures. The 4
new CUDA kernel tests were skipped. Triton 3.8.0 compiled both new kernels
offline for SM80, SM90, and SM100 over four head-count/head-dimension pairs
(24 compilations); this checks generated code, not GPU execution.
