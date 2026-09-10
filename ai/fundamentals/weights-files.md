---
title: Inside a Downloaded Model Folder
description: Every file in a real Hugging Face repo, using DeepSeek-V4-Flash as the example.
lang: en
ref: weights-files
nav_order: 2
---

# Inside a Downloaded Model Folder

A downloaded model is a **package**, not a single file. The weights are only one
part of it. The rest describes how to build the model class, how to turn text
into token IDs, and how to format a prompt.

This page walks through a real repository,
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731),
and then covers the files you will meet in other repositories.

## The Example Repository

```text
DeepSeek-V4-Flash-0731/                        # 167 GB total
├── .gitattributes                             #   1.52 kB
├── LICENSE                                    #   1.08 kB  (MIT)
├── README.md                                  #   7.24 kB  (model card)
├── config.json                                #   1.89 kB
├── generation_config.json                     #    170  B
├── model-00001-of-00048.safetensors           #   1.06 GB
├── model-00002-of-00048.safetensors           #   3.57 GB
├── ...                                        #   ~3.6 GB each
├── model-00048-of-00048.safetensors
├── model.safetensors.index.json               #   5.60 MB
├── tokenizer.json                             #   6.37 MB
├── tokenizer_config.json
├── encoding/                                  # prompt encoding (no Jinja template)
│   ├── README.md
│   ├── encoding_dsv4.py
│   ├── test_encoding_dsv4.py
│   └── tests/test_input_*.json
└── inference/                                 # reference local-inference code
    ├── README.md
    ├── config.json
    ├── convert.py
    ├── generate.py
    └── kernel.py
```

Two things are already visible:

- The 48 weight shards are more than 99% of the download. Everything that makes
	those tensors usable fits in a few megabytes.
- There is **no** `chat_template.jinja` and no `.bin` file. Both are deliberate
	choices by the publisher, explained below.

## Quick Reference

| File | Category | Required to run? | In this repo? | What it does |
| --- | --- | --- | :---: | --- |
| `*.safetensors` | Weights | Yes | Yes | The actual tensors (parameters) |
| `model.safetensors.index.json` | Weights | Yes, if sharded | Yes | Maps each tensor name to the shard that contains it |
| `pytorch_model*.bin` | Weights | Yes, if no SafeTensors | No | Legacy pickle-based weight files |
| `*.gguf` | Weights + metadata | Yes | No | Self-contained quantized model for GGML runtimes |
| `config.json` | Architecture | Yes | Yes | Model class, layer sizes, and quantization settings |
| `generation_config.json` | Inference | No | Yes | Default sampling and stopping settings |
| `tokenizer.json` | Tokenizer | Yes | Yes | Fast tokenizer: vocabulary plus merge and normalization rules |
| `tokenizer_config.json` | Tokenizer | Yes | Yes | Tokenizer class, special tokens, and context length |
| `tokenizer.model` / `vocab.json` + `merges.txt` | Tokenizer | Sometimes | No | Slow or legacy tokenizer sources |
| `special_tokens_map.json` | Tokenizer | No | No | Names the BOS, EOS, PAD, and UNK tokens |
| `chat_template.jinja` | Prompting | No, but important for chat | No | Formats messages into the model's expected prompt string |
| `preprocessor_config.json` | Multimodal | Yes, for vision/audio | No | Image or audio preprocessing parameters |
| `adapter_model.safetensors` + `adapter_config.json` | Fine-tune | Yes, for LoRA | No | A small delta applied on top of a base model |
| `README.md`, `LICENSE`, `.gitattributes` | Metadata | No | Yes | Documentation, terms of use, and Git LFS rules |

## Weight Files

### `model-000NN-of-00048.safetensors`

The parameters themselves. A large model is **sharded** so that each file stays
at a practical size and can be downloaded, resumed, cached, and memory-mapped
independently. Here shard 1 is 1.06 GB (embeddings and the first dense layers)
and the remaining 47 are about 3.6 GB each.

The name `model-00002-of-00048.safetensors` means "shard 2 of 48". Shards are
not layer groups you can pick from — you need all 48.

Internally the file is a JSON header plus a raw tensor buffer, so a loader can
read the header and then map only the tensors it needs, without executing any
code from the file.

> **Why only 167 GB for a ~300B-parameter model?** The `quantization_config`
> block in `config.json` says the checkpoint is stored in FP8 (`e4m3`) with
> 128x128 weight blocks, and `expert_dtype` is `fp4`, so the MoE experts are
> stored at 4 bits. The average is roughly 4-5 bits per parameter rather than
> the 16 bits a BF16 checkpoint would use.

### `model.safetensors.index.json`

The shard map, 5.6 MB here because a 43-layer MoE model with 256 experts per
layer has an enormous number of individually named tensors. It has two parts:

- `metadata.total_size` — total bytes of all weights, useful for a memory check.
- `weight_map` — one entry per tensor, such as
	`"model.layers.20.mlp.experts.7.down_proj.weight": "model-00023-of-00048.safetensors"`.

Without this index a loader would have to open all 48 shards to find a tensor.
If it is missing or stale, loading fails even when every shard is present.

### Files this repo does *not* have

- **`pytorch_model*.bin`** — the older PyTorch pickle format, with
	`pytorch_model.bin.index.json` as its index. Many repositories still ship both;
	when they do you only need the SafeTensors set. Pickle files can execute
	arbitrary code on load, so only load `.bin` weights from sources you trust.
- **`*.gguf`** — the single self-describing file used by `llama.cpp`, Ollama,
	and LM Studio. GGUF embeds the architecture metadata, tokenizer, and chat
	template inside the weight file, which is why a GGUF download needs no
	`config.json` or `tokenizer.json`. The suffix encodes the quantization, such as
	`Q4_K_M`. GGUF builds are produced by converting a checkpoint like this one,
	usually by the community rather than the original publisher.

## `config.json`

Tells the framework **what to build** before any weight is loaded. Real values
from this repository:

| Field | Value | Meaning |
| --- | --- | --- |
| `architectures` | `["DeepseekV4ForCausalLM"]` | The model class to instantiate |
| `model_type` | `deepseek_v4` | Short id used for auto-class lookup |
| `transformers_version` | `4.57.1` | The version the config was written for |
| `hidden_size` | `4096` | Residual stream width |
| `num_hidden_layers` | `43` | Depth |
| `num_attention_heads` / `num_key_value_heads` | `64` / `1` | Unequal values mean grouped-query attention; one KV head is multi-query, which shrinks the KV cache |
| `vocab_size` | `129280` | Embedding rows; must match the tokenizer |
| `max_position_embeddings` | `1048576` | 1M-token context |
| `rope_theta`, `rope_scaling` | `10000`, YARN `factor: 16` over `original_max_position_embeddings: 65536` | Trained at 64K, extended to 1M by position-embedding scaling |
| `torch_dtype` | `bfloat16` | The dtype tensors are dequantized into at runtime |
| `quantization_config` | `quant_method: fp8`, `fmt: e4m3`, `weight_block_size: [128, 128]` | How to read the stored low-bit weights |
| `tie_word_embeddings` | `false` | The output head has its own matrix |

MoE-specific fields describe the routing:

| Field | Value | Meaning |
| --- | --- | --- |
| `n_routed_experts` | `256` | Experts per MoE layer |
| `n_shared_experts` | `1` | Expert always applied, regardless of routing |
| `num_experts_per_tok` | `6` | Experts activated per token — why only a fraction of the parameters run per token |
| `moe_intermediate_size` | `2048` | Width of each expert |
| `expert_dtype` | `fp4` | Experts are stored at 4 bits |

A few fields only make sense for this architecture:
`num_nextn_predict_layers`, `dspark_target_layer_ids`, and `dspark_block_size`
belong to the **DSpark speculative-decoding module** that ships inside the same
checkpoint. That is why the vLLM and SGLang commands in the model card enable
speculative decoding without pointing at a separate draft model.

> If `config.json` and the weights disagree, loading fails with shape mismatch
> errors. Edit it only when you know exactly what you are changing — the model
> card here explicitly tells you to remove `expert_dtype` if you want FP8
> experts instead of FP4.

## `generation_config.json`

Default **decoding** settings, separate from the architecture. The whole file is
170 bytes:

```json
{
  "bos_token_id": 0,
  "eos_token_id": 1,
  "do_sample": true,
  "temperature": 1.0,
  "top_p": 1.0
}
```

Other repositories add `top_k`, `max_new_tokens`, or `repetition_penalty` here.
`eos_token_id` matters most in practice: a wrong value is a common cause of a
model that never stops generating. Note that these defaults are not always the
publisher's recommendation — the model card asks for `top_p = 0.95` in agentic
scenarios. Your runtime's own settings override this file.

## Tokenizer Files

The tokenizer converts text to token IDs, and those IDs index directly into the
embedding matrix. A tokenizer from a different model produces meaningless output
even when the tensor shapes happen to match.

### `tokenizer.json`

The complete **fast tokenizer** serialized from the Rust `tokenizers` library:
normalizer, pre-tokenizer, the full 129,280-entry vocabulary, merge rules,
post-processor, and decoder. At 6.37 MB it is the second-largest file in the
repository. When it is present this single file is enough, which is why there is
no `tokenizer.model`, `vocab.json`, or `merges.txt` here.

### `tokenizer_config.json`

How to construct the tokenizer:

| Field | Value |
| --- | --- |
| `tokenizer_class` | `PreTrainedTokenizerFast` |
| `model_max_length` | `1048576` |
| `add_bos_token` / `add_eos_token` | `false` / `false` |
| `bos_token` | `<｜begin▁of▁sentence｜>` |
| `eos_token` | `<｜end▁of▁sentence｜>` |
| `pad_token` | `<｜end▁of▁sentence｜>`, reused as in most causal LMs |
| `unk_token` | `null`; byte-level BPE has no unknown token |

`add_bos_token: false` is worth noting: the caller is responsible for the BOS
token, because the prompt format is built outside the tokenizer.

### Tokenizer files you will see elsewhere

- `tokenizer.model` — SentencePiece binary (Llama 2, Mistral, Gemma).
- `vocab.json` + `merges.txt` — BPE vocabulary and merge ranks (GPT-2 style).
- `vocab.txt` — WordPiece vocabulary (BERT style).
- `special_tokens_map.json` — maps `bos_token`, `eos_token`, `pad_token`, and
	`unk_token` to concrete tokens. This repo folds that information into
	`tokenizer_config.json` instead.

## Prompt Formatting: The `encoding/` Folder

Most instruct models ship a `chat_template.jinja`, or a `chat_template` string
inside `tokenizer_config.json`, that turns a message list into the exact prompt
the model was tuned on. **This release ships neither.** The model card says so
directly and provides `encoding/encoding_dsv4.py` instead:

```python
from encoding_dsv4 import encode_messages, parse_message_from_completion_text

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]
prompt = encode_messages(messages, thinking_mode="thinking")
# "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜><think>"

tokens = tokenizer.encode(prompt)
```

The reason is that a Jinja template cannot express what this model needs: two
thinking modes, dropping earlier turns' reasoning, three `reasoning_effort`
levels injected as a prompt prefix, a DSML tool-calling block, and auxiliary
task tokens such as `<｜title｜>` and `<｜query｜>`. The folder also provides
`parse_message_from_completion_text`, the inverse function that splits raw
output back into `reasoning_content`, `content`, and `tool_calls`.

Whichever form it takes, this is the file people most often skip — and skipping
it is the most common reason an instruct model gives weak or rambling answers.
The model is being asked in a format it never saw during fine-tuning.

## The `inference/` Folder

Reference code, not part of the model:

- `convert.py` — converts the Hugging Face shards into the layout this demo
	expects, given an expert count and a model-parallel degree.
- `generate.py` — an interactive and batch chat driver launched with `torchrun`.
- `kernel.py` — the custom quantized kernels.
- `config.json` — a *separate* config for this demo, not the Transformers one at
	the repository root.

Publishers include folders like this when a model needs kernels or a launch path
that upstream frameworks do not yet have. You can ignore it entirely when
serving with vLLM or SGLang.

## Files You See in Other Model Types

### Multimodal models

- `preprocessor_config.json` — image resize, crop, rescale, and normalization
	values, or audio sampling parameters.
- `processor_config.json` — how the tokenizer and the feature extractor are
	combined.

### LoRA and other adapters

- `adapter_model.safetensors` — only the low-rank delta weights, typically
	megabytes rather than gigabytes.
- `adapter_config.json` — `base_model_name_or_path`, rank `r`, `lora_alpha`, and
	the target module names.

An adapter is useless on its own; it must be applied to the exact base model it
names.

### Other quantized checkpoints

GPTQ and AWQ builds use a separate `quantize_config.json` with `bits`,
`group_size`, `sym`, and `desc_act`, rather than the `quantization_config` block
inside `config.json` used here. Mismatched quantization metadata produces
garbage output rather than a clean error, so keep it with its weights.

### Custom-code models

`modeling_*.py` and `configuration_*.py` carry the architecture implementation
for repositories whose architecture is not yet merged into Transformers. Loading
them requires `trust_remote_code=True`, which executes the publisher's Python.
This repo has no such files at the root, because `deepseek_v4` is supported by
Transformers 4.57.1 directly.

### Training and fine-tuning leftovers

`optimizer.pt`, `scheduler.pt`, `trainer_state.json`, `training_args.bin`, and
`rng_state.pth` belong to a training checkpoint. They let training resume, and
`optimizer.pt` alone can be twice the size of the weights. Delete them for
inference-only deployments.

## What You Actually Need

| Goal | Keep |
| --- | --- |
| Transformers inference | `config.json`, all shards, the index, and the tokenizer files |
| vLLM / SGLang serving | The same set, plus `generation_config.json`; quantization settings travel inside `config.json` |
| Correct chat behaviour | The chat template — or, here, the `encoding/` folder |
| `llama.cpp` / Ollama / LM Studio | The `*.gguf` file only |
| Fine-tuning a base model | Everything above, plus the license and model card |
| Resuming a training run | The full checkpoint, including optimizer and scheduler state |

A practical rule: **weights alone are not a model**. The 167 GB of tensors here
are inert without the few megabytes of configuration, vocabulary, and
prompt-encoding files sitting next to them. Keep them together, or the download
cannot be reconstructed into a working model later.

## References

- [DeepSeek-V4-Flash-0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [Hugging Face model repository files](https://huggingface.co/docs/hub/en/models-uploading)
- [Transformers configuration reference](https://huggingface.co/docs/transformers/en/main_classes/configuration)
- [Transformers chat templates](https://huggingface.co/docs/transformers/en/chat_templating)
- [Tokenizers library](https://huggingface.co/docs/tokenizers/en/index)
- [PEFT adapter format](https://huggingface.co/docs/peft/en/index)
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
