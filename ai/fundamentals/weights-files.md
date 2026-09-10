# What Is Inside an LLM Model Repository?

A downloadable model is usually a **package of related files**, not one file.
The package has four essential jobs:

| Part | Question it answers | Typical files |
| --- | --- | --- |
| **Weights** | What values did the model learn? | `model.safetensors`, weight shards, or a `.gguf` file |
| **Configuration** | What architecture should be built around those values? | `config.json` |
| **Tokenizer or processor** | How is user input converted to model inputs and back? | `tokenizer.json`, `tokenizer.model`, or processor files |
| **Runtime code** | How are the layers and forward pass implemented? | Usually supplied by Transformers, llama.cpp, or another runtime |

Generation defaults, documentation, licenses, and training state may also be
included. A weight file alone is therefore not always enough to run a model.

## Example: DeepSeek-V4-Flash

The official [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
release is a text-generation model distributed with sharded SafeTensors weights.
Its repository is a useful example because the model package must describe not
only many weight shards, but also a DeepSeek V4 architecture and FP8 weight
representation. The abbreviated tree below intentionally omits the exact shard
count because repositories can be revised:

```text
DeepSeek-V4-Flash/
|-- README.md
|-- LICENSE
|-- config.json
|-- generation_config.json
|-- model.safetensors.index.json
|-- model-00001-of-00NNN.safetensors
|-- ...
|-- model-00NNN-of-00NNN.safetensors
|-- tokenizer.json
|-- tokenizer_config.json
`-- ...
```

`00NNN` is a placeholder, not the published shard count. Consult
`model.safetensors.index.json` for the files required by the revision being
downloaded. File names are conventions rather than a universal standard, so a
repository may omit some files or include additional code and metadata.

### File-by-File Meaning

| File | What it contains | When it is used |
| --- | --- | --- |
| `model-xxxxx-of-yyyyy.safetensors` | Named tensors containing the learned parameters; DeepSeek-V4-Flash uses many shards | Loaded for inference or fine-tuning; every shard named by the index is part of the full checkpoint |
| `model.safetensors.index.json` | A map from each tensor name to the shard that stores it | Lets a loader find tensors without scanning or loading every shard first |
| `config.json` | Architecture type and dimensions, such as layer count, hidden size, attention heads, vocabulary size, and activation | Used to construct the model object before filling it with weights |
| `tokenizer.json` | A complete fast-tokenizer definition: vocabulary, tokenization model, normalization, and pre/post-processing rules | Converts text to token IDs and token IDs back to text |
| `tokenizer_config.json` | Tokenizer options, model input limits, special-token behavior, and often the chat template | Controls how the tokenizer is instantiated and how chat messages are formatted |
| `special_tokens_map.json` | Names the beginning, end, padding, unknown, and other special tokens | Older or compatibility-oriented tokenizer loading; newer repositories may keep the same data elsewhere |
| `generation_config.json` | Suggested decoding defaults such as end-token IDs, sampling settings, or beam-search settings | Read by generation APIs; callers can override every value |
| `README.md` | Model card: intended use, examples, limitations, evaluation, and provenance | Read by people; it is not required for tensor computation |
| `LICENSE` | Legal terms for using and redistributing the model | Must be checked before use or distribution; it is not loaded by the model runtime |

The index is small JSON data compared with the weights. This deliberately
abbreviated DeepSeek-V4-Flash-style example omits the release-specific shard
count and most tensor entries:

```json
{
  "weight_map": {
		"model.embed_tokens.weight": "model-00001-of-00NNN.safetensors",
		"model.layers.0.self_attn.q_a_proj.weight": "model-00001-of-00NNN.safetensors",
		"model.layers.0.self_attn.kv_proj.weight": "model-00001-of-00NNN.safetensors",
		"model.layers.3.mlp.experts.gate_up_proj": "model-0000X-of-00NNN.safetensors",
		"model.norm.weight": "model-00NNN-of-00NNN.safetensors",
		"lm_head.weight": "model-00NNN-of-00NNN.safetensors"
  }
}
```

A real index also has `metadata.total_size`, the total tensor data size in bytes.
The exact tensor names and shard boundaries depend on the model revision and
exporting library; use the actual index rather than constructing shard names
from this illustration.

## What Is Inside a Weight File?

A SafeTensors checkpoint is conceptually a **state dictionary**: tensor names
mapped to multidimensional arrays. For each tensor, the file records:

- its name, such as `model.layers.0.self_attn.q_a_proj.weight`;
- its data type, such as BF16, FP16, or an integer type;
- its shape, such as `[4096, 4096]`;
- byte offsets locating its data in the file; and
- optionally, string metadata about the checkpoint.

The numeric tensor data follows the metadata header. SafeTensors does not store
executable Python objects, which is why it is safer to open than a pickle-based
PyTorch checkpoint from an unknown source.

A decoder-only DeepSeek mixture-of-experts model contains tensor groups like
these. The exact names should be checked in its shard index:

| Example tensor name | Meaning |
| --- | --- |
| `model.embed_tokens.weight` | Lookup table that turns each token ID into a hidden vector |
| `model.layers.0.self_attn.q_a_proj.weight` | First low-rank query projection in the attention block |
| `model.layers.0.self_attn.q_a_norm.weight` | Normalization applied between the two query projections |
| `model.layers.0.self_attn.q_b_proj.weight` | Expands the low-rank query representation into attention heads |
| `model.layers.0.self_attn.kv_proj.weight` | Produces the shared key/value representation; V4 uses one KV head for all query heads |
| `model.layers.0.self_attn.o_a_proj.weight` | First, grouped stage of the attention output projection |
| `model.layers.0.self_attn.o_b_proj.weight` | Maps the grouped attention output back to the model's hidden size |
| `model.layers.0.self_attn.compressor.kv_proj.weight` | Projects states for the compressed long-range attention branch |
| `model.layers.3.mlp.gate.weight` | Router that scores which experts should process each token |
| `model.layers.3.mlp.experts.gate_up_proj` | Packed 3D gate/up weights for every routed expert in that layer |
| `model.layers.3.mlp.experts.down_proj` | Packed 3D down-projection weights for every routed expert |
| `model.layers.3.mlp.shared_experts.gate_proj.weight` | Gate projection in the shared expert that processes every token |
| `model.layers.0.input_layernorm.weight` | Scale parameters for normalization before attention |
| `model.layers.0.post_attention_layernorm.weight` | Scale parameters for normalization before the feed-forward block |
| `model.norm.weight` | Final normalization parameters |
| `lm_head.weight` | Maps final hidden vectors to vocabulary logits for next-token prediction |

DeepSeek V4 also has learned compressor, indexer, attention-sink, and
manifold-constrained hyper-connection parameters. Names can still vary between
model revisions and runtime conversions. Bias tensors may be present, and
`lm_head.weight` may be tied to the token embedding rather than stored as a
separate copy. The FP8 DeepSeek-V4-Flash checkpoint also stores scale tensors
needed to reconstruct or execute its low-precision matrix weights.

The weight file normally does **not** explain the complete forward pass. The
runtime combines its model implementation with `config.json`, then matches the
implementation's parameter names and shapes to the stored tensors. A wrong
architecture or incompatible config produces missing-key, unexpected-key, or
shape-mismatch errors.

## Important Configuration Files

### `config.json`: The Blueprint

A shortened DeepSeek-V4-Flash architecture configuration has fields like these:

```json
{
	"architectures": ["DeepseekV4ForCausalLM"],
	"model_type": "deepseek_v4",
	"vocab_size": 129280,
	"hidden_size": 4096,
	"moe_intermediate_size": 2048,
	"num_hidden_layers": 43,
	"num_attention_heads": 64,
	"num_key_value_heads": 1,
	"head_dim": 512,
	"q_lora_rank": 1024,
	"n_routed_experts": 256,
	"n_shared_experts": 1,
	"num_experts_per_tok": 6,
	"max_position_embeddings": 1048576,
	"sliding_window": 128
}
```

The runtime uses `model_type` and, when present, `architectures` to select an
implementation, then uses the dimensions to create tensors with the expected
shapes. The expert fields describe the mixture-of-experts layout: the checkpoint
contains 256 routed experts per layer, but only 6 are selected for each token,
plus one shared expert.
`torch_dtype` describes a preferred or original dtype, but loaders may choose a
different runtime dtype. `quantization_config` tells a compatible runtime how to
interpret the FP8 tensors and scales; an FP8 tag alone is not sufficient
implementation detail.

### Tokenizer Files: The Vocabulary Contract

The model only sees integer token IDs. The tokenizer files define the exact
mapping between text and those IDs. Using a tokenizer from a merely similar
model can silently produce poor output because the same ID may represent a
different byte sequence or token.

Common alternatives are:

| Files | Tokenizer family |
| --- | --- |
| `tokenizer.json` | Single-file fast tokenizer used by the Hugging Face Tokenizers library |
| `tokenizer.model` | SentencePiece model, common in Llama-derived and multilingual models |
| `vocab.json` plus `merges.txt` | Byte-pair encoding layout used by GPT-2-like tokenizers |
| `vocab.txt` | WordPiece vocabulary used by BERT-like tokenizers |
| `added_tokens.json` | Tokens added after the original tokenizer was trained |

A chat model also needs the correct chat template, often stored in
`tokenizer_config.json` or a dedicated template file. The template inserts the
roles, separators, and control tokens expected during instruction tuning.

## How a Loader Uses the Package

Transformers can resolve the package automatically:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "deepseek-ai/DeepSeek-V4-Flash"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
	model_id,
	 device_map="auto",
)

inputs = tokenizer("Weight files contain", return_tensors="pt").to(model.device)
output_ids = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

This is the standard package-loading pattern, not a promise that the checkpoint
fits on one workstation. DeepSeek-V4-Flash requires a runtime and aggregate
accelerator memory that support its architecture and FP8 representation; follow
the model card's serving instructions for real deployment.

At a high level, the loader:

1. reads `config.json` and constructs the architecture;
2. reads the shard index, if present;
3. loads each named tensor into its matching model parameter;
4. constructs the tokenizer from its own files; and
5. applies generation defaults when `generate()` is called.

To inspect tensor names directly without constructing the model:

```python
from pathlib import Path

from safetensors import safe_open

model_directory = Path("DeepSeek-V4-Flash")
first_shard = next(model_directory.glob("model-*.safetensors"))

with safe_open(
	first_shard,
	 framework="pt",
	 device="cpu",
) as checkpoint:
	 print(checkpoint.metadata())
	 print(list(checkpoint.keys())[:10])
	 embedding = checkpoint.get_tensor("model.embed_tokens.weight")
	 print(embedding.shape, embedding.dtype)
```

`get_tensor()` materializes that tensor in memory, so avoid retrieving very
large tensors merely to list the keys.

## Other Common Model Packages

### One Unsharded SafeTensors File

Small models may use this layout:

```text
SmallLM/
|-- config.json
|-- model.safetensors
|-- tokenizer.json
`-- tokenizer_config.json
```

There is no index because one file contains every weight tensor.

### GGUF for Local Inference

```text
DeepSeek-V4-Flash-GGUF/
|-- README.md
|-- DeepSeek-V4-Flash-Q4_K_M-00001-of-000NN.gguf
|-- ...
`-- DeepSeek-V4-Flash-Q4_K_M-000NN-of-000NN.gguf
```

A GGUF file commonly bundles quantized weights, tensor metadata, architecture
metadata, and tokenizer metadata into one memory-mappable file. It is intended
for GGML-compatible runtimes such as llama.cpp rather than being a drop-in
replacement for SafeTensors in Transformers.

The layout above represents a possible community conversion, not the official
SafeTensors repository. Very large GGUF models are commonly split into multiple
files, and the actual quantization and shard count depend on the publisher.

The suffix `Q4_K_M` identifies a particular 4-bit-family GGUF quantization, not
the model architecture. Multimodal GGUF distributions may include a separate
vision projector file, often named with `mmproj`.

### LoRA or PEFT Adapter

```text
DeepSeek-V4-Flash-Domain-LoRA/
|-- README.md
|-- adapter_config.json
`-- adapter_model.safetensors
```

`adapter_model.safetensors` stores only the learned low-rank changes, usually a
small fraction of the full model. `adapter_config.json` identifies settings such
as rank, scaling, targeted modules, and the expected base model. The adapter
cannot normally run alone; a PEFT-compatible loader applies it to the matching
base model. It can also be merged into a copy of the base weights for deployment.

### Legacy PyTorch Checkpoint

Older repositories may contain `pytorch_model.bin`, or sharded `.bin` files plus
`pytorch_model.bin.index.json`. They serve the same basic role as SafeTensors
weights, but are commonly serialized with Python pickle. Load them only from a
trusted source because pickle data can execute code during deserialization.

Files named `.pt`, `.pth`, or `.ckpt` are generic checkpoint names. Their
contents are application-defined: one might contain only a model state
dictionary, while another also contains optimizer state and arbitrary Python
objects.

### Training-Resume Checkpoint

```text
checkpoint-12000/
|-- config.json
|-- model.safetensors
|-- optimizer.pt
|-- scheduler.pt
|-- trainer_state.json
|-- rng_state.pth
`-- tokenizer.json
```

The model weights are enough to start ordinary inference or a new fine-tuning
run. The other files preserve training progress:

| File | Purpose |
| --- | --- |
| `optimizer.pt` | Optimizer moments and other state needed to continue with the same optimization trajectory |
| `scheduler.pt` | Learning-rate scheduler position and state |
| `trainer_state.json` | Step count, metrics, best-checkpoint information, and trainer bookkeeping |
| `rng_state.pth` | Random-number generator states used to make a resumed run more reproducible |
| `scaler.pt` | Optional gradient-scaler state for FP16 mixed-precision training |

These files can be much larger than expected: Adam-style optimizer state may
consume more storage than the model weights. They are not needed for serving.

### Multimodal and Exported Models

Image, audio, or video models may add `preprocessor_config.json`,
`processor_config.json`, feature-extractor files, or separate projector weights.
These define resizing, normalization, sampling rates, and how multiple input
modalities are assembled.

Deployment exports use different artifacts, for example `model.onnx` plus
external tensor-data files, or a TensorRT `.engine`/`.plan`. Such files combine
or compile graph structure and weights for a target runtime. They are derived
deployment artifacts, not interchangeable source checkpoints.

## Practical Checklist

Before downloading or loading a model, check:

1. **Is it a full model or only an adapter?** Look for full weight shards versus
	`adapter_model.safetensors`.
2. **Does the runtime support the format and quantization?** SafeTensors, GGUF,
	GPTQ, and AWQ require different loaders or kernels.
3. **Are all shards present?** Every file named by the index is required.
4. **Is the tokenizer included and matched to the model?** Similar vocabulary
	sizes do not guarantee compatible token IDs.
5. **How much memory is required?** File size approximates weight storage, not
	total runtime memory; the KV cache and temporary buffers need additional RAM
	or VRAM.
6. **Does it require custom code?** Repositories with custom modeling Python may
	ask for `trust_remote_code=True`, which executes repository code and should be
	reviewed first.
7. **What does the license permit?** Open weights do not necessarily mean
	unrestricted commercial use or redistribution.

## References

- [DeepSeek-V4-Flash model repository](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [DeepSeek V4 implementation in Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/deepseek_v4)
- [Transformers model loading, configuration, and sharded checkpoints](https://huggingface.co/docs/transformers/main/en/models)
- [Transformers chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [SafeTensors format](https://github.com/huggingface/safetensors)
- [PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint)
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
