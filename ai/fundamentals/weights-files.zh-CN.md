---
title: 模型文件夹里都有什么
description: 以 DeepSeek-V4-Flash 为例，逐个讲解真实 Hugging Face 仓库里的每个文件。
lang: zh
ref: weights-files
nav_order: 2
---

# 模型文件夹里都有什么

下载下来的模型是一个**软件包**，而不是单个文件。权重只是其中一部分，其余文件负责
描述如何构建模型类、如何把文本转成 token ID，以及如何拼装提示词。

本文以真实仓库
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
为例逐个文件讲解，再补充你在其他仓库里会遇到的文件。

## 示例仓库

```text
DeepSeek-V4-Flash-0731/                        # 合计 167 GB
├── .gitattributes                             #   1.52 kB
├── LICENSE                                    #   1.08 kB（MIT）
├── README.md                                  #   7.24 kB（模型卡）
├── config.json                                #   1.89 kB
├── generation_config.json                     #    170  B
├── model-00001-of-00048.safetensors           #   1.06 GB
├── model-00002-of-00048.safetensors           #   3.57 GB
├── ...                                        #   每个约 3.6 GB
├── model-00048-of-00048.safetensors
├── model.safetensors.index.json               #   5.60 MB
├── tokenizer.json                             #   6.37 MB
├── tokenizer_config.json
├── encoding/                                  # 提示词编码（没有 Jinja 模板）
│   ├── README.md
│   ├── encoding_dsv4.py
│   ├── test_encoding_dsv4.py
│   └── tests/test_input_*.json
└── inference/                                 # 官方本地推理参考代码
    ├── README.md
    ├── config.json
    ├── convert.py
    ├── generate.py
    └── kernel.py
```

有两点一眼可见：

- 48 个权重分片占了下载量的 99% 以上，而让这些张量真正可用的所有文件加起来只有
	几 MB。
- 这里**没有** `chat_template.jinja`，也没有 `.bin` 文件。这两点都是发布方的刻意
	选择，下文会解释。

## 速查表

| 文件 | 类别 | 运行必需？ | 本仓库有？ | 作用 |
| --- | --- | --- | :---: | --- |
| `*.safetensors` | 权重 | 是 | 有 | 真正的张量（参数） |
| `model.safetensors.index.json` | 权重 | 分片时必需 | 有 | 把每个张量名映射到所在分片 |
| `pytorch_model*.bin` | 权重 | 没有 SafeTensors 时必需 | 无 | 基于 pickle 的旧版权重文件 |
| `*.gguf` | 权重 + 元数据 | 是 | 无 | 面向 GGML 系运行时的自包含量化模型 |
| `config.json` | 架构 | 是 | 有 | 模型类、层规模与量化设置 |
| `generation_config.json` | 推理 | 否 | 有 | 默认采样与停止设置 |
| `tokenizer.json` | 分词器 | 是 | 有 | 快速分词器：词表加合并与归一化规则 |
| `tokenizer_config.json` | 分词器 | 是 | 有 | 分词器类、特殊 token 与上下文长度 |
| `tokenizer.model` / `vocab.json` + `merges.txt` | 分词器 | 有时需要 | 无 | 慢速或旧版分词器的原始词表 |
| `special_tokens_map.json` | 分词器 | 否 | 无 | 指明 BOS、EOS、PAD、UNK 分别是哪个 token |
| `chat_template.jinja` | 提示词 | 非必需，但对话场景很关键 | 无 | 把消息列表拼成模型期望的提示词字符串 |
| `preprocessor_config.json` | 多模态 | 视觉/音频模型必需 | 无 | 图像或音频预处理参数 |
| `adapter_model.safetensors` + `adapter_config.json` | 微调 | LoRA 必需 | 无 | 叠加在基座模型之上的小增量 |
| `README.md`、`LICENSE`、`.gitattributes` | 元数据 | 否 | 有 | 文档、使用条款与 Git LFS 规则 |

## 权重文件

### `model-000NN-of-00048.safetensors`

参数本体。大模型会被**分片（sharding）**，让每个文件保持在合理体积，从而可以独立
下载、断点续传、缓存和内存映射。这里第 1 个分片是 1.06 GB（嵌入层和最前面的稠密
层），其余 47 个各约 3.6 GB。

`model-00002-of-00048.safetensors` 表示"第 2 片，共 48 片"。分片不是可以挑着下载的
层分组——48 个一个都不能少。

文件内部是一个 JSON 头加上原始张量缓冲区，因此加载器可以先读头部，再只映射需要的
张量，全过程不执行文件里的任何代码。

> **约 3000 亿参数的模型为什么只有 167 GB？** `config.json` 里的
> `quantization_config` 表明检查点以 FP8（`e4m3`）存储，权重分块为 128×128；而
> `expert_dtype` 是 `fp4`，即 MoE 专家按 4 bit 存储。平均下来大约每参数 4-5 bit，
> 而不是 BF16 检查点的 16 bit。

### `model.safetensors.index.json`

分片索引表。这里有 5.6 MB，是因为一个 43 层、每层 256 个专家的 MoE 模型拥有数量
惊人的独立命名张量。它包含两部分：

- `metadata.total_size`——所有权重的总字节数，可用来做显存/内存预估。
- `weight_map`——每个张量一条记录，例如
	`"model.layers.20.mlp.experts.7.down_proj.weight": "model-00023-of-00048.safetensors"`。

没有这个索引，加载器就得打开全部 48 个分片去找一个张量。索引缺失或过期时，即使
分片齐全也会加载失败。

### 本仓库*没有*的文件

- **`pytorch_model*.bin`**——较早的 PyTorch pickle 格式，索引是
	`pytorch_model.bin.index.json`。很多仓库两种格式都提供，这时只需要 SafeTensors
	那一套。pickle 文件在加载时可以执行任意代码，因此只从可信来源加载 `.bin` 权重。
- **`*.gguf`**——`llama.cpp`、Ollama、LM Studio 使用的单一自描述文件。GGUF 把架构
	元数据、分词器和对话模板都嵌进权重文件，所以 GGUF 下载不需要 `config.json` 或
	`tokenizer.json`。后缀标明量化方式，例如 `Q4_K_M`。GGUF 版本由本仓库这样的检查点
	转换而来，通常由社区而非原发布方制作。

## `config.json`

告诉框架在加载任何权重之前**要构建什么**。以下是本仓库的真实取值：

| 字段 | 取值 | 含义 |
| --- | --- | --- |
| `architectures` | `["DeepseekV4ForCausalLM"]` | 要实例化的模型类 |
| `model_type` | `deepseek_v4` | 供 Auto 类查找的架构短标识 |
| `transformers_version` | `4.57.1` | 该配置对应的框架版本 |
| `hidden_size` | `4096` | 残差流宽度 |
| `num_hidden_layers` | `43` | 层数 |
| `num_attention_heads` / `num_key_value_heads` | `64` / `1` | 两者不等表示分组查询注意力；KV 头为 1 即多查询注意力，可大幅缩小 KV 缓存 |
| `vocab_size` | `129280` | 嵌入矩阵行数，必须与分词器一致 |
| `max_position_embeddings` | `1048576` | 100 万 token 上下文 |
| `rope_theta`、`rope_scaling` | `10000`，YARN `factor: 16`，`original_max_position_embeddings: 65536` | 在 64K 上训练，通过位置编码缩放扩展到 1M |
| `torch_dtype` | `bfloat16` | 运行时张量反量化后的数据类型 |
| `quantization_config` | `quant_method: fp8`、`fmt: e4m3`、`weight_block_size: [128, 128]` | 如何解读存储的低比特权重 |
| `tie_word_embeddings` | `false` | 输出头拥有独立的权重矩阵 |

MoE 相关字段描述路由方式：

| 字段 | 取值 | 含义 |
| --- | --- | --- |
| `n_routed_experts` | `256` | 每个 MoE 层的专家数 |
| `n_shared_experts` | `1` | 与路由无关、始终参与计算的共享专家 |
| `num_experts_per_tok` | `6` | 每个 token 激活的专家数——这正是每 token 只跑一小部分参数的原因 |
| `moe_intermediate_size` | `2048` | 单个专家的宽度 |
| `expert_dtype` | `fp4` | 专家以 4 bit 存储 |

还有几个字段只对该架构有意义：`num_nextn_predict_layers`、
`dspark_target_layer_ids` 和 `dspark_block_size` 属于随同一份检查点发布的
**DSpark 投机解码模块**。这也是模型卡里的 vLLM / SGLang 命令能直接开启投机解码、
而无需另外指定草稿模型的原因。

> 一旦 `config.json` 与权重不一致，加载会报形状不匹配的错误。只有在完全清楚自己
> 在改什么时才动它——本模型卡就明确说明：想让专家用 FP8 而非 FP4，就删掉
> `expert_dtype`。

## `generation_config.json`

与架构无关的默认**解码**设置。整个文件只有 170 字节：

```json
{
  "bos_token_id": 0,
  "eos_token_id": 1,
  "do_sample": true,
  "temperature": 1.0,
  "top_p": 1.0
}
```

其他仓库还会在这里加上 `top_k`、`max_new_tokens` 或 `repetition_penalty`。实践中
最关键的是 `eos_token_id`：取值错误是模型停不下来的常见原因。另外要注意，这些默认
值不一定等于发布方的推荐值——本模型卡在智能体场景下建议 `top_p = 0.95`。运行时自身
的设置会覆盖本文件。

## 分词器文件

分词器把文本转成 token ID，而这些 ID 会直接索引嵌入矩阵。用了别的模型的分词器，
即使张量形状恰好对得上，输出也会是无意义的内容。

### `tokenizer.json`

由 Rust `tokenizers` 库序列化出来的完整**快速分词器**：归一化器、预分词器、完整的
129,280 条词表、合并规则、后处理器和解码器。6.37 MB，是仓库里第二大的文件。只要它
存在，单个文件就够用了——这也是这里没有 `tokenizer.model`、`vocab.json` 和
`merges.txt` 的原因。

### `tokenizer_config.json`

描述如何构造分词器：

| 字段 | 取值 |
| --- | --- |
| `tokenizer_class` | `PreTrainedTokenizerFast` |
| `model_max_length` | `1048576` |
| `add_bos_token` / `add_eos_token` | `false` / `false` |
| `bos_token` | `<｜begin▁of▁sentence｜>` |
| `eos_token` | `<｜end▁of▁sentence｜>` |
| `pad_token` | `<｜end▁of▁sentence｜>`，与多数因果语言模型一样复用 EOS |
| `unk_token` | `null`；字节级 BPE 没有未知词 token |

`add_bos_token: false` 值得留意：BOS token 由调用方负责添加，因为提示词格式是在
分词器之外拼装的。

### 在其他仓库会见到的分词器文件

- `tokenizer.model`——SentencePiece 二进制词表（Llama 2、Mistral、Gemma）。
- `vocab.json` + `merges.txt`——BPE 词表与合并优先级（GPT-2 风格）。
- `vocab.txt`——WordPiece 词表（BERT 风格）。
- `special_tokens_map.json`——把 `bos_token`、`eos_token`、`pad_token`、`unk_token`
	映射到具体 token。本仓库把这些信息合并进了 `tokenizer_config.json`。

## 提示词拼装：`encoding/` 目录

多数指令模型会提供 `chat_template.jinja`，或在 `tokenizer_config.json` 里放一个
`chat_template` 字符串，用来把消息列表拼成模型微调时使用的确切提示词。**本次发布
两者都没有。** 模型卡直接说明了这一点，并改为提供 `encoding/encoding_dsv4.py`：

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

原因是 Jinja 模板表达不了这个模型需要的东西：两种思考模式、丢弃早前轮次的推理
内容、以提示词前缀形式注入的三档 `reasoning_effort`、DSML 工具调用块，以及
`<｜title｜>`、`<｜query｜>` 这类辅助任务 token。该目录还提供了反向函数
`parse_message_from_completion_text`，把原始输出拆回 `reasoning_content`、
`content` 和 `tool_calls`。

不管以哪种形式存在，这都是最容易被跳过的文件——而跳过它正是指令模型回答变差或
啰嗦跑题的最常见原因：你是在用模型微调时从未见过的格式提问。

## `inference/` 目录

参考代码，不属于模型本身：

- `convert.py`——按给定的专家数与模型并行度，把 Hugging Face 分片转换成该 demo
	需要的布局。
- `generate.py`——用 `torchrun` 启动的交互式与批量对话入口。
- `kernel.py`——自定义量化算子。
- `config.json`——这个 demo *专用*的配置，不是仓库根目录那个 Transformers 配置。

当模型需要上游框架尚未支持的算子或启动路径时，发布方就会附上这类目录。用 vLLM 或
SGLang 部署时完全可以忽略它。

## 其他类型模型中的文件

### 多模态模型

- `preprocessor_config.json`——图像缩放、裁剪、缩放系数与归一化参数，或音频采样
	参数。
- `processor_config.json`——分词器与特征提取器如何组合。

### LoRA 等适配器

- `adapter_model.safetensors`——只包含低秩增量权重，通常是 MB 级而非 GB 级。
- `adapter_config.json`——`base_model_name_or_path`、秩 `r`、`lora_alpha` 以及目标
	模块名。

适配器单独存在毫无意义，必须叠加到它所指明的那个基座模型上。

### 其他量化检查点

GPTQ 和 AWQ 版本使用独立的 `quantize_config.json`，里面是 `bits`、`group_size`、
`sym`、`desc_act`，而不是本仓库这种写在 `config.json` 里的 `quantization_config`
块。量化元数据对不上时不会干净地报错，而是直接输出乱码，所以务必让它跟权重待在
一起。

### 带自定义代码的模型

`modeling_*.py` 和 `configuration_*.py` 承载架构实现，出现在架构尚未合入
Transformers 的仓库中。加载它们需要 `trust_remote_code=True`，这会执行发布方的
Python 代码。本仓库根目录没有这类文件，因为 Transformers 4.57.1 已原生支持
`deepseek_v4`。

### 训练与微调的残留文件

`optimizer.pt`、`scheduler.pt`、`trainer_state.json`、`training_args.bin` 和
`rng_state.pth` 属于训练检查点，用于恢复训练。仅 `optimizer.pt` 就可能是权重体积的
两倍。只做推理部署时可以删掉。

## 到底需要哪些文件

| 目标 | 需要保留 |
| --- | --- |
| Transformers 推理 | `config.json`、全部分片、索引文件，以及分词器文件 |
| vLLM / SGLang 部署 | 同上，再加 `generation_config.json`；量化设置随 `config.json` 一起 |
| 对话行为正确 | 对话模板——在本仓库里是 `encoding/` 目录 |
| `llama.cpp` / Ollama / LM Studio | 只要 `*.gguf` 文件 |
| 基于基座模型微调 | 以上全部，再加许可证与模型卡 |
| 恢复训练 | 完整检查点，包含优化器与调度器状态 |

一条实用原则：**光有权重不等于有模型**。这里 167 GB 的张量，离开旁边那几 MB 的
配置、词表和提示词编码文件就是一堆死数据。把它们放在一起保存，否则以后无法把这次
下载还原成一个能跑的模型。

## 参考资料

- [DeepSeek-V4-Flash-0731 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [Hugging Face 模型仓库文件说明](https://huggingface.co/docs/hub/en/models-uploading)
- [Transformers 配置参考](https://huggingface.co/docs/transformers/en/main_classes/configuration)
- [Transformers 对话模板](https://huggingface.co/docs/transformers/en/chat_templating)
- [Tokenizers 库](https://huggingface.co/docs/tokenizers/en/index)
- [PEFT 适配器格式](https://huggingface.co/docs/peft/en/index)
- [GGUF 规范](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
