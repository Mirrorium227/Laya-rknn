# Laya-rknn

**简体中文** | [English](README.en.md)

[![GitHub stars](https://img.shields.io/github/stars/Mirrorium227/Laya-rknn?style=flat)](https://github.com/Mirrorium227/Laya-rknn/stargazers) [![GitHub issues](https://img.shields.io/github/issues/Mirrorium227/Laya-rknn)](https://github.com/Mirrorium227/Laya-rknn/issues) [![Platform](https://img.shields.io/badge/platform-RK3576-blue)](https://github.com/Mirrorium227/Laya-rknn) [![Precision](https://img.shields.io/badge/precision-FP16-green)](#rk3576-实测结果) [![RKNN](https://img.shields.io/badge/RKNN-2.3.2-orange)](https://github.com/airockchip/rknn-toolkit2)

在 RK3576 上运行 Laya multilingual，使用 RKNN 加速推理。提供 96、256、1024 三个固定长度版本，每个版本由三张 FP16 图组成，支持完整的 Laya API 输出。

项目地址：[Mirrorium227/Laya-rknn](https://github.com/Mirrorium227/Laya-rknn)。问题与建议可提交到 [Issues](https://github.com/Mirrorium227/Laya-rknn/issues)。

## RK3576 实测结果

2026-10-03，在 RK3576 上测量已预热的完整 `Agent.predict` 延迟，包含分词、CPU 前后处理、RKNN 运算和图间传输。96桶使用30条输入、每条3轮，256和1024桶各使用12条输入、每条3轮。算子 profiling 在独立进程中采集。

| 长度 | 原 CPU P50/P95 ms | 三图 FP16 P50/P95 ms | P50 加速比 |
|---|---:|---:|---:|
| 96 | 844.39 / 950.37 | 102.09 / 103.87 | 8.27× |
| 256 | 1994.87 / 2452.93 | 290.38 / 363.26 | 6.87× |
| 1024 | 8640.98 / 11542.18 | 2809.23 / 3094.45 | 3.08× |

36条独立校准输入的类别与原CPU一致，输出均为有限值。主评测54条加 MASSIVE 补充144条，共198条输入；其中1条在 FP16 下改变类别，位于96桶，占0.51%。原18条人工夹具的 CPU 和 FP16 准确率均为13/18。MASSIVE 补充集将17个意图映射到六个功能域，新增映射标签和长输入标签的审核状态为待人工复核。

| 主评测长度 | 类别翻转 | 概率最大绝对误差 | logits max / MAE / RMSE |
|---|---:|---:|---:|
| 96 | 1/30 | 0.0185 | 0.08629 / 0.01860 / 0.02445 |
| 256 | 0/12 | 0.0205 | 0.19372 / 0.04126 / 0.06368 |
| 1024 | 0/12 | 0.0711 | 0.52185 / 0.12671 / 0.17486 |

三个桶的实机 profile 均记录531个 NPU 节点和26个 CPU 节点。NPU dtype 为529个 F16、1个 FLOAT 和1个 INT8掩码 Where。

完整 CPU Agent 和三个桶同时加载后，一次加载与路由检查测得 RSS约3.43GiB、HWM约3.62GiB。图间传输量随长度 L 的平方增长；1024桶已有样本的逐条 P50 加速比为1.16–4.28倍。行动头在现有样本中输出的概率集中于1。

## 模型与计算分工

Laya 0.3.22 是兼容 Jev 接口的开源决策模型。本项目使用 [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)，固定 revision 为 `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`，原权重 SHA256 为 `9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`。

网络包含 256000×768 词表、22 层 mmBERT 编码器、类型嵌入、2 层决策 Transformer、共享 scorer 和行动头。模型将问题、候选和 state 编码为一个序列，在候选 marker 处计算 logits，再由 CPU 解码为概率和答案。行动头接收 CLS 向量和四个统计特征。

| 位置 | 运算 |
|---|---|
| CPU | 分词、词表查表、两处浮点 NaN 清零、头部 key bias、原行动头、温度/概率与 API 解码 |
| RKNN | embedding norm、22 层编码器、类型嵌入、2 层决策头、scorer；图内部分算子由 CPU 执行 |

两处注意力 NaN 保护是三张图的切分点。CPU 清除 NaN、保留 ±Inf，然后将概率、V 和残差送入下一张图。主要浮点计算使用 FP16，掩码和索引采用对应的整数类型。

适配器沿用 `Agent.predict(state, questions)`，返回 answers、probabilities、action 和 usage 等原有字段。每次 NPU 调用处理一行输入，批量输入逐行执行。每个问题支持最多6个候选；根据问题、候选、state 和特殊 token 的总长度选择96、256或1024桶。超过1024 token或6个候选时抛出 `ValueError`，问题和候选的编码预算沿用上游规则。

## 仓库与模型文件

仓库包含运行源码、依赖说明、三个桶的形状 manifest 和 [模型校验清单](delivery/fp16/model_manifest.json)。在电脑上克隆源码：

```bash
git clone https://github.com/Mirrorium227/Laya-rknn.git
cd Laya-rknn
```

FP16 模型包单独分发，下载地址：[谷歌云盘](https://drive.google.com/drive/folders/1zbYvTjf-k_drJArINAAcPtpKbAms7apF?usp=sharing) · [123云盘](https://1815656308.share.123pan.cn/123pan/kEBzVv-RS1tA)。

模型包包含9个 RKNN 文件、3个形状 manifest、`delivery/fp16/model_manifest.json`、模型包说明和一个 `.7z` 文件。部署时另外准备完整 Laya 权重、Rockchip Runtime、uv 和 Python 依赖，步骤见下文。校验清单同时记录模型文件和 Runtime 的 SHA256。

在电脑上下载并解压模型包，将解压目录中的 `bucket-study` 和 `delivery` 合并到仓库根目录。以下 Linux 示例将解压目录记为 `fp16_assets`：

```bash
# 从仓库根目录执行；先将下载的模型包解压到fp16_assets。
ASSET_ROOT="$PWD/fp16_assets"
# 若压缩包带有外层文件夹，将ASSET_ROOT改为直接包含bucket-study、delivery的目录。
cp -a "$ASSET_ROOT/bucket-study/." ./bucket-study/
cp -a "$ASSET_ROOT/delivery/." ./delivery/
```

部署后的文件路径如下：

```text
models/laya-multilingual/                       # 固定 revision 的完整模型快照
  model.safetensors
  rl_agent_config.json
  encoder/config.json
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
rknn-2.3.2/lib/librknnrt.so                     # ARM64 Runtime 2.3.2
bucket-study/artifacts/l96_split_fp16/part0/model.rknn
bucket-study/artifacts/l96_split_fp16/part1/model.rknn
bucket-study/artifacts/l96_split_fp16/part2/model.rknn
bucket-study/artifacts/l256_split_fp16/part{0,1,2}/model.rknn
bucket-study/artifacts/l1024_split_fp16/part{0,1,2}/model.rknn
```

`part{0,1,2}` 表示三个独立目录。三个桶的形状 manifest 位于 `bucket-study/split/l{96,256,1024}/manifest.json`。

Runtime 来自 SDK 2.3.2 的 `rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so`，SHA256 为 `d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8`。代码通过仓库内的绝对路径加载该库，并核对 Runtime 版本和 `/proc/self/maps`。实测版本为 `2.3.2 (429f97ae6b@2025-04-09T09:09:27)`，驱动为0.9.8。

## 首次部署

运行环境为 Linux ARM64、Python3.11.2，依赖见 [requirements.txt](requirements.txt)。以下使用 `/home/radxa/Laya_rknn` 作为板端仓库目录，Runtime 路径随仓库位置确定。

### 1. 准备完整 Laya 权重

在已安装 uv 的联网电脑上，从仓库根目录执行以下命令，下载固定 revision 的模型快照。完成后将 `models/laya-multilingual` 上传到板端仓库根目录。

```bash
uv run --with huggingface-hub==0.36.2 python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='convaiinnovations/laya-multilingual',
    revision='e4e9ddf21a7b1903b7acffd8814ad4307bf63a67',
    local_dir='models/laya-multilingual',
    allow_patterns=['model.safetensors', 'rl_agent_config.json', 'encoder/*', 'tokenizer/*'],
)
PY
```

离线加载使用完整快照中的权重、`encoder` 配置和 `tokenizer` 文件。原权重 SHA256 见前文。

### 2. 准备 Runtime、uv 与 Python 依赖

- 将 RKNN Toolkit 2.3.2 SDK 中的 ARM64 `librknnrt.so` 放到 `rknn-2.3.2/lib/`。SDK来源为 [Rockchip RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2)，库文件按前文 SHA256 核对。
- 将 Linux ARM64 的 uv 可执行文件放到 `tools/uv/uv`，安装方法见 [uv 官方说明](https://docs.astral.sh/uv/getting-started/installation/)。
- 板端使用 `/usr/bin/python3.11`。离线安装时准备兼容 Linux ARM64 / CPython3.11 的 `wheelhouse`，包含 `requirements.txt` 及全部间接依赖，其中 torch 为 `2.9.1+cpu`。

推理通过 C API 调用项目内的 Runtime；Python 环境负责分词、CPU 运算和 API 输出。

### 3. 建立环境并安装依赖

在仓库根目录设置环境变量、创建虚拟环境并安装依赖。每次打开新终端时重新设置这些变量，缓存、字节码和临时文件都写入项目目录。

```bash
cd /home/radxa/Laya_rknn
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_DOWNLOADS=never
export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR"
chmod +x tools/uv/uv
tools/uv/uv venv --python /usr/bin/python3.11 .venv
tools/uv/uv pip install --python .venv/bin/python --no-index --find-links wheelhouse -r requirements.txt
```

已有同版本 `.venv` 时直接复用。联网安装使用以下两条命令；模型加载仍按前面的环境变量使用本地快照。

```bash
tools/uv/uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.1+cpu'
tools/uv/uv pip install --python .venv/bin/python -r requirements.txt
```

### 4. 核对资产

权重、Runtime 和模型包就位后，在仓库根目录检查文件大小、SHA256 和配置文件：

```bash
tools/uv/uv run --offline --no-project --python .venv/bin/python python - <<'PY'
import hashlib
import json
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

manifest = json.loads(Path('delivery/fp16/model_manifest.json').read_text(encoding='utf-8-sig'))
for item in manifest['entries']:
    path = Path(item['path'])
    assert path.is_file(), f'缺少文件: {path}'
    assert path.stat().st_size == item['bytes'], f'大小不符: {path}'
    assert sha256(path) == item['sha256'], f'SHA256不符: {path}'
model = Path(manifest['checkpoint']['local_directory'])
assert sha256(model / 'model.safetensors') == manifest['checkpoint']['weight_sha256']
for name in ('rl_agent_config.json', 'encoder/config.json',
             'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json'):
    assert (model / name).is_file(), f'缺少模型配置: {name}'
print('13项交付资产及原权重校验通过，模型配置齐全')
PY
```

## 推理用法

### 六域中文短指令

```bash
tools/uv/uv run --offline --no-project --python .venv/bin/python examples/predict_fp16.py '明天早上七点叫我起床'
tools/uv/uv run --offline --no-project --python .venv/bin/python examples/predict_fp16.py '导航到最近的地铁站'
```

示例使用 `split_fp16`、flags=0、8个 CPU 线程，输出完整 JSON。六个候选为日历安排、闹钟计时、音量控制、音乐点播、天气查询和交通出行。首次调用会导入 Python 库、加载 CPU 权重并初始化当前桶的 RKNN 上下文；后续调用复用已加载的模型。

### 自定义候选与批量输入

在仓库根目录执行以下例子，按业务需要修改候选名称和描述：

```bash
tools/uv/uv run --offline --no-project --python .venv/bin/python python - <<'PY'
import json
import sys
sys.path.insert(0, 'bucket-study')
import torch
from laya import Agent
from api import Buckets

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
questions = {'intent': {
    'type': 'choice',
    'instructions': '根据当前请求选择要调用的工具',
    'criteria': {'设置闹钟': '创建闹钟或倒计时',
                 '查询天气': '查询天气或气温',
                 '其他': '不属于上述功能的请求'},
}}
agent = Agent('models/laya-multilingual', device='cpu', compile=False, fast=False)
backend = Buckets(agent, 'split_fp16', flags=0)
try:
    result = agent.predict('明天早上七点叫我起床', questions)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('本次问题行的路由:', backend.last)
    results = agent.predict_batch(['明天会下雨吗', '设置十分钟倒计时'], questions, batch_size=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
finally:
    backend.close()
PY
```

`state` 接受文本、JSON字典或对话列表。`questions` 的键是问题ID，`choice` 的 `criteria` 使用候选名称→描述字典。接口还支持 `score`（有序等级）和 `noul`（真值概率），三类短输入的 schema 检查均已通过。分类评测使用中文短指令和带历史材料的长输入。

`predict_batch` 按输入顺序返回结果，适配器逐行调用 NPU。`backend.last` 单独记录最近一次 forward 的token数和桶选择。长度预算包含问题、候选、state 和特殊token，以编码后的token数计算；超过1024 token或某问题超过6个候选时抛出 `ValueError`。

### 读取输出与释放资源

对于上述 `intent` 问题，可读取：

```python
answer = result['answers']['intent']
label = answer['choice']
probabilities = answer['probabilities']
act_probability = answer['action']['act_probability']
usage = result['usage']
```

返回值还包含原API的 confidence 等字段。示例将完整结果打印为 JSON；原始 logits 由模型层产生，再解码为 API 中的概率和答案。

常驻程序复用同一个 `Agent` 和 `Buckets`，同一实例的调用串行执行。三个桶按需加载；结束时调用 `backend.close()`，释放 RKNN 上下文并恢复原CPU forward。

## 本地实验与来源

`.gitignore` 按交付文件白名单管理上传内容，新增源码或文档时将对应路径加入白名单。本地保留原始实验、INT8/128历史、SDK、数据集和协作记录。

Laya 0.3.22 源码采用 Apache-2.0，模型许可见对应模型卡。  
上游来源：[Laya](https://huggingface.co/convaiinnovations/laya)、[multilingual checkpoint](https://huggingface.co/convaiinnovations/laya-multilingual)。


## FP16 交付与 INT8 实验

本次交付使用三图 FP16，沿用原权重、CPU 前后处理和完整 API 输出。

我们也在 RKNN Toolkit 2.3.2 上测试了局部 INT8：将最后一个决策层 FFN 的两个卷积量化，其余主要计算保留 FP16。该方案分为四张图，包含额外的精度转换。198条评测的分类结果与四图 FP16 基线一致；96、256、1024长度的完整 API P50 分别改善约0.71%、0.93%、4.35%，短桶的 P95 变化方向不一致。

编码主干的 INT8 和混合精度实验记录了明显的类别漂移或非有限中间输出。这些结果与局部量化数据一并保存在本地实验记录中。
