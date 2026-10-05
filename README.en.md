# Laya-rknn

[简体中文](README.md) | **English**

[![GitHub stars](https://img.shields.io/github/stars/Mirrorium227/Laya-rknn?style=flat)](https://github.com/Mirrorium227/Laya-rknn/stargazers) [![GitHub issues](https://img.shields.io/github/issues/Mirrorium227/Laya-rknn)](https://github.com/Mirrorium227/Laya-rknn/issues) [![Platform](https://img.shields.io/badge/platform-RK3576-blue)](https://github.com/Mirrorium227/Laya-rknn) [![Precision](https://img.shields.io/badge/precision-FP16-green)](#rk3576-benchmark-results) [![RKNN](https://img.shields.io/badge/RKNN-2.3.2-orange)](https://github.com/airockchip/rknn-toolkit2)

Run Laya multilingual on RK3576 with RKNN acceleration. The adapter provides fixed sequence lengths of 96, 256, and 1024, with three FP16 graphs per length and the complete Laya API output.

Repository: [Mirrorium227/Laya-rknn](https://github.com/Mirrorium227/Laya-rknn). Send questions and suggestions through [Issues](https://github.com/Mirrorium227/Laya-rknn/issues).

## RK3576 benchmark results

Measured on October 3, 2026, using warmed-up `Agent.predict` calls. Latency includes tokenization, CPU preprocessing and postprocessing, RKNN computation, and transfers between graphs. The 96-token bucket uses 30 inputs with three runs each; the 256- and 1024-token buckets use 12 inputs with three runs each. Operator profiling runs in a separate process.

| Length | Original CPU P50/P95 ms | Three-graph FP16 P50/P95 ms | P50 speedup |
|---|---:|---:|---:|
| 96 | 844.39 / 950.37 | 102.09 / 103.87 | 8.27× |
| 256 | 1994.87 / 2452.93 | 290.38 / 363.26 | 6.87× |
| 1024 | 8640.98 / 11542.18 | 2809.23 / 3094.45 | 3.08× |

All 36 independent calibration inputs produce finite outputs and the same classes as the original CPU model. Evaluation covers 54 primary inputs and 144 additional MASSIVE inputs, for a total of 198. One input in the 96-token bucket changes class under FP16, a flip rate of 0.51%. Both CPU and FP16 score 13/18 on the existing manually labeled fixtures. The MASSIVE subset maps 17 intents to six functional domains; the new mappings and long-input labels are pending human review.

| Primary evaluation length | Class flips | Maximum absolute probability error | Logits max / MAE / RMSE |
|---|---:|---:|---:|
| 96 | 1/30 | 0.0185 | 0.08629 / 0.01860 / 0.02445 |
| 256 | 0/12 | 0.0205 | 0.19372 / 0.04126 / 0.06368 |
| 1024 | 0/12 | 0.0711 | 0.52185 / 0.12671 / 0.17486 |

Each bucket's device profile records 531 NPU nodes and 26 CPU nodes. The NPU dtype breakdown is 529 F16, one FLOAT, and one INT8 mask Where.

A single loading and routing check with the complete CPU Agent and all three buckets resident measured about 3.43 GiB RSS and 3.62 GiB HWM. Transfer volume between graphs grows with the square of sequence length L. Per-input P50 speedups for the tested 1024-token bucket range from 1.16× to 4.28×. The action head's probabilities cluster around 1 on the current samples.

## Model and computation

Laya 0.3.22 is an open-source decision model with a Jev-compatible API. This project uses [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual), pinned to revision `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`. The original weight file has SHA256 `9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`.

The network consists of a 256000×768 vocabulary table, a 22-layer mmBERT encoder, type embeddings, two decision Transformer layers, a shared scorer, and an action head. Questions, candidates, and state form a single sequence. The scorer computes logits at candidate markers, and the CPU decodes them into probabilities and answers. The action head receives the CLS vector and four statistical features.

| Location | Computation |
|---|---|
| CPU | Tokenization, vocabulary lookup, two floating-point NaN guards, head key bias, original action head, temperature/probability processing, and API decoding |
| RKNN | Embedding normalization, 22 encoder layers, type embeddings, two decision layers, and scorer; some graph operators execute on the CPU |

The two attention NaN guards define the graph boundaries. The CPU clears NaNs, retains ±Inf, and sends probabilities, V, and the residual to the next graph. The main floating-point computation uses FP16; masks and indices use their corresponding integer types.

The adapter keeps `Agent.predict(state, questions)` and its answers, probabilities, action, usage, and other fields. Each NPU call processes one row; batches run row by row. Each question supports up to six candidates. The total encoded length of the question, candidates, state, and special tokens selects the smallest fitting bucket: 96, 256, or 1024. Inputs above 1024 tokens or six candidates raise `ValueError`. Question and candidate encoding budgets follow the upstream rules.

## Repository and model files

The repository contains inference code, dependency instructions, shape manifests for all three buckets, and the [asset checksum manifest](delivery/fp16/model_manifest.json). Clone the source on your computer:

```bash
git clone https://github.com/Mirrorium227/Laya-rknn.git
cd Laya-rknn
```

Download the FP16 model bundle separately from [Google Drive](https://drive.google.com/drive/folders/1zbYvTjf-k_drJArINAAcPtpKbAms7apF?usp=sharing) or [123Pan](https://1815656308.share.123pan.cn/123pan/kEBzVv-RS1tA).

The bundle contains nine RKNN files, three shape manifests, `delivery/fp16/model_manifest.json`, bundle notes, and a `.7z` file. Deployment also uses the complete Laya checkpoint, Rockchip Runtime, uv, and Python dependencies, prepared in the steps below. The checksum manifest records SHA256 values for both the model files and Runtime.

Download and extract the model bundle on your computer, then merge its `bucket-study` and `delivery` directories into the repository root. This Linux example uses `fp16_assets` as the extraction directory:

```bash
# Run from the repository root after extracting the downloaded bundle into fp16_assets.
ASSET_ROOT="$PWD/fp16_assets"
# If the archive has an outer directory, set ASSET_ROOT to the directory containing bucket-study and delivery.
cp -a "$ASSET_ROOT/bucket-study/." ./bucket-study/
cp -a "$ASSET_ROOT/delivery/." ./delivery/
```

The deployed layout is:

```text
models/laya-multilingual/                       # Complete checkpoint at the pinned revision
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

`part{0,1,2}` means three separate directories. The shape manifests are at `bucket-study/split/l{96,256,1024}/manifest.json`.

Runtime comes from SDK 2.3.2 at `rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so`, with SHA256 `d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8`. The code loads this library through an absolute path inside the repository and checks its version and `/proc/self/maps`. The tested Runtime version is `2.3.2 (429f97ae6b@2025-04-09T09:09:27)`, with driver 0.9.8.

## Initial setup

The tested environment is Linux ARM64 with Python 3.11.2. Dependencies are listed in [requirements.txt](requirements.txt). These examples use `/home/radxa/Laya_rknn` as the board's repository directory; the Runtime path follows the repository location.

### 1. Prepare the complete Laya checkpoint

On an internet-connected computer with uv installed, run this from the repository root to download the pinned checkpoint. Then upload `models/laya-multilingual` to the board's repository root.

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

Offline loading uses the checkpoint's weights, `encoder` configuration, and `tokenizer` files. The original weight SHA256 is listed above.

### 2. Prepare Runtime, uv, and Python dependencies

- Place the ARM64 `librknnrt.so` from RKNN Toolkit 2.3.2 in `rknn-2.3.2/lib/`. The SDK source is [Rockchip RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2); check the library against the SHA256 above.
- Place the Linux ARM64 uv executable at `tools/uv/uv`. See the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
- Use `/usr/bin/python3.11` on the board. For offline installation, prepare a `wheelhouse` for Linux ARM64 / CPython 3.11 containing `requirements.txt` and all transitive dependencies, including `torch==2.9.1+cpu`.

Inference calls the project's Runtime through the C API. The Python environment handles tokenization, CPU computation, and API output.

### 3. Create the environment and install dependencies

From the repository root, set the environment variables, create the virtual environment, and install dependencies. Set these variables in each new terminal so caches, bytecode, and temporary files stay inside the project.

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

Reuse an existing `.venv` with the same dependency versions. For online installation, use these two commands; model loading uses the local checkpoint configured by the environment variables above.

```bash
tools/uv/uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.1+cpu'
tools/uv/uv pip install --python .venv/bin/python -r requirements.txt
```

### 4. Verify the assets

With the checkpoint, Runtime, and bundle in place, run this from the repository root to check file sizes, SHA256 values, and configuration files:

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
    assert path.is_file(), f'Missing file: {path}'
    assert path.stat().st_size == item['bytes'], f'Size mismatch: {path}'
    assert sha256(path) == item['sha256'], f'SHA256 mismatch: {path}'
model = Path(manifest['checkpoint']['local_directory'])
assert sha256(model / 'model.safetensors') == manifest['checkpoint']['weight_sha256']
for name in ('rl_agent_config.json', 'encoder/config.json',
             'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json'):
    assert (model / name).is_file(), f'Missing model configuration: {name}'
print('13 assets and the original weights verified; model configuration is complete')
PY
```

## Inference

### Chinese commands across six domains

```bash
tools/uv/uv run --offline --no-project --python .venv/bin/python examples/predict_fp16.py '明天早上七点叫我起床'
tools/uv/uv run --offline --no-project --python .venv/bin/python examples/predict_fp16.py '导航到最近的地铁站'
```

The example uses `split_fp16`, flags=0, and eight CPU threads, and prints the complete JSON result. Its six candidate domains are calendar, alarms/timers, volume, music, weather, and navigation. The first call imports Python libraries, loads the CPU checkpoint, and initializes the selected bucket's RKNN contexts. Later calls reuse the loaded models.

### Custom candidates and batch inputs

Run this from the repository root and edit the candidate names and descriptions to suit your task. The sample inputs and candidates are in Chinese, matching the intent-classification examples above.

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
    print('Routing for this forward:', backend.last)
    results = agent.predict_batch(['明天会下雨吗', '设置十分钟倒计时'], questions, batch_size=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
finally:
    backend.close()
PY
```

`state` accepts text, a JSON dictionary, or a conversation list. Keys in `questions` are question IDs; `choice.criteria` maps candidate names to descriptions. The API also supports `score` for ordered levels and `noul` for truth probabilities. Short-input schema checks pass for all three types. Classification evaluation uses Chinese commands and longer inputs with historical context.

`predict_batch` returns results in input order and calls the NPU row by row. `backend.last` separately records the token count and bucket selection from the most recent forward. The budget includes question, candidates, state, and special tokens, measured after encoding. Inputs above 1024 tokens or six candidates per question raise `ValueError`.

### Read the output and release resources

For the `intent` question above:

```python
answer = result['answers']['intent']
label = answer['choice']
probabilities = answer['probabilities']
act_probability = answer['action']['act_probability']
usage = result['usage']
```

The result also contains the original API's confidence and other fields. The example prints the complete result as JSON. Raw logits come from the model and are decoded into the API's probabilities and answers.

For a long-running process, reuse one `Agent` and `Buckets` instance and serialize calls to it. Buckets load on demand. Call `backend.close()` at shutdown to release RKNN contexts and restore the original CPU forward.

## Local experiments and sources

`.gitignore` uses an explicit delivery-file allowlist. Add new source or documentation paths to that list. The local workspace retains the original experiments, INT8 and 128-token history, SDK, datasets, and collaboration records.

Laya 0.3.22 source uses Apache-2.0. Model licenses are listed in the corresponding model cards.
Upstream sources: [Laya](https://huggingface.co/convaiinnovations/laya), [multilingual checkpoint](https://huggingface.co/convaiinnovations/laya-multilingual).

## FP16 delivery and INT8 experiments

This release uses three FP16 graphs, the original weights, CPU preprocessing and postprocessing, and the complete API output.

We also tested partial INT8 with RKNN Toolkit 2.3.2, quantizing the two FFN convolutions in the last decision layer while keeping the remaining main computation in FP16. This path uses four graphs and additional precision conversions. Its classifications match the four-graph FP16 baseline on all 198 evaluation inputs. Full-API P50 improves by about 0.71%, 0.93%, and 4.35% at lengths 96, 256, and 1024; P95 changes in different directions for the short buckets.

Encoder INT8 and mixed-precision experiments recorded substantial class drift or non-finite intermediate outputs. These results and the partial-quantization measurements are retained in the local experiment records.
