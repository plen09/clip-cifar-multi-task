# CLIP × CIFAR-100：多任务损失加权、扰动与梯度冲突实验

>
> 本项目包含三条可独立复现实验线：
>
> 1. **Clean Baseline**：Fine（100 类）+ Coarse（20 类），复现 Equal / Uncertainty Weighting / GradNorm 的损失平衡行为。
> 2. **Perturbation**：在已缓存的 CLIP 特征或训练标签上添加受控扰动，检验方法稳定性。
> 3. **Heterogeneous MTL（推荐主实验）**：Fine（100 类）+ Rotation（4 类），测量共享梯度余弦、冲突比例，并比较 **PCGrad**；另有 Fine + Color 的廉价流程对照。
>
> **定位：** 这是多任务学习算法在冻结预训练图像特征上的**方法迁移验证**，**不是**对 Kendall / GradNorm / PCGrad 原论文数据表的严格数值复现。所有结果以你实际运行的输出为准；本项目不附带 CIFAR-100、预训练权重或伪造的实测精度。

---

## 1. 先看这个：推荐运行路线

**如果你已经完成旧版 `Fine + Coarse` 实验、且有 `data/features/train.pt` 和 `test.pt`：** 不必重新提取干净特征。从下面的步骤开始。

所有命令均在 `clip_cifar100_mtl/` **项目根目录**执行。

### 第一步：验证环境和已有代码

```bash
python check_env.py --test-clip --model-name ViT-B-32
python -m pytest -q tests
```

> `ViT-B-32` 只是示例。如果你的本地权重对应 QuickGELU，则所有加载/特征提取命令都改成 `ViT-B-32-quickgelu`。**依据下载来源的模型卡确定，不能凭 `.safetensors` 文件名猜测。**

### 第二步：先跑无需 CLIP 推理的 Color 对照

```bash
python prepare_color_features.py
python run_hetero.py --scenario color --quick
```

它只复用已有的干净 CLIP 特征，并从本地 CIFAR-100 像素计算颜色标签；**不需要再运行 ViT-B/32 推理**。

### 第三步：准备正式 Rotation 特征

```bash
python prepare_rotation_features.py --batch-size 64 --model-name ViT-B-32
```

这一步在原始图像上产生 0° / 90° / 180° / 270° 四种视角，再由**同一个冻结 CLIP 编码器**分别抽取特征。匹配的干净 0° 缓存可以复用。

### 第四步：快速检查训练闭环

```bash
python run_hetero.py --scenario rotation --quick
```

`--quick` 只运行 **1 个 seed × 2 epochs**，用于检查代码/设备/输出，**不是正式实验**。

### 第五步：正式运行，推荐优先做这一条

```bash
python run_hetero.py \
  --scenario rotation \
  --seeds 0 1 2 \
  --epochs 30 \
  --batch-size 256 \
  --diagnose-every 1 \
  --skip-existing
```

此命令默认比较：

`fine_only`、`aux_only`、`equal`、`uncertainty`、`gradnorm`、`pcgrad`。

输出分别写入 `runs/rotation/` 与 `results/rotation/`，不会与旧版 `runs/`、`results/` 混在一起。

**Windows PowerShell：** 上面的 `\` 续行符是 Bash 语法；在 PowerShell 中请写成**一行**：

```powershell
python run_hetero.py --scenario rotation --seeds 0 1 2 --epochs 30 --batch-size 256 --diagnose-every 1 --skip-existing
```

---

## 2. 实验设计与研究问题

### 2.1 为什么旧版 Fine + Coarse 的差距可能很小？

CIFAR-100 官方细类 Fine 与粗类 Coarse 有直接层级关系。对同一图像，粗类标签由细类决定，两个任务高度相关。冻结 CLIP 的语义表征后，只训练小型共享 MLP，损失加权对表示学习的可影响空间也受到限制。

因此，之前的 Clean Baseline 是**算法正确运行与低成本对照**，而不是能充分体现任务梯度竞争的强冲突设置。即使没有观察到明显的准确率提升，也不等于 GradNorm 无效。

### 2.2 主实验：Fine + Rotation

```text
本地 CIFAR-100 图像 (32×32)
        |
        +-- 在原始像素上旋转 0°/90°/180°/270°
        |
        v
OpenCLIP ViT-B/32 + 对应 eval 预处理
    [冻结；不更新 CLIP 参数]
        |
    512 维归一化特征（FP16 缓存）
        |
        v
共享、可训练 MLP:
Linear(512,256) -> GELU -> Linear(256,256) -> GELU
        |
        +--------------------+
        |                    |
    Fine Head             Rotation Head
    Linear(256,100)       Linear(256,4)
        |                    |
    Fine CE               Rotation CE
        +--------+-----------+
                 |
 Equal / Fixed / Uncertainty / GradNorm / PCGrad
                 |
   更新共享 MLP（及对应任务头）
```

- **Fine（100 类）**：判断图像中的物体细类别。
- **Rotation（4 类）**：判断当前图像相对于原图逆时针旋转的角度：`0、90、180、270`。
- 两个头接收的是**同一张已旋转图像产生的同一特征**，共享 MLP 参与两个任务的梯度更新。
- Rotation 的四个输入特征来自**像素旋转后重新编码**，并非对同一个静态特征硬贴角度标签。
- 每张训练图像每个 epoch 仅抽取**一个角度**，角度与样本编号、seed、epoch 确定性关联；每连续四轮覆盖四个角度。相同 seed 下，各方法得到相同的旋转分配。
- Fine 与 Rotation 的监督目标**有潜在梯度竞争**，但**不同任务不等于必然发生负余弦冲突**；必须测量。

### 2.3 数据划分与评估协议

| 集合 | 图像数量 | 来源和用途 |
|---|---:|---|
| Train | 45,000 | 官方 CIFAR-100 train；Fine 类分层划分，用于梯度更新 |
| Validation | 5,000 | 同一官方 train 的剩余部分；选择最佳 checkpoint |
| Test | 10,000 | 官方 test；选择 checkpoint 后才评估 |

默认 `--split-seed 2026` 固定 Train/Validation 划分，与训练 seed `0 1 2` 区分开。

Rotation 场景的评估有意采用**不同但固定的视角协议**：

- **Fine Accuracy**：Validation/Test 只在 **0°** 视角评估，分母分别是 5,000 / 10,000。
- **Rotation Accuracy**：Validation/Test 的每张图都评估 **四个角度**，预测数量分别是 20,000 / 40,000。
- **多任务 checkpoint 选择**：`(val_fine_acc + val_aux_acc) / 2`；单任务只依据自己的验证准确率。
- 训练时的 Fine 输入包含多种角度；**不能**把新实验的 Fine-only 分数与旧版只用原始 0° 训练的 Fine-only 作严格的因果对比。
- Test 不是训练/调参依据；当前脚本在各个训练 run 选好最佳验证 checkpoint 后执行一次最终测试。

### 2.4 Color 低成本对照（不是主实验替代品）

`Fine + Color` 复用原始 0° CLIP 特征，计算每张原始图像的 `mean(R) - mean(B)`，按**45k 训练子集**的三个四分位阈值分成四类。验证/测试不参与拟合分位阈值。

Color 任务旨在检查**异质标签 + 梯度诊断代码**能否低成本运行。Color 与 Rotation 任务定义不同，**不应直接比较两者的辅助任务准确率来宣称算法优劣**。

---

## 3. 数据、权重与项目目录

### 3.1 必需的本地文件

你已经下载的资源应放在：

```text
clip_cifar100_mtl/
├── data/
│   ├── cifar-100-python.tar.gz         # 官方 CIFAR-100 Python 压缩包
│   ├── cifar-100-python/                # 自动离线解压；首次可不存在
│   │   ├── train
│   │   ├── test
│   │   └── meta
│   ├── features/                        # 旧版干净特征，可复用
│   │   ├── train.pt
│   │   ├── test.pt
│   │   └── cache_info.json
│   ├── hetero_rotation/                 # 新：四角度特征缓存
│   └── hetero_color/                    # 新：Color 标签对照缓存
├── weights/
│   └── open_clip_model.safetensors     # 你自己下载的本地 checkpoint
├── src/
│   ├── balancing.py                    # Uncertainty / GradNorm
│   ├── data.py                         # CIFAR 标签、划分、特征读入
│   ├── hetero.py                       # 新 MLP、梯度余弦、PCGrad
│   ├── model.py                        # 旧 Fine + Coarse MLP
│   ├── offline_assets.py               # 离线解压、checkpoint 检查
│   ├── perturb.py                      # 标签/特征扰动、少样本
│   └── utils.py
├── check_env.py
├── prepare_features.py                 # 旧版 0° 特征
├── prepare_rotation_features.py        # 新：4-way Rotation 特征
├── prepare_color_features.py           # 新：Color 对照特征/标签
├── train.py                            # 旧 Fine + Coarse 单次训练
├── run_all.py                          # 旧 Fine + Coarse 批量训练
├── train_hetero.py                     # 新异质任务单次训练
├── run_hetero.py                       # 新异质任务批量训练
├── summarize_hetero.py                 # 新结果汇总
├── plot_hetero.py                      # 新诊断绘图
├── run_perturbations.py                # 旧特征/标签扰动批量实验
├── compare_perturbations.py
├── summarize.py
├── plot_results.py
├── smoke_test.py
├── tests/
├── analysis/                           # 事后诊断脚本
├── docs/guides/                        # 复现与专题指南
├── docs/mtl_exploratory_paper/           # 中文论文源码与图像
├── requirements.txt
└── README.md                            # 本文件
```

`data/`、`weights/`、`runs/`、`results/` 是重要本地资产。**升级代码时不要删除或覆盖这些目录。** 发布的 ZIP 不包含大型特征、图片、预训练权重或训练结果。

### 3.2 CIFAR-100 的离线读取

`prepare_features.py`、`prepare_rotation_features.py` 等会调用 `ensure_offline_cifar100`：

1. 检查 `data/cifar-100-python/train`、`test`、`meta` 是否存在。
2. 如果不存在，读取本地 `data/cifar-100-python.tar.gz`，核对官方归档 MD5 后安全解压。
3. **不会主动联网下载 CIFAR-100。**

如果已正确解压，就不必再次解压。

### 3.3 Safetensors 格式、模型来源、QuickGELU

这是本项目最容易影响研究结论的地方：

- `open_clip_model.safetensors` 这个文件名**不能证明**模型来自 OpenAI 官方预训练。
- 即使 ViT-B/32 权重维度吻合，也**无法**仅凭权重形状确认训练语料、预处理、GELU/QuickGELU 激活。
- 当前 `src/offline_assets.py` 按 **OpenCLIP 格式**验证关键视觉张量形状，并调用 OpenCLIP 的本地 `pretrained=<本地路径>` 加载；它不保证任意 Hugging Face Transformers 格式权重都能读取。
- 当前项目支持 `--model-name ViT-B-32` 与 `--model-name ViT-B-32-quickgelu`；OpenAI 原始 CLIP 的 ViT-B/32 使用 QuickGELU。**用模型下载页或模型卡确定模型名及处理配置。**
- OpenCLIP 内置 eval 预处理被用于提取特征；代码**不会**自动核实它是否与来源模型卡的全部预处理细节一致。

本项目会记录 checkpoint **SHA256**，在四角度特征复用 0° 缓存时检查**权重 SHA256、模型名、是否归一化、样本标签顺序**；这些校验有助于避免混用，但**不是预处理配置的独立认证**。

若无法验证来源，论文中建议写「**OpenCLIP ViT-B/32（local checkpoint; pretraining origin unverified）**」，不要直接声称「OpenAI 官方 ViT-B/32」。

---

## 4. RTX 5070 环境安装

推荐 Python **3.10–3.12** 和支持 **RTX 5070 / Compute Capability 12.0 (sm_120)** 的 CUDA 版 PyTorch。具体 torch/torchvision 版本应采用**当前 PyTorch 官方安装页面给出的配套组合**，不要把旧命令里某个版本号当成永远有效的唯一解法：

- 官方安装向导：<https://pytorch.org/get-started/locally/>
- 可选历史 wheel：<https://pytorch.org/get-started/previous-versions/>

### 4.1 创建虚拟环境

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux / macOS（本项目 CUDA 主要面向 Linux / Windows）**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4.2 安装 PyTorch 与项目依赖

先按 PyTorch 官网的操作系统、Python 和 CUDA 选项生成命令。**示例：当官方仍提供该 CUDA 12.8 wheel 组合时**可用：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

再安装：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 主要涉及 `numpy`、`Pillow`、`tqdm`、`matplotlib`、`open_clip_torch`、`safetensors`、`pytest`。需要先装**成套兼容**的 torch/torchvision；本项目**不需要**安装 `git+https://github.com/openai/CLIP.git`。

> **离线资源 ≠ 离线安装依赖。** 本项目运行时读取本地 CIFAR 和本地模型，不自动下载它们；首次安装 Python 依赖仍需你事先准备 wheel 或联网安装。

### 4.3 检查 GPU 和本地权重

```bash
nvidia-smi
python check_env.py
python check_env.py --test-clip --weights weights/open_clip_model.safetensors --model-name ViT-B-32
python -m pytest -q tests
```

检查输出包含：

- `torch.cuda.is_available()` 对应 CUDA 可用；
- GPU 名称接近 `NVIDIA GeForce RTX 5070`；
- Compute Capability 为 `12.0`，且测试用 CUDA 矩阵乘法成功；
- 本地 CLIP `encode_image` 输出为 `(1, 512)`；
- 单元测试通过。

已核对的本版本项目测试：**22 项单元测试在 CPU 环境通过**。这仅证明代码测试通过，**不代表真实 CIFAR-100 和你的 RTX 5070 上已经获得新实验成绩**。

---

## 5. 离线验证与基础特征缓存

如果你还没有 `data/features/train.pt` 和 `test.pt`，先提取**原图 0°**特征：

```bash
python prepare_features.py \
  --cifar-root data \
  --cifar-archive data/cifar-100-python.tar.gz \
  --weights weights/open_clip_model.safetensors \
  --model-name ViT-B-32 \
  --batch-size 64
```

Windows PowerShell 请改用单行：

```powershell
python prepare_features.py --cifar-root data --cifar-archive data/cifar-100-python.tar.gz --weights weights/open_clip_model.safetensors --model-name ViT-B-32 --batch-size 64
```

产物：

```text
data/features/train.pt         # 50,000 × 512
data/features/test.pt          # 10,000 × 512
data/features/cache_info.json
```

默认 CLIP 输出做 L2 normalize 后，以 FP16 缓存；全部 60,000 张原图的纯 FP16 特征约 **61.4 MB**（不含少量标签/元数据）。

**不必反复提取：** 如果已存在这两个缓存，脚本默认不改动它们。若**确认**换过权重、模型名或预处理，请先备份旧缓存与实验结果，再用 `--force` 重新提取；新旧实验的 runs/results 必须放在不同目录中。

如果 Windows 的数据加载卡住或显存不足，可尝试：

```bash
python prepare_features.py --batch-size 32 --num-workers 0
```

注意：只改变 `--model-name` 而不 `--force` 时，原 `prepare_features.py` 可能直接跳过已存在的缓存；**不要把“跳过提取”误认为缓存已自动变成新模型的结果**。

---

## 6. 主实验：Fine + Rotation

### 6.1 提取四角度特征（只需在配置固定后做一次）

```bash
python prepare_rotation_features.py \
  --weights weights/open_clip_model.safetensors \
  --model-name ViT-B-32 \
  --batch-size 64 \
  --num-workers 2
```

如果模型卡确认 QuickGELU：

```bash
python prepare_rotation_features.py --model-name ViT-B-32-quickgelu --batch-size 64
```

如果现有 `data/features/` 的 checkpoint SHA256、模型名或归一化方式与当前配置不符，会报错阻止复用 0°。当你确定需要**重新计算所有角度**时：

```bash
python prepare_rotation_features.py --no-reuse-clean --batch-size 32 --num-workers 0
```

要替换**已经存在的** `data/hetero_rotation/` 缓存，才额外添加 `--force`。请先备份原缓存；`--no-reuse-clean` 不等于 `--force`。

生成：

```text
data/hetero_rotation/train.pt        # 50,000 × 4 × 512
data/hetero_rotation/test.pt         # 10,000 × 4 × 512
data/hetero_rotation/cache_info.json
```

四角度 FP16 特征理论大小约 **245.8 MB**（十进制 MB）；实际 `.pt` 文件略大。已有匹配 0° 缓存时，只需要重新编码其余三个角度。**冻结 CLIP 特征提取是这套实验中最主要的 GPU 开销**；后续多任务 MLP 训练不再运行 CLIP。

### 6.2 Quick smoke test

```bash
python run_hetero.py --scenario rotation --quick
```

输出在独立的：

```text
runs/rotation_quick/
results/rotation_quick/
```

### 6.3 正式实验（6 方法 × 3 seeds）

```bash
python run_hetero.py --scenario rotation --seeds 0 1 2 --epochs 30 --batch-size 256 --diagnose-every 1 --skip-existing
```

建议优先看：

```text
results/rotation/summary.csv
results/rotation/paired_differences.csv
results/rotation/figures/gradient_cosine.png
results/rotation/figures/gradient_conflict_rate.png
results/rotation/figures/raw_grad_fine.png
results/rotation/figures/raw_grad_aux.png
```

### 6.4 优先补充的受控消融实验

**A. 只用 20% 训练数据：**

```bash
python run_hetero.py --scenario rotation --train-fraction 0.2 --seeds 0 1 2 --epochs 30 --diagnose-every 1 --runs-dir runs/rotation_20pct --results-dir results/rotation_20pct --skip-existing
```

**B. 固定损失权重，用于检验“动态调度”是否真的有额外收益：**

```bash
python run_hetero.py --scenario rotation --methods fixed --fixed-fine 0.75 --fixed-aux 1.25 --seeds 0 1 2 --epochs 30 --runs-dir runs/rotation_fixed_075_125 --results-dir results/rotation_fixed_075_125
```

注意 `fixed` 不在默认六种方法里；需要显式写 `--methods fixed`。**不同固定权重组合必须使用不同的输出目录**，不能因为 `seed` 相同就复用既有 `results.json`。

**C. GradNorm 的 α 对比：**

```bash
python run_hetero.py --scenario rotation --methods gradnorm --alpha 0.5 --seeds 0 1 2 --epochs 30 --runs-dir runs/rotation_alpha05 --results-dir results/rotation_alpha05
```

建议每次只改变一个变量，其它设置、同 seed 的旋转分配和验证协议保持不变。

---

## 7. 低成本对照：Fine + Color

**前提：** 已有干净缓存 `data/features/train.pt` 和 `test.pt`，以及本地 CIFAR 原始像素文件。

```bash
python prepare_color_features.py
python run_hetero.py --scenario color --quick
python run_hetero.py --scenario color --seeds 0 1 2 --epochs 30 --diagnose-every 1 --skip-existing
```

对应独立目录：

```text
data/hetero_color/
runs/color/
results/color/
```

标签来自原始图像 RGB 色差的四分位组，不是训练集外部标签；训练集划分 seed 默认 `2026`，并记录在缓存元数据中。`run_hetero.py` 会检查 Color 阈值的 split seed 是否和训练选项匹配。

**解释边界：** Color4 是人工构造的代理任务；就算它能分出 4 类，也不意味着它与 Fine 产生强方向冲突。优先以 `Equal` 的共享梯度余弦/冲突比例进行判断。

---

## 8. 保留的 Fine + Coarse 基线

这部分延续旧版流程，**不使用 Rotation/Color 缓存**：

```bash
# 已有 data/features/train.pt、test.pt 后：
python run_all.py --quick

# 正式运行
python run_all.py --seeds 0 1 2 --epochs 30 --batch-size 256 --skip-existing
```

对比方法：

- `fine_only`：只优化 100 类 Fine。
- `coarse_only`：只优化官方 20 类 Coarse。
- `equal`：两个 CE 等权。
- `uncertainty`：Kendall 分类近似损失。
- `gradnorm`：动态梯度范数平衡。
- `fixed`：额外自定义固定权重，可手动加入 `--methods fixed`。

输出：

```text
runs/<method>/seed_<n>/metrics.csv
runs/<method>/seed_<n>/best.pt
runs/<method>/seed_<n>/results.json
results/clean/summary.csv
results/clean/summary.json
results/clean/figures/*.png
```

**注意：** 这条旧实验使用 `src/model.py` 的旧共享 MLP；异质实验则使用 `src/hetero.py` 的**两层共享 GELU MLP**。对比 Clean 与 Hetero 的原始绝对准确率时，任务、网络与训练视角不完全相同，不能简单把差值归因为“梯度冲突更强”。

---

## 9. 保留的扰动实验

扰动流程针对 **Fine + Coarse 的旧版干净缓存**，不是自动对 Rotation 图像加噪声。

```bash
python run_perturbations.py --quick
python run_perturbations.py --skip-existing
```

默认四个推荐场景：

| 场景名称 | 扰动 | 主要诊断问题 |
|---|---|---|
| `gaussian07` | 训练阶段对 512-d 特征加入相对 L2 强度约 0.7 的高斯噪声 | 特征质量下降时是否稳定 |
| `coarse20` | 20% 训练样本的 Coarse 标签替换成另一个错误类别 | 粗任务监督污染 |
| `fine20` | 20% 训练样本的 Fine 标签替换成另一个错误类别 | 细任务监督污染 |
| `lowdata10` | 只使用约 10% 原训练子集 | 少样本下的任务平衡 |

其他可选场景：`gaussian03`、`dropout20`、`coarse40`、`clean`。

```bash
python run_perturbations.py --scenarios coarse20 coarse40 --seeds 0 1 2 --epochs 30 --skip-existing
python run_perturbations.py --scenarios gaussian03 gaussian07 dropout20 --skip-existing
python run_perturbations.py --dry-run
```

结果写入：

```text
runs/perturbations/<scenario>/<method>/seed_<n>/
results/perturbations/<scenario>/summary.csv
results/perturbations/<scenario>/figures/
results/perturbations/overview.csv
```

**实验控制：** 噪声只加于**训练集**，验证/测试保持干净；旧缓存不会被重写。`Gaussian` 是对 CLIP **输出向量**加噪声，**不等于**在原始图像像素上加高斯噪声。标签噪声是**错误标签替换**，不是标签平滑。

单独污染某一任务后，GradNorm 未必降低该任务权重：如果损失下降更慢，它可能被赋予更高的相对训练速度补偿目标。要对照各任务的损失、权重轨迹和测试表现，避免只看最终准确率。

更详细的扰动解释仍可参考 `docs/guides/perturbations.md`。

---

## 10. 算法与梯度诊断

设两个任务的交叉熵分别为 \(L_f\)、\(L_a\)，其中 \(a\) 表示当前实验的 Aux（Rotation / Color；旧版为 Coarse）。共享可训练参数记作 \(W\)。

### 10.1 对比方法

| `--method` | 优化内容 | 研究用途 |
|---|---|---|
| `fine_only` | 仅 \(L_f\) | Fine 单任务参考 |
| `aux_only` | 仅 \(L_a\) | Rotation/Color 单任务参考 |
| `equal` | \(L_f + L_a\) | 最重要的双任务基线 |
| `fixed` | \(w_f L_f + w_a L_a\) | 排除“只是找到更好的固定比例” |
| `uncertainty` | 联合学习任务对数方差 | Kendall 分类近似 |
| `gradnorm` | 根据共享梯度大小和相对训练速度学习正权重 | GradNorm |
| `pcgrad` | 出现梯度冲突时投影共享梯度 | 梯度方向处理 |

**Uncertainty Weighting（Kendall 分类近似）**

\[
L_{\mathrm{UW}}
= e^{-s_f}L_f+\tfrac{1}{2}s_f
+ e^{-s_a}L_a+\tfrac{1}{2}s_a,
\qquad s_i=\log\sigma_i^2.
\]

`weight_fine/aux` 是 \(e^{-s_i}\) 的**有效 CE 乘数**，不包含额外的 \(s_i/2\) 项。这里是 Kendall 的**分类近似表达式**，不是对任意温度 Softmax NLL 的精确等价变换。

**GradNorm**

\[
G_i(t)=\left\|\nabla_W\left(w_i(t)L_i(t)\right)\right\|_2,\quad
r_i(t)=\frac{L_i(t)/L_i(0)}
{\operatorname{mean}_j\left[L_j(t)/L_j(0)\right]},
\]

\[
G_i^*(t)=\overline{G}(t)\,[r_i(t)]^\alpha,\quad
L_{\mathrm{grad}}=\sum_i\left|G_i(t)-G_i^*(t)\right|.
\]

本实现使用独立的任务权重优化器；计算目标范数时执行 `detach`，网络损失使用任务权重的脱离梯度快照，更新权重后将其归一化为 `sum(weights)=2`。默认 `--alpha 1.5`、`--weight-lr 0.01`。

**PCGrad**

当两任务共享梯度 \(g_f\cdot g_a<0\) 时，计算去除相互冲突分量的投影；在本实现中，只覆盖**共享 MLP 参数**的梯度，两个任务头依然接受普通 CE 的反向梯度。双任务实现见 `src/hetero.py::pcgrad_combine`。

**最重要的区分：** GradNorm 主要处理**梯度幅度/相对训练速度**；PCGrad 处理**梯度方向冲突**。二者没有“在所有数据集必定提高 Accuracy”的保证。

### 10.2 梯度余弦与冲突率的计算

对于**加权之前**的两个任务损失：

\[
g_f=\nabla_W L_f,\quad
g_a=\nabla_W L_a,\quad
\cos(g_f,g_a)
=\frac{g_f^\top g_a}{\|g_f\|_2\|g_a\|_2}.
\]

- **`grad_cosine`**：当前 epoch 内**被诊断批次**的原始梯度余弦**平均值**；小于 0 的某批次表明该批次两个任务对共享参数存在方向冲突。
- **`grad_conflict_rate`**：被诊断批次中 `cos < 0` 的**比例**，取值 `[0,1]`，不是测试准确率下降百分比。
- **`raw_grad_norm_fine/aux`**：两个**未加权 CE**对整个共享 MLP 参数组的 L2 梯度范数，观察量级不平衡。
- **GradNorm 诊断**：`gradnorm_G_*`、`gradnorm_target_*`、`gradnorm_r_*` 分别用于看实际加权梯度、目标和相对逆训练速率。

默认 `--diagnose-every 10`：Equal/UW/GradNorm 通常每十个 batch 检查一次；PCGrad 因投影需求**每个 batch**计算梯度。**跨方法比较冲突率时建议统一 `--diagnose-every 1`**，否则采样频率不同，统计可能不完全可比。

正的**平均**余弦并不意味着从未出现负余弦；反过来，部分批次出现负余弦也不代表 PCGrad 一定能提高最终泛化。

### 10.3 关键指标与注意事项

| 指标 | 含义 | 注意 |
|---|---|---|
| `test_fine_acc_pct` | Test Fine Accuracy，单位 % | Rotation 中 Fine 只评估 0° |
| `test_aux_acc_pct` | Test Rotation/Color Accuracy，单位 % | Rotation 有四角度、40k 个预测 |
| `test_mean_acc_pct` | 两任务准确率的简单算术均值，单位 % | 只做便利汇总，不代表统一损失或最优公平性 |
| `grad_conflict_rate_all_epochs` | 全训练期间被采样批次中负余弦比例 | 取值 0–1；受诊断采样频率影响 |
| `grad_cosine_all_epochs_mean` | 被采样批次的平均余弦 | 看原始方向竞争，不是加权后余弦 |
| `final_weight_fine/aux` | 载入最佳验证 checkpoint 的最终权重 | 对照不同方法时注意权重定义 |
| `best_epoch` | 最佳验证 checkpoint 所在轮次 | 避免用测试准确率选 Epoch |

Rotation 均匀四分类的**随机猜测期望准确率**为 25%；若 `aux_only` 接近 25%，优先怀疑当前冻结特征无法有效表示方向、数据预处理削弱方向信息或训练不充分，**不要**直接归因于 GradNorm / PCGrad 失效。

---

## 11. 运行参数、续跑与配置隔离

### 11.1 `run_hetero.py` 常用参数

| 选项 | 默认值 | 作用 |
|---|---|---|
| `--scenario` | `rotation` | `rotation` 或 `color` |
| `--methods` | 六种默认方法 | 可选 `fixed`；空格分隔多个方法 |
| `--seeds` | `0 1 2` | 逐 seed 运行 |
| `--epochs` | `30` | 总训练轮次 |
| `--batch-size` | `256` | 共享 MLP 的 batch size |
| `--lr` | `1e-3` | 网络 AdamW 学习率 |
| `--alpha` | `1.5` | GradNorm 非均衡系数 |
| `--weight-lr` | `0.01` | GradNorm 权重优化器学习率 |
| `--fixed-fine` / `--fixed-aux` | `1.0 / 1.0` | `fixed` CE 权重 |
| `--diagnose-every` | `10` | 每隔多少 batch 诊断一次；PCGrad 仍每批测 |
| `--train-fraction` | `1.0` | Fine 分层保留训练集的比例 |
| `--split-seed` | `2026` | 固定 train/val 划分 |
| `--device` | `auto` | `auto`、`cuda`、`cpu` |
| `--features-dir` | `data/hetero_<scenario>` | 读取训练和测试缓存 |
| `--runs-dir` | `runs/<scenario>` | 单 run 输出 |
| `--results-dir` | `results/<scenario>` | 汇总输出 |
| `--quick` | 关闭 | 强制 1 seed / 2 epochs，并使用 quick 输出目录 |
| `--dry-run` | 关闭 | 只打印运行计划，不实际训练 |
| `--skip-existing` | 关闭 | 跳过已经存在 `results.json` 的 run |
| `--overwrite` | 关闭 | 允许覆盖对应 run 的结果 |
| `--no-plots` | 关闭 | 跳过绘图 |

先预览待运行的命令：

```bash
python run_hetero.py --scenario rotation --methods equal gradnorm pcgrad --seeds 0 1 2 --dry-run
```

`train_hetero.py` 是单次训练入口，支持额外的 `--hidden-dim`（默认 256）及 `--weight-decay`（默认 `1e-4`）；批量入口 `run_hetero.py` **没有**暴露这两个选项。如需改变它们，可直接调用单次训练：

```bash
python train_hetero.py --scenario rotation --method gradnorm --seed 0 --hidden-dim 128 --weight-decay 0.0001 --runs-dir runs/rotation_hidden128
```

之后自行汇总：

```bash
python summarize_hetero.py --runs-dir runs/rotation_hidden128 --out-dir results/rotation_hidden128
python plot_hetero.py --runs-dir runs/rotation_hidden128 --out-dir results/rotation_hidden128/figures
```

### 11.2 `--skip-existing` 不是断点续训

- `--skip-existing` 只根据该 run 的 `results.json` 是否存在决定跳过；**不会校验新命令和旧命令参数是否一致**。
- 训练意外中止、`metrics.csv` 已存在但 `results.json` 不存在时，原脚本**不能从某个 epoch 自动恢复优化器状态**。检查残留目录并备份后，可用 `--overwrite` **从头重跑该 seed**。
- **不能同时**指定 `--skip-existing` 和 `--overwrite`。
- 不同 `train-fraction`、`alpha`、固定权重组合、网络结构、CLIP 权重、模型名、缓存和实验设计应使用**不同的 `--runs-dir` 与 `--results-dir`**。
- 将 Color 和 Rotation 混进同一个 `runs-dir` 会触发汇总错误；这是防止混用实验条件的保护。

### 11.3 建议的标准化文件命名

```text
runs/rotation/                 # 默认主实验
results/rotation/
runs/rotation_20pct/           # train-fraction=0.2
results/rotation_20pct/
runs/rotation_alpha05/         # GradNorm alpha=0.5
results/rotation_alpha05/
runs/rotation_fixed_075_125/   # fixed weights 0.75/1.25
results/rotation_fixed_075_125/
```

尽量不要覆盖已用于报告中的原始 `metrics.csv`、`results.json` 或 `best.pt`。

---

## 12. 结果文件与指标释义

### 12.1 单 run 文件

```text
runs/rotation/
├── fine_only/seed_0/
│   ├── metrics.csv
│   ├── best.pt
│   └── results.json
├── aux_only/seed_0/...
├── equal/seed_0/...
├── uncertainty/seed_0/...
├── gradnorm/seed_0/...
└── pcgrad/seed_0/...
```

- `metrics.csv`：每个 epoch 的训练/验证 CE、Validation Accuracy、两任务权重、梯度统计、训练累计秒数等。
- `best.pt`：**按验证分数**选出的模型和（适用时）权重平衡器状态；不是“最终 epoch 强制保存”。
- `results.json`：最佳 epoch、测试准确率、最终权重、冲突率、样本数量、PyTorch/GPU、原权重 SHA256 与参数配置等。

对于 `fine_only` / `aux_only`，另一任务没有被优化，相应的**报告准确率为 null/空值**；不要将其当作 `0%` 或参与双任务平均。程序仍可前向计算未训练头，但它不是有效的单任务指标。

### 12.2 汇总文件

```text
results/rotation/
├── summary.csv
├── summary.json
├── paired_differences.csv
└── figures/
    ├── validation_fine.png
    ├── validation_aux.png
    ├── gradient_cosine.png
    ├── gradient_conflict_rate.png
    ├── raw_grad_fine.png
    ├── raw_grad_aux.png
    ├── weight_fine.png
    ├── weight_aux.png
    ├── gradnorm_rate_fine.png
    └── gradnorm_rate_aux.png
```

- **`summary.csv`**：各方法在多 seed 下的测试准确率、冲突率、任务权重等的均值和样本标准差。文件字段含 `*_mean` / `*_std`。
- **`paired_differences.csv`**：同一 seed 下该方法相对 `equal` 的准确率差异，字段 `delta_vs_equal_pp`；单位是**百分点（pp）**，不是相对增长百分比。
- **`figures/*.png`**：来自 `metrics.csv` 的逐 epoch 曲线，通常展示 seed 均值及可用时的标准差带。部分缺少适用指标的曲线/方法可能不会显示。

需要重新生成汇总而不重新训练：

```bash
python summarize_hetero.py --runs-dir runs/rotation --out-dir results/rotation
python plot_hetero.py --runs-dir runs/rotation --out-dir results/rotation/figures
```

示例解释：若 `delta_vs_equal_pp = +0.30`，表示**比同 seed 的 Equal 高 0.30 个百分点**，不是“提高 30%”。

---

## 13. 如何写实验结论

### 13.1 先回答四个问题

1. **真的出现梯度冲突了吗？** 先看 Equal 的 `grad_conflict_rate`、原始 `grad_cosine` 和两个原始梯度范数；不要仅凭任务名字推断冲突。
2. **是幅度不平衡，还是方向冲突？** GradNorm 的权重变化/相对逆训练速率主要与幅度和相对学习速度有关；PCGrad 则针对负内积作方向投影。
3. **训练动态改善了最终泛化吗？** 必须看最佳验证选模后的 Test Fine/Rotation Accuracy，并用同 seed 的 Equal 做配对差值；不能用漂亮的梯度图代替准确率证据。
4. **收益是否超过随机波动和更简单的固定权重？** 至少 3 seeds，最好补 `fixed`、训练样本比例、不同 α 消融；对 0.1–0.3 pp 的细微差距不要直接声称“统计显著”。

### 13.2 一个适合报告的比较框架

| 模块 | 必须报告的内容 | 研究价值 |
|---|---|---|
| 原始 Clean | Fine/Coarse 单任务和 Equal/UW/GradNorm | 低成本、可比的算法迁移基线 |
| Hetero Rotation | Fine-only、Rotation-only、Equal、UW、GradNorm、PCGrad | 检验潜在异质任务竞争 |
| Gradient diagnostics | `cos`、`conflict rate`、原始范数与权重曲线 | 区分方向冲突与训练速度不平衡 |
| Ablation | Fixed weights、少样本、GradNorm α | 排除更简单的解释 |
| Reproducibility | seeds、split、权重 SHA、模型名、GPU/PyTorch | 保证他人可检查实验条件 |

建议在论文/报告中使用严谨表述：

> 本研究在冻结的本地 OpenCLIP ViT-B/32 图像特征上，采用 CIFAR-100 物体分类与旋转角预测构建异质双任务设置，比较等权、任务不确定性加权、GradNorm 与 PCGrad，并记录共享层原始梯度余弦及负梯度比例。该研究属于**方法迁移与受控机制分析**，不以复制原论文特定数据集上的绝对指标为目标。

若平均梯度余弦基本为正、冲突率低，正确结论应是**当前设置没有形成明显的负梯度冲突**，而不是把它包装成已实现“强冲突数据集”。如果 Rotation-only 也接近 25%，说明冻结特征的角度可解码性需要优先诊断。解冻 CLIP 最后一层或增加 LoRA **属于下一阶段新实现**；当前 ZIP 没有提供该训练路线，不应把它写成已支持的选项。

---

## 14. 常见报错与排查

| 现象/报错 | 可能原因 | 建议处理 |
|---|---|---|
| `torch.cuda.is_available()` 为 False | CPU 版 torch、显卡驱动或 CUDA wheel 不兼容 | 按 PyTorch 官网安装成套 CUDA 版 torch + torchvision；先运行 `check_env.py` |
| `sm_120` warning / CUDA kernel 报错 | 当前 PyTorch wheel 对 RTX 50 系列支持不完整 | 更新到官方支持 RTX 5070 的 wheel；不要只看 CUDA Toolkit 版本 |
| `Local weights not found` | 权重路径或工作目录不对 | 在项目根目录运行，核对 `weights/open_clip_model.safetensors` |
| `missing visual.*` / `Wrong visual architecture` | checkpoint 不是可直接加载的 OpenCLIP ViT-B/32 state dict | 对照来源模型卡，确认架构/格式；不要强行忽略 |
| 模型加载成功但结果异常 | GELU/QuickGELU、预处理或权重来源可能不匹配 | 核对模型卡与 `--model-name`，重新提取特征并隔离旧结果 |
| `CIFAR-100 archive MD5 mismatch` | 压缩包不完整或不是官方 Python 归档 | 检查下载文件，不要跳过校验把来源不明数据当官方数据 |
| `No clean cache ... computing all four rotations` | 没有可复用的旧版 0° 特征 | 正常行为：会提取四个角度，花费更多首次编码时间 |
| `different checkpoint ... model name or non-normalized` | 缓存与当前 CLIP 配置不一致 | 根据真实权重核对，必要时用 `--no-reuse-clean` 重算四角度 |
| `Rotation cache exists` | 缓存目录已存在 | 确认要替换才使用 `--force`，否则避免破坏既有缓存 |
| `Missing ... train.pt or test.pt` | 未执行特征准备脚本，或路径不正确 | 先运行对应 `prepare_*_features.py` |
| `Experiment output exists` | 同一方法/seed 的 `metrics.csv` 已存在 | 保留结果则换新目录；确认重跑用 `--overwrite` |
| `--skip-existing` 后结果不更新 | 旧 `results.json` 仍存在 | 新配置用新 runs/results 目录；不要错误地复用旧产物 |
| `Color quantile thresholds ... different split seed` | Color 缓存的 split seed 与训练不一致 | 用一致的 `--split-seed` 重新准备 Color 缓存 |
| Windows 处理图片时卡住 | DataLoader worker 兼容性 | 采用 `--num-workers 0` |
| `CUDA out of memory` | 特征提取 batch 偏大 | 将 `prepare_rotation_features.py --batch-size` 由 64 调成 32 或 16 |
| 图表没有某方法/指标 | 单任务无相应指标，或日志缺失/未完成 | 检查 `metrics.csv`、`results.json`；不要填入虚构数值 |
| Rotation Accuracy ≈ 25% | 冻结 CLIP 对方向信息可能不敏感、任务难或配置不匹配 | 先检查 Rotation-only、0°/90° 特征差异、标签和模型配置 |

---

## 15. 可复现性检查清单

正式发表/提交实验报告前建议逐项确认：

- [ ] 保存本地模型权重的**来源 URL / 模型卡 / 训练来源**，确认 GELU/QuickGELU 和预处理。
- [ ] `check_env.py --test-clip` 通过；记录 **GPU、PyTorch、torchvision 和 CUDA** 版本。
- [ ] `python -m pytest -q tests` 通过。
- [ ] 保存 `data/features/cache_info.json` 和/或 `data/hetero_rotation/cache_info.json`。
- [ ] 统一 `--split-seed 2026` 与训练 seeds `0 1 2`。
- [ ] 固定训练/验证/测试协议（尤其 Rotation：Fine 看 0°，Aux 看四角度）。
- [ ] 主实验统一 `--diagnose-every 1`，避免因诊断采样频率不同而误比冲突率。
- [ ] 每个消融使用独立 runs/results 目录，不用旧 `results.json` 充当新配置结果。
- [ ] 保留全部 `metrics.csv`、`results.json`、`best.pt` 与 `paired_differences.csv`。
- [ ] 用同 seed 配对比较 Equal；给出均值 ± 标准差，明确只有三组 seeds 的统计局限。
- [ ] 结果表中不把 `null` 单任务成绩当作 0，也不把 `pp` 当成相对百分比。
- [ ] 明确写出：**冻结 CLIP、CIFAR-100 构造任务、方法迁移而非原论文严格数值复现**。

可另外保存当前环境快照：

```bash
python -m pip freeze > environment_pip_freeze.txt
```

---

## 16. 参考资料

**算法论文**

1. Kendall, Gal, Cipolla. *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*. CVPR 2018. <https://arxiv.org/abs/1705.07115>
2. Chen, Badrinarayanan, Lee, Rabinovich. *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*. ICML 2018. <https://proceedings.mlr.press/v80/chen18a.html>
3. Yu et al. *Gradient Surgery for Multi-Task Learning*. NeurIPS 2020. <https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html>

**实现与数据**

- OpenCLIP（本地权重加载、QuickGELU 注意事项）：<https://github.com/mlfoundations/open_clip>
- PyTorch 官方安装：<https://pytorch.org/get-started/locally/>
- torchvision CIFAR100：<https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html>

本项目是独立实验教学/科研实现；没有重新发布第三方数据集或模型权重。请遵守对应数据集、预训练权重及软件的许可条件。
