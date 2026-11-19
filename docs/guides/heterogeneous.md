# CLIP ViT-B/32 × CIFAR-100：异质多任务与梯度冲突实验（RTX 5070 12GB）

> 相对于旧版 Fine100 + Coarse20：**新实验的主任务是 Fine100 + 图像 Rotation4**；再增加一个不必重提取 CLIP 特征的 **Fine100 + Color4** 低成本对照。均使用冻结 CLIP 编码器 + 共享可训练 MLP。特别增加两任务梯度余弦、负余弦比例，以及 PCGrad。该方案属于算法迁移验证，不是原论文 Cityscapes/NYUv2 指标复现。

## 1. 实验逻辑

- **Fine（100 类）**：识别 CIFAR-100 细类，例如 apple、bus；使用 CE。
- **Rotation（4 类）**：同一张图像先在原始 32×32 像素上逆时针旋转 0°、90°、180°、270°，再使用你本地同一套 CLIP 权重和预处理将 4 个角度分别编码为 512-d 特征。任务预测旋转角；同样使用 CE。
- **为什么有机会产生冲突**：语义分类往往希望旋转不改变类别，旋转预测却需要保留方向信息，目标存在潜在张力。但 **不保证** 一定存在负梯度余弦，应该测量而不是预设。
- **训练**：每张训练图片每轮仅选 1 个角度。角度由 `(原始样本序号, epoch, seed)` 确定，连续四轮覆盖四种角度；同一个 seed 在各方法中完全一致。Fine 和 Rotation 在**同一个旋转输入**上预测，各自梯度均回传到共享 MLP。
- **验证/测试**：Fine 在 0° 图片上计算 5k/10k 个图像准确率；Rotation 在同一验证/测试集合的四个角度上计算 20k/40k 个方向预测准确率。选最佳 checkpoint 只用验证成绩；测试集最后评估一次。
- **对照**：`fine_only`, `aux_only`, `equal`, `uncertainty`, `gradnorm`, `pcgrad`；可额外尝试 `fixed`。所有方法用同样的训练/验证分割、初始化 seed 和旋转分配。
- **Color 控制实验**：由原始图像平均 `R-B` 色差计算 4 个**训练集四分位**标签。对每张原始图像使用已有干净 CLIP 特征；无需 CLIP 推理。分位数阈值**只使用 45k 训练集**拟合，以避免验证/测试信息泄露。它只是廉价的异质标签任务，不意味着必然产生更强梯度冲突。

## 2. 需要哪些已有文件？

解压后在根目录 `clip_cifar100_mtl/` 下保留：

```
data/
  cifar-100-python.tar.gz
  cifar-100-python/      # 有也可以，没有会从上面的压缩包离线解压
  features/train.pt      # 你之前提取的干净 CLIP 特征；可选，但强烈推荐复用
  features/test.pt
weights/
  open_clip_model.safetensors
```

不要用新的压缩包覆盖你已经提取的大型 `data/` 或已有的 `weights/`、`runs/`、`results/`。新代码单独创建 `data/hetero_rotation/`、`runs/rotation/`、`results/rotation/`，并不会修改以前的实验结果。

### 权重一致性

新版重用 0° CLIP 特征前会验证 `data/features/*` 的 `checkpoint_sha256`、模型名和归一化方式与当前 `weights/open_clip_model.safetensors` 一致。不一致时会直接报错，**不会静默混合两套权重的特征**。如果确认要用现有的本地权重重新抽取全部四种角度，则加 `--no-reuse-clean`，不必修改原有特征。

`open_clip_model.safetensors` 文件名不能证明它是 OpenAI 官方预训练的 checkpoint。请参考模型卡选 `ViT-B-32` 或 `ViT-B-32-quickgelu`；OpenAI 原始模型使用 QuickGELU。新实验使用的仍是冻结 CLIP 特征，并不是在 5070 上端到端微调 ViT。

## 3. 安装和自检

已有旧版项目的 Python 环境可复用。RTX 5070 12GB 需要安装与你的 GPU/CUDA 兼容的 PyTorch + torchvision，然后运行：

```bash
python -m pip install -r requirements.txt
python check_env.py --test-clip
python -m pytest -q tests
```

如果之前在 `prepare_features.py` 用 `--model-name ViT-B-32-quickgelu`，下文命令也加同样的参数，并保持权重路径一致。程序不会自动下载 CLIP 权重或 CIFAR-100。

## 4. 快速零推理对照：Fine + Color

```bash
python prepare_color_features.py
python run_hetero.py --scenario color --quick
# 如运行无误再正式跑：
python run_hetero.py --scenario color --seeds 0 1 2 --epochs 30 --skip-existing
```

它只复用 `data/features/train.pt` 和 `test.pt`。主要用于先确认训练流程和梯度诊断正常，再做旋转主实验。

## 5. 正式异质实验：Fine + Rotation（推荐）

**第一步：准备多角度缓存**

```bash
python prepare_rotation_features.py --batch-size 64 --model-name ViT-B-32
```

如果模型卡指明 QuickGELU：

```bash
python prepare_rotation_features.py --batch-size 64 --model-name ViT-B-32-quickgelu
```

如 CUDA 显存不够，改为 `--batch-size 32`。如果没有可复用的 `data/features/*.pt`，脚本会自行提取 0°，因此总共会提取 4 种角度；有可复用的缓存时只需提取其余 3 种角度。缓存空间大约 **246 MB**（60,000×4×512×2 bytes，实际文件还有少量元数据）。

**第二步：快速验证**

```bash
python run_hetero.py --scenario rotation --quick
```

会在单独 `runs/rotation_quick/` 与 `results/rotation_quick/` 输出两轮训练的自检结果。两轮不足以得出科研结论。

**第三步：正式比较**

```bash
python run_hetero.py --scenario rotation --seeds 0 1 2 --epochs 30 --diagnose-every 1 --skip-existing
```

默认方法：Fine-only、Rotation-only、Equal、Uncertainty、GradNorm、PCGrad；30 轮 × 3 seeds。设置 `--diagnose-every 1` 可以让 Equal / Uncertainty / GradNorm 与 PCGrad **每批次都测量梯度冲突**，更公平，代价是训练速度略慢。训练共享 MLP 规模很小，特征提取是主要的 GPU 计算开销。

**第四步：额外的任务困难度对照（可选）**

使用同一缓存，减少训练数据到 20%；一定要使用**新**输出目录，不能和完整训练混在一起：

```bash
python run_hetero.py --scenario rotation --train-fraction 0.2 --seeds 0 1 2 --epochs 30 --runs-dir runs/rotation_20pct --results-dir results/rotation_20pct --skip-existing
```

**第五步：固定权重对照（可选）**

```bash
python run_hetero.py --scenario rotation --methods fixed --fixed-fine 0.75 --fixed-aux 1.25 --seeds 0 1 2 --epochs 30 --skip-existing
```

可进一步比较不同固定权重（若调整固定权重，请为每个权重配置使用**独立**的 `--runs-dir` 和 `--results-dir`，避免混淆）。

## 6. 输出说明

```
runs/rotation/
  equal/seed_0/metrics.csv
  equal/seed_0/results.json
  equal/seed_0/best.pt
  gradnorm/seed_0/...
  ...
results/rotation/
  summary.csv
  paired_differences.csv
  summary.json
  figures/
    validation_fine.png
    validation_aux.png
    gradient_cosine.png
    gradient_conflict_rate.png
    raw_grad_fine.png
    raw_grad_aux.png
    weight_fine.png
    weight_aux.png
    gradnorm_rate_fine.png
    gradnorm_rate_aux.png
```

- `grad_cosine`：每 `--diagnose-every` 个批次测量一次在**全部共享 MLP 参数**上的原始两任务梯度余弦，cos<0 为方向冲突；PCGrad 为执行投影必须每批计算。
- `grad_conflict_rate`：上述被采样的批次中 cos<0 的比例。两种方法的采样频率并不一致，跨方法比较时优先查看 `Equal / UW / GradNorm` 的统一频率；需要更精确时设置 `--diagnose-every 1` 让各方法每批诊断（更耗训练时间）。
- `raw_grad_norm_fine/aux`：单任务损失在共享参数上的原始梯度范数，可观察**量级不平衡**。
- `weight_fine/aux`：当前优化方法对分类 CE 的乘数；Uncertainty 的加性项另计。
- `test_mean_acc_pct`：Fine 与 Aux 的算术平均分数，只用作方便汇总，不足以衡量所有任务的相对收益。
- `paired_differences.csv`：与相同 seed 的 Equal Weight 比较两个任务和平均准确率的差值，单位百分点。
- *单任务*与*多任务*使用完全相同的随机旋转训练输入（唯一差别是优化损失），确保公平比较。**不能直接将新实验 Fine-only 与上一版仅原始 0° 训练的 Fine-only 作因果比较**。

## 7. 怎样解释真正有价值的结果？

1. 看 `Equal` 的 cos 曲线。如果大量批次出现负余弦，两任务确实存在共享梯度方向竞争；若几乎没有，不能声称构造了“强冲突”任务。
2. 对比 GradNorm 的权重轨迹和原始梯度范数。**GradNorm 平衡梯度大小/训练速度，不负责消除方向冲突**。
3. 看 PCGrad 是否在冲突明显时比 Equal 更好；如果没有改善也应如实报告（投影不保证泛化提升）。
4. 看 Fine-only、Rotation-only 与双任务方法相差多少；在同一个测试集、同一个任务定义内对比。
5. Rotation 的随机猜测基线为 **25%**。如果 Rotation-only 都只有约 25%，说明这一冻结 CLIP 表征中当前旋转任务难以解码，而不是“动态方法没用”。可以考虑改变辅助任务或对 CLIP 最后一层做受控微调。
6. 至少跑 3 种 seed，增加配对差值及方差；不要将小于随机波动的微小精度提升写成“显著优于”。

## 8. 安全与边界

- 不联网下载模型 / 数据。
- 不会修改已有 `data/features/`、`runs/`、`results/`；新缓存与新结果目录分开。
- 原本的 Fine+Coarse 代码仍保留在包中，可做旧基线。
- 代码在合成特征上通过自动测试与端到端 CPU 运行；**这不是实际 CIFAR-100 或你的 RTX 5070 的实验结果**。真实 CLIP 权重来源必须核对模型卡。
- 若本地 PyTorch / torchvision 不兼容 RTX 50 系列，先运行旧版 `check_env.py` 处理驱动和 CUDA；无必要重新下载已完成的 CIFAR 和权重。

### 方法文献

- Chen et al., GradNorm (ICML 2018): https://proceedings.mlr.press/v80/chen18a.html
- Yu et al., Gradient Surgery for Multi-Task Learning (NeurIPS 2020): https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
- OpenCLIP official repository: https://github.com/mlfoundations/open_clip
