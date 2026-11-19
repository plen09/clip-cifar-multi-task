# 第二轮：CLIP 特征扰动与任务冲突实验（离线，可直接复用缓存）

> 基于原始离线实验包，直接复用 `data/features/train.pt` / `test.pt`。不需要重新运行或下载 CLIP，也不修改已经保存的特征。保留原有 `runs/`、`results/`，新实验默认写到 `runs/perturbations/`、`results/perturbations/`。

## 0. 拷贝方式

把这个 ZIP 解压成 `clip_cifar100_mtl/`，**覆盖旧项目中的 `.py` 源码即可，不要删除你电脑上的 data、weights、runs、results。**

在项目根目录检查文件是否存在：

```text
data/features/train.pt       # 之前已经提取的 CLIP 512 维特征
data/features/test.pt
results/clean/summary.csv          # 可选；旧实验的 Clean Baseline
```

## 1. 推荐直接跑（4 条条件 × 3 种方法 × 3 seeds × 30 epochs）

```bash
# 从最小测试开始：使用 seed=0、2 epochs，结果写进 *_quick 目录
python run_perturbations.py --quick

# 正式推荐组合（不会覆盖之前的 runs/ 和 results/）
python run_perturbations.py

# 如果中途停止，可直接跳过已完整结束的组合
python run_perturbations.py --skip-existing
```

默认四项：

- `gaussian07`：训练时对 CLIP 512-d 特征加相对 L2 强度 sigma=0.7 的高斯噪声，每次取样重新生成。
- `coarse20`：随机翻转 20% 的训练集 coarse 标签，故意制造与 fine 标签不一致的样本。
- `fine20`：随机翻转 20% 的训练集 fine 标签。
- `lowdata10`：只用原训练集的 10%（约 4500 张），fine 分层抽样。

每组只训练 Equal、Uncertainty、GradNorm；单任务基线选做。均保持原来 5k 验证集和 10k 官方测试集**完全干净**，仍在验证集选最佳 checkpoint，最后只测一次测试集。

## 2. 自定义强度、方法、随机种子

```bash
# 只跑两个更强冲突条件，含三组随机种子
python run_perturbations.py --scenarios coarse20 coarse40 --seeds 0 1 2

# 测试特征噪声梯度
python run_perturbations.py --scenarios gaussian03 gaussian07 dropout20

# 包含单任务基线（对标签噪声尤其有参考价值）
python run_perturbations.py --scenarios coarse20 fine20 --methods fine_only coarse_only equal uncertainty gradnorm

# 清洁对照（若之前 clean 配置一致，可复用旧 results/clean/summary.csv，不必再跑）
python run_perturbations.py --scenarios clean

# 比较 GradNorm 学到的约 0.92/1.08 与固定权重（在同一噪声条件中）
python run_perturbations.py --scenarios coarse20 \
  --methods equal fixed uncertainty gradnorm --fixed-fine 0.92 --fixed-coarse 1.08

# 指定一个方法单独跑，保留原始目录
python train.py --method gradnorm --seed 0 --epochs 30 \
  --label-noise coarse --label-noise-rate 0.4 --runs-dir runs/perturbations/coarse40

# 训练集特征扰动
python train.py --method equal --seed 0 --epochs 30 \
  --feature-noise gaussian --feature-noise-level 0.7 --runs-dir runs/perturbations/gaussian07

# 查看有哪些任务、会执行哪些命令（不运行训练）
python run_perturbations.py --dry-run
```

可选 `--scenarios` 名称：`gaussian03 gaussian07 dropout20 coarse20 coarse40 fine20 lowdata10 clean`。需要增加组合可直接编辑 `run_perturbations.py` 的 `SCENARIOS` 字典。

如已做完某个完整场景，建议使用 `--skip-existing`；覆盖原本场景需要显式 `--overwrite`，不能同时使用这两个参数。

## 3. 结果如何比较？

```text
runs/perturbations/<scenario>/<method>/seed_<n>/metrics.csv
runs/perturbations/<scenario>/<method>/seed_<n>/results.json
runs/perturbations/<scenario>/<method>/seed_<n>/best.pt
results/perturbations/<scenario>/summary.csv
results/perturbations/<scenario>/figures/*.png
results/perturbations/overview.csv
```

`overview.csv` 把不同场景汇总起来，**最重要的列是 `*_delta_vs_equal_pp`**（同一场景、同一指标，动态方法平均成绩减去 Equal 的平均成绩，单位百分点）。如果你原来的 `results/clean/summary.csv` 还在，它会加入 `existing_clean` 行以便横向参照；但不要把每种噪声条件与干净训练直接混淆成同一任务。

可手动重新汇总：

```bash
python compare_perturbations.py --results-dir results/perturbations --baseline results/clean/summary.csv
```

## 4. 实验解释的四个要点

1. **所有扰动只影响训练样本。** 训练时随机特征噪声只加在 `train_loader`，标签噪声只改训练数据的独立副本。原来的 `train.pt` / `test.pt` 绝不写入或覆盖。验证和测试没有任何扰动。
2. **特征噪声 ≠ 图片噪声。** Gaussian 是直接扰动 CLIP 输出后的 512 维向量，不等于像素加噪；这里验证的是冻结特征上 MLP 的稳定性，而不是真实图像腐蚀下 CLIP 的鲁棒性。特征噪声每个 minibatch 动态生成，且保留原始每行 L2 范数。`sigma` 表示 512-d 噪声向量期望 L2 范数约为原向量的 sigma 倍，不是每一维都加 sigma。`dropout20` 是随机把 20% 特征坐标设为零并保持其余向量方向的归一化。
3. **标签噪声用错误标签，不是标签平滑。** 按 Bernoulli(p) 随机选择训练样本，然后在其它类别中随机选一个新标签；原标签一定改变。噪声种子固定为 corruption_seed+seed，在相同 seed 下所有方法的被污染样本完全一样。Coarse 标签错误时 fine 仍正确，反之亦然。这是人为制造的任务冲突，不能代表真实标注噪声分布。
4. **低样本比例与 clean 不能直接等量比较。** train fraction=0.1 从 45,000 样本中按 fine 类别固定抽取约 4,500；Val/Test 数量不变。若想将“减少训练样本”作为独立变量，要比较此条件下各加权方法与同条件 Equal 的结果。

**负结果也是结果。** 特别是 GradNorm：它可能因为一个有噪声的任务损失下降慢而提高该任务梯度贡献，未必能有效识别标签噪声并自动降权。

建议重点记录：Clean→Noise 条件下的 Fine / Coarse Acc、每个场景动态加权相对于等权的差值、任务权重曲线、GradNorm 的相对逆训练速率和目标梯度，以及随机种子间的变化。只有三次种子时，0.1–0.3pp 的细微差距仍不宜直接宣称统计显著。

## 5. 自检（无需 GPU、数据和 CLIP 模型）

```bash
python -m unittest discover -s tests -v
python smoke_test.py
python run_perturbations.py --dry-run
```

测试包含：噪声确定性、Fine 分层采样、Coarse/Fine 标签互不污染、特征归一化、3 种训练方式 CPU 小样本端到端跑通，以及原有算法/本地文件代码测试。

## 6. 研究路线建议

这个扰动包属于 **低成本压力测试**，不是原始 GradNorm/Kendall 实验的严格数值复现。建议先看 `coarse20` 和 `fine20` 是否能制造两任务间的训练速度差异；若权重仍变化很小，优先改成更异质的多任务（例如图片旋转角预测+Fine 分类——需要对旋转图像重新提取特征），或改用小模型联合微调共享编码器。
