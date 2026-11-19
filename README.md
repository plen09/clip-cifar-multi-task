# 冻结 CLIP 特征上的多任务学习初探

从 Uncertainty Weighting 与 GradNorm 的论文阅读出发，研究任务权重如何影响 Fine/Coarse 分类；进一步以 Fine + Rotation 测量共享梯度方向，并引入 PCGrad。项目是方法迁移和机制验证。

## 阅读入口

- [中文 LaTeX 论文](docs/mtl_exploratory_paper/main.tex)：方法推导、真实结果、Acc 曲线和局限；项目文件夹可用 XeLaTeX 编译。
- [完整复现指南](docs/guides/reproduction.md)：环境、本地资源、特征提取、训练与评估协议。
- [异质任务说明](docs/guides/heterogeneous.md) · [扰动实验说明](docs/guides/perturbations.md)。

## 项目结构

```text
src/                         数据、模型、权重调整与梯度处理
tests/                       单元测试
analysis/                    基础/扰动/异质任务事后诊断
docs/
  guides/                    使用和复现指南
  mtl_exploratory_paper/      main.tex 与独立 images/ 资源
prepare_*.py                 特征/标签准备入口
train.py / train_hetero.py   单次训练
run_*.py                    批量实验
summarize*.py / plot_*.py    汇总与绘图
results/                    可发布的 CSV/JSON 汇总和图像
  clean/                    Fine + Coarse 干净基线
  rotation/                 Fine + Rotation 主实验
  color/                    Fine + Color 对照
  perturbations/            扰动概览及各场景子目录
runs/                       与 results/ 对应的原始训练记录（不上传）
  clean/                    <method>/seed_<n>/
  rotation/                 <method>/seed_<n>/
  color/                    <method>/seed_<n>/
  perturbations/            <scenario>/<method>/seed_<n>/
data/ weights/ logs/        其他本地资产，不纳入 Git
```

根目录训练入口不变，默认输出已统一到上述分层结构。原始 `results.json`、`metrics.csv` 和 `best.pt` 随 `runs/` 保留在本地，公开汇总不等于公开全部原始训练记录。历史 JSON 中的 `config.runs_dir` 保留运行时原值，不因搬迁改写；分析脚本根据新目录发现记录。上传前如需完整复核，应另行选择可发布的运行记录。

## 最小运行流程

先按复现指南安装兼容的 PyTorch 与 `requirements.txt` 依赖，并准备本地 CIFAR-100 和权重。项目不会自动下载这两类资产。所有命令在项目根目录执行。

```powershell
python -m pytest -q tests
python prepare_features.py
python run_all.py --seeds 0 1 2 --epochs 30
python prepare_rotation_features.py
python run_hetero.py --scenario rotation --seeds 0 1 2 --epochs 30 --diagnose-every 1 --skip-existing
```

权重的模型结构、激活与预处理必须根据来源模型卡核对；不能仅凭文件名认定是 OpenAI 官方预训练权重。已有缓存不因改参数自动变成新缓存，替换前请备份。

事后分析入口：

```powershell
python analysis/analyze_preliminary.py
python analysis/analyze_perturbations.py
python analysis/analyze_hetero.py
```

这些脚本读取本地运行记录，重新生成对应 `results/` 子目录下的分析 JSON。快速自检使用 `clean_quick/`、`rotation_quick/` 或 `color_quick/`，避免覆盖正式结果。

## 当前观察与证据

Fine + Rotation 下，GradNorm 相对 Equal 的 Fine 准确率下降约 1.29 pp、Rotation 提高约 0.84 pp；PCGrad 的平均收益约 0.11 pp，但三个种子方向不一致。该结果更支持任务取舍和适用条件的讨论，而不是稳定整体优势。主表见 [Rotation 汇总](results/rotation/summary.csv)，逐种子比较见 [配对差值](results/rotation/paired_differences.csv)。