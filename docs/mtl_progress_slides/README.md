# 多任务学习进展汇报

- `main.pdf`：给导师过目的 17 页演示稿。
- `main.tex`：可编辑的中文 Beamer 源码。
- `images/table_1.png`、`images/table_2.png`：阅读笔记中的两张论文结果表。
- `images/clean_validation_*.png`：干净 Fine + Coarse 的验证准确率。
- `images/coarse20_validation_*.png`：20% Coarse 训练标签噪声下的验证准确率。
- `images/rotation_validation_*.png`：Fine + Rotation 的验证准确率。

在本目录用 XeLaTeX 编译两次：

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

阅读部分按 `../main.md` 的顺序展开，包含多任务建模、回归和分类的概率推导、分类近似、GradNorm 的完整定义及两张论文结果表。实验部分依据本人描述的经过、`../mtl_exploratory_paper/main.tex` 与项目中的结果汇总。没有新增训练。
