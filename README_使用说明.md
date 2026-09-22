# EEG-RSA 一键分析工具（Python版）

## 这个版本解决什么问题

这是一个面向**任务态 EEG** 的基础 RSA 图形化工具。用户不需要 MATLAB，也不需要自己写 Python 代码。

完整流程：

`.set → 自动读取事件 → 选择条件 → 选择时间窗 → 选择距离 → Neural RDM → 建立/导入 Model RDM → RSA相关 → 出图 + CSV导出`

## 推荐数据

优先使用已经完成预处理、分段（epoched）的 EEGLAB `.set`。工具也会尝试读取连续 `.set`，并根据 EEGLAB event/annotation 在所选时间窗内临时分段。

若 `.set` 使用外部 `.fdt`，请保证 `.set` 与 `.fdt` 位于同一文件夹。

## 如何运行

Windows 用户直接双击：

`启动_RSA.bat`

首次运行会检查并安装：MNE-Python、MNE-RSA、NumPy、SciPy、Pandas、Matplotlib。

其中 MNE-Python 负责 EEGLAB `.set` / Epoch 数据读取，MNE-RSA 负责 RDM/RSA 核心计算。

## 当前基础版到底算什么

假设用户选择 4 个条件和 140–200 ms：

1. 每个条件提取该时间窗内所有有效 trial；
2. 对 trial 和时间点取平均；
3. 每个条件得到一个 `1 × channel` 的 EEG 空间模式；
4. 对所有条件的空间模式两两计算距离；
5. 得到 `condition × condition` Neural RDM；
6. 与用户建立/导入的 Model RDM 比较。

因此当前版本研究的是：**不同任务条件在指定时间窗内的 EEG 多通道空间模式有多相似/不相似。**

## Neural RDM 距离

基础版提供：

- Correlation distance：`1 - r`，默认推荐；
- Euclidean distance；
- Cosine distance；
- Cityblock / Manhattan distance；
- Chebyshev distance。

Crossnobis / Mahalanobis 不在基础版里伪装成普通下拉选项。它们需要更严格的噪声协方差估计和/或 run/fold 交叉验证结构，后续适合单独增加“高级 RSA”模块。

## Model RDM

有三种方法：

1. 点击“建立 / 编辑 Model RDM”，直接在界面里填写上三角；
2. 点击“导出模板”，用 Excel/WPS 填写后再导入；
3. 直接导入已有 CSV。

要求：

- 矩阵必须与当前选择条件数量一致；
- 必须对称；
- 对角线必须为 0；
- 数值越大表示模型假设中越不相似。

## RDM 比较

当前提供：

- Spearman（默认）；
- Pearson；
- Kendall tau-a（MNE-RSA）。

## 自动输出

每次运行自动建立一个时间戳结果目录，包含：

- `01_条件空间模式.csv`
- `02_Neural_RDM.csv`
- `03_Model_RDM.csv`
- `04_RSA结果.csv`
- `05_分析参数.json`
- `06_Neural_RDM.png`
- `07_Model_RDM.png`
- `08_RSA散点图.png`

## 统计解释提醒

当前基础版的 RSA 相关首先用于单被试/单数据集的表征几何分析和教学演示。界面给出的相关系数 p 值只是该 RDM 向量相关的描述性结果，**不能替代正式的被试组水平统计、置换检验、多重比较校正或噪声天花板分析。**

## 下一阶段可扩展

在基础版真实数据流程跑稳后，再增加：

- time-resolved RSA / RDM movie；
- sensor searchlight；
- Crossnobis；
- run/fold 交叉验证；
- 多 Model RDM 回归/partial RSA；
- 组水平统计与 permutation / cluster correction；
- noise ceiling。
