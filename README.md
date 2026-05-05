# Retinal OCT 图像分类 — 计算机视觉课程项目

南安普顿大学计算机视觉模块课程作业。使用**传统计算机视觉方法**（手工特征 + 经典机器学习）与**深度学习方法**（预训练 CNN 微调）对视网膜光学相干断层扫描（OCT）图像进行四分类。

**类别**: CNV（脉络膜新生血管）、DME（糖尿病黄斑水肿）、DRUSEN（早期 AMD）、NORMAL（正常）

## 数据集

使用 [Kaggle 2017 OCT 数据集](https://www.kaggle.com/datasets/paultimothymooney/kermany2018?utm_source=chatgpt.com)，共 84,484 张 JPEG 图像，按患者级别划分为训练/验证/测试集，确保零患者泄漏。

| 划分 | CNV | DME | DRUSEN | NORMAL | 总计 |
|------|-----|-----|--------|--------|------|
| train_final | 31,890 | 9,727 | 7,385 | 22,555 | 71,557 |
| val_final | 5,315 | 1,621 | 1,231 | 3,760 | 11,927 |
| test_final | 242 | 242 | 242 | 242 | 968 |

数据目录结构: `data/raw/{train,val,test}/{CNV,DME,DRUSEN,NORMAL}/*.jpeg`

## 项目结构

```
coursework2/
├── main.py                  # CLI 入口，7 个子命令
├── src/
│   ├── config.py            # 项目常量、路径配置（Paths dataclass）
│   ├── data.py              # 数据集扫描、患者感知划分生成、审计报告
│   ├── preprocess.py        # 图像预处理（灰度读取、缩放、归一化）
│   ├── features.py          # 传统特征提取（HOG、LBP、SIFT+BoVW）
│   ├── traditional_models.py # 传统分类器工厂（LinearSVM、RBF SVM、RandomForest）
│   ├── train.py             # 传统实验编排（单次运行 + 3×3 矩阵遍历）
│   ├── evaluate.py          # 评估指标（准确率、Macro-F1）与混淆矩阵可视化
│   ├── analysis.py          # 后处理分析（结果汇总、误差图库、策略说明）
│   ├── processed.py         # 预处理数据集导出（供团队共享）
│   ├── utils.py             # 通用工具（随机种子、日志、计时器、SHA1）
│   ├── draw.py              # 报告图表生成（~12 张可视化图）
│   ├── dl_models.py         # 兼容模块，重导出 DLRunConfig
│   ├── dl_train.py          # 兼容模块，重导出 DL 实验运行函数
│   └── deeplearning/        # 深度学习子包（主实现）
│       ├── models.py        # 模型构建器（ResNet18/34/50、VGG16、MobileNetV2）
│       ├── trainer.py       # 训练框架（DLRunConfig、Dataset、Trainer）
│       ├── augmentations.py # 数据增强（医学安全变换流水线）
│       ├── run_experiment.py # 端到端 DL 实验运行器
│       ├── dataset_manifest.py # 清单辅助函数（划分选择、路径解析、分层采样）
│       ├── collect_results.py # 结果收集与协议校验
│       ├── train_deep.py    # 独立 DL 训练 CLI
│       └── run_all_deep_models.py # 批量运行全部 5 种 DL 架构
├── notebooks/
│   ├── 01_data_visualisation.ipynb  # 数据探索与可视化
│   ├── 02_feature_demo.ipynb        # 特征提取演示（HOG/LBP/SIFT）
│   ├── 03_results_analysis.ipynb    # 结果分析与排名
│   ├── deeplearningexplain.ipynb    # 深度学习可解释性分析
│   ├── interpretation.ipynb         # 模型解释
│   └── all_5_results_dashboard.ipynb # 全模型结果仪表盘
├── document/latex/          # CVPR 格式论文（LaTeX 源码 + PDF）
├── data/
│   ├── raw/                 # 原始 OCT 图像数据集
│   ├── processed/           # 预处理后的图像导出
│   └── interim/             # 中间产物（特征缓存、清单文件）
└── outputs/
    ├── tables/              # 结果 CSV 表格
    ├── figures/             # 可视化图表
    ├── models/              # 训练好的模型检查点
    └── logs/                # 训练日志
```

## 模块与代码文件说明

### 核心流水线

| 文件 | 职责 |
|------|------|
| [main.py](main.py) | 统一 CLI 入口，提供 `audit`、`train-trad`、`train-matrix`、`train-dl`、`final-test`、`analysis-artifacts`、`export-processed`、`collect-results` 共 7 个子命令 |
| [src/config.py](src/config.py) | 定义 `CLASS_NAMES`（4 类别）、`RAW_SPLITS`、`Paths` 路径配置 dataclass 及根目录自动解析函数 |
| [src/utils.py](src/utils.py) | 通用工具：`set_seed()` 固定随机种子、`setup_logger()` 双通道日志（控制台+文件）、`timed()` 上下文计时器、`file_sha1()` 文件哈希 |

### 数据处理

| 文件 | 职责 |
|------|------|
| [src/data.py](src/data.py) | 数据集扫描（遍历 `data/raw/` 解析文件名提取患者/图像 ID）、SHA1 校验完整性、使用 `StratifiedGroupKFold` 生成患者感知的 train/val 划分、构建审计报告 |
| [src/preprocess.py](src/preprocess.py) | 统一的图像预处理函数：`read_grayscale()` OpenCV 灰度读取、`resize_image()` 缩放、`normalize_to_unit()` 归一化至 [0,1] |
| [src/processed.py](src/processed.py) | 预处理批量导出工具：读取划分清单、按配置（尺寸/RGB/归一化）预处理全部图像、输出为 PNG 或 NPY 格式并生成处理清单 CSV |

### 传统方法（手工特征 + 经典 ML）

| 文件 | 职责 |
|------|------|
| [src/features.py](src/features.py) | 三种手工特征提取：**HOG**（方向梯度直方图）、**LBP**（局部二值模式）、**SIFT + BoVW**（SIFT 关键点 + MiniBatchKMeans 词袋模型）。支持 NPZ 压缩缓存的存取 |
| [src/traditional_models.py](src/traditional_models.py) | 分类器工厂：`LinearSVC`（类平衡）、`SVC`（RBF 核，概率输出）、`RandomForestClassifier`（400 棵树，子采样类平衡）。含训练、预测、模型序列化封装 |
| [src/train.py](src/train.py) | 传统实验编排：`TraditionalTrainConfig` 指定特征/分类器/尺寸/种子，`run_traditional_experiment()` 执行端到端实验（特征提取→训练→评估→保存），`run_traditional_matrix()` 遍历 3×3 组合并导出排名表 |

### 深度学习方法（预训练 CNN 微调）

| 文件 | 职责 |
|------|------|
| [src/deeplearning/models.py](src/deeplearning/models.py) | 基于 torchvision 的模型构建：支持 ResNet18/34/50、VGG16、MobileNetV2，替换分类头为 4 类输出，可选骨干冻结 |
| [src/deeplearning/trainer.py](src/deeplearning/trainer.py) | 核心训练框架：`DLRunConfig` 超参数配置、`OCTManifestDataset` 清单驱动的 PyTorch Dataset、`DLTrainer` 管理设备/模型/优化器/数据加载/训练循环/验证/预测 |
| [src/deeplearning/augmentations.py](src/deeplearning/augmentations.py) | 图像变换流水线：训练集使用 RandomHorizontalFlip、RandomRotation(10°)、RandomResizedCrop、ColorJitter；验证/测试集使用 Resize + ImageNet 标准化 |
| [src/deeplearning/run_experiment.py](src/deeplearning/run_experiment.py) | 端到端 DL 实验：清单行选择→数据集构建→训练循环（最佳 Macro-F1 检查点）→最终评估→保存全部产物（历史记录、混淆矩阵、预测 CSV、结果行、元数据 JSON） |
| [src/deeplearning/dataset_manifest.py](src/deeplearning/dataset_manifest.py) | 清单工具函数：`select_split_rows()` 选取指定划分、`attach_data_paths()` 解析原始/处理后路径、`stratified_cap()` 分层采样（调试用） |
| [src/deeplearning/collect_results.py](src/deeplearning/collect_results.py) | 结果汇总器：合并所有传统和 DL 结果 CSV、列名规范化、去重、协议校验（评估划分合法性、重复 test_final 检查、缺失指标检查） |

### 评估与分析

| 文件 | 职责 |
|------|------|
| [src/evaluate.py](src/evaluate.py) | 评估指标计算：`compute_metrics()` 返回准确率/Macro-F1/Macro-Precision/Macro-Recall，`per_class_table()` 生成逐类指标 DataFrame，`save_confusion_matrix()` seaborn 热力图 |
| [src/analysis.py](src/analysis.py) | 分析工具：`build_traditional_summary()` 聚合结果并排序、`build_dl_comparison_template()` 传统与 DL 对比、`write_policy_note()` 评估协议文档、`save_error_gallery()` 误分类图像图库 |
| [src/draw.py](src/draw.py) | 数据驱动的可视化模块：自动发现 `outputs/runs/deeplearning/` 下的训练产物，基于真实 history.csv 绘制学习曲线（Loss + Macro-F1），基于 predictions.csv 中的 softmax 概率绘制 One-vs-Rest ROC 曲线（含 Macro-Avg）。支持单模型与多模型对比图，可通过 `python -m src.draw` 或 `main.py` 子命令调用 |

### 兼容模块

| 文件 | 职责 |
|------|------|
| [src/dl_models.py](src/dl_models.py) | 薄包装，将 `DLRunConfig` 从 `src.deeplearning.trainer` 重导出至 `src.dl_models` |
| [src/dl_train.py](src/dl_train.py) | 薄包装，将 `run_dl_experiment` 和 `DLProtocolError` 从 `src.deeplearning.run_experiment` 重导出至 `src.dl_train` |

### Notebooks

| 文件 | 内容 |
|------|------|
| [01_data_visualisation.ipynb](notebooks/01_data_visualisation.ipynb) | 数据探索：加载审计报告和划分清单，绘制类别分布柱状图，展示每类样本图像网格 |
| [02_feature_demo.ipynb](notebooks/02_feature_demo.ipynb) | 特征提取演示：对 100 张样本计算 HOG/LBP 特征矩阵，可视化 HOG 描述子和 LBP 直方图，演示 SIFT-BoVW 流水线 |
| [03_results_analysis.ipynb](notebooks/03_results_analysis.ipynb) | 结果分析：加载传统方法结果矩阵，按 Macro-F1 排名，生成特征×分类器透视表和热力图 |
| [deeplearningexplain.ipynb](notebooks/deeplearningexplain.ipynb) | 深度学习可解释性：Grad-CAM 可视化、注意力图分析 |
| [interpretation.ipynb](notebooks/interpretation.ipynb) | 模型解释分析 |
| [all_5_results_dashboard.ipynb](notebooks/all_5_results_dashboard.ipynb) | 全部 5 种 DL 模型结果仪表盘 |

## 方法概览

### 传统方法（3 特征 × 3 分类器 = 9 种组合）

| 特征提取 | 分类器 |
|----------|--------|
| HOG（方向梯度直方图） | Linear SVM |
| LBP（局部二值模式） | RBF SVM |
| SIFT + Bag of Visual Words | Random Forest |

**最佳传统流水线**: HOG + Linear SVM，测试集准确率 86.88%，Macro-F1 86.72%

### 深度学习方法（5 种架构）

- **ResNet18**（最佳: 94.72% 准确率, 92.50% Macro-F1）
- ResNet34
- ResNet50
- VGG16
- MobileNetV2

全部使用 ImageNet 预训练权重，AdamW 优化器，类别加权交叉熵损失，医学安全数据增强。

## 环境依赖

- Python 3.10+
- numpy, pandas, matplotlib, seaborn
- opencv-contrib-python（SIFT 支持）
- scikit-image, scikit-learn
- PyTorch, TorchVision
- tqdm, joblib, scipy, Pillow

## 快速开始

```bash
# 1. 审计数据集，生成划分清单
python main.py audit

# 2. 运行传统方法 3×3 实验矩阵
python main.py train-matrix

# 3. 运行深度学习实验（以 ResNet18 为例）
python main.py train-dl --model resnet18 --epochs 30 --batch-size 64

# 4. 汇总结果
python main.py collect-results

# 5. 绘制学习曲线（自动发现所有 DL 训练历史）
python main.py draw-learning

# 6. 绘制 ROC 曲线（基于真实预测概率分数）
python main.py draw-roc

# 7. 生成分析图表
python main.py analysis-artifacts

# 8. 导出预处理数据（供团队共享）
python main.py export-processed

# 可选：指定特定模型绘制
python main.py draw-learning --models resnet18 vgg16
python main.py draw-roc --models resnet18
```

## 关键设计决策

- **零患者泄漏**: 使用 `StratifiedGroupKFold` 按患者 ID 分组，训练/验证/测试集之间无重叠患者
- **严格评估协议**: `test_final` 被锁定，仅可通过 `--confirm-final-report` 标志使用，模型选择完全由 `val_final` 上的 Macro-F1 驱动
- **特征缓存**: 提取的特征缓存在 `data/interim/` 的 NPZ 文件中，加速传统方法迭代
- **产物组织**: 中间产物分别存放在 `outputs/runs/traditional/` 和 `outputs/runs/deeplearning/` 的结构化目录中
