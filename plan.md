# SwinIR 微调迁移指南 — Ubuntu 2×RTX 30

## TL;DR

继续在低压 SEM 图像上微调 SwinIR-M x4 超分辨率模型。已完成 10000 iter（PSNR=26.60 dB），需要从 10000 iter checkpoint 续训到 70000 iter。

---

## 环境要求

```bash
# 创建 conda 环境
conda create -n sr python=3.10
conda activate sr

# 安装 PyTorch (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装 BasicSR 依赖
cd BasicSR
pip install -r requirements.txt
pip install -e .  # 以 editable 模式安装 BasicSR

# 安装评估依赖
pip install lpips timm scikit-image
```

---

## 目录结构

```
workspace/
├── BasicSR/                    # 主仓库（本仓库）
│   ├── basicsr/
│   ├── options/train/SwinIR/
│   │   └── finetune_SwinIR_SRx4_SEM.yml  # 训练配置
│   ├── experiments/
│   │   └── finetune_SwinIR_SRx4_SEM/     # 自动创建
│   └── ...
├── SwinIR/                     # SwinIR 仓库（只需模型定义文件）
│   └── model_zoo/
│       └── 001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth  # 预训练权重
├── datasets/
│   ├── swinir_train/           # 训练数据（见下方准备步骤）
│   │   ├── HR/                 # ~19256 张 HR 图
│   │   ├── LR_x4/             # ~19256 张 4x bicubic LR 图
│   │   └── meta_info.txt
│   └── eval/
│       ├── SEMimg/             # 210 张低压 SEM HR 图
│       └── LR/                 # 210 张 4x bicubic LR 图
└── release_checkpoints/        # 从 release 下载的 checkpoint
    ├── net_g_10000.pth         # 10000 iter 模型权重
    ├── 10000.state             # 训练状态（optimizer + scheduler）
    └── 001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth  # 预训练权重
```

---

## 准备步骤

### 1. 克隆仓库

```bash
git clone <basic_sr_repo_url> -b cuda-sem-finetune
git clone <swinir_repo_url>
```

### 2. 下载 Release Checkpoint

从 GitHub Release 下载 `checkpoints_sem_sr.zip`，解压到：
- `SwinIR/model_zoo/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth`
- `BasicSR/experiments/finetune_SwinIR_SRx4_SEM/models/net_g_10000.pth`
- `BasicSR/experiments/finetune_SwinIR_SRx4_SEM/training_states/10000.state`

### 3. 准备训练数据

训练数据需要从原始 SEM 数据集生成。运行：

```bash
# 需要先准备原始 SEM 数据到 datasets/train/ 下
# 然后运行数据准备脚本（在 BasicSR 目录下）
python prepare_swinir_data.py
```

或者直接从之前的机器复制 `datasets/swinir_train/` 目录。

### 4. 准备评估数据

```bash
# 从原始 eval 数据生成 LR
python generate_eval_lr.py
```

或者直接复制 `datasets/eval/` 目录。

---

## 训练命令

```bash
cd BasicSR
python -m basicsr.train -opt options/train/SwinIR/finetune_SwinIR_SRx4_SEM.yml
```

**自动续训**：配置中 `auto_resume: true`，会自动从最新 checkpoint 恢复。

---

## 当前进度

| 指标 | 值 |
|------|-----|
| **已完成 iter** | 10000 / 70000 |
| **已完成 epoch** | ~2 / 15 |
| **当前 PSNR** | 26.60 dB (5k), 26.57 dB (10k) |
| **学习率** | 1e-4 (Adam) |
| **下个 milestone** | 30000 iter → lr 降至 5e-5 |
| **速度** | ~1.0s/iter (XPU batch4), 预计 CUDA batch8 更快 |

### 对比基线

| 模型 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| Bicubic | 25.38 | 0.6480 | 0.5918 |
| RealESRGAN_pretrained | 24.90 | 0.6065 | 0.5338 |
| SwinIR-L_real (预训练) | 24.67 | 0.6128 | 0.4875 |
| RealESRGAN_finetuned | 24.58 | 0.5641 | 0.3866 |
| **SwinIR-M_finetuned (5k)** | **26.60** | - | - |
| **SwinIR-M_finetuned (10k)** | **26.57** | - | - |

---

## 配置说明

训练配置在 `options/train/SwinIR/finetune_SwinIR_SRx4_SEM.yml`：

```yaml
# 关键参数
num_gpu: 2                    # 2×RTX 30
batch_size_per_gpu: 8         # 每卡 batch size（如果 OOM 可降为 4 或 6）
gt_size: 192                  # 训练 patch 大小
lr: 1e-4                      # 学习率
total_iter: 70000             # 总迭代数
save_checkpoint_freq: 5000    # 每 5000 iter 存档
val_freq: 5000                # 每 5000 iter 验证
```

**如果 OOM**：降低 `batch_size_per_gpu` 到 4 或 6，或降低 `gt_size` 到 128。

---

## 评估命令

训练完成后，评估微调模型：

```bash
# 用 BasicSR 的 inference 脚本
python inference/inference_swinir.py \
    --task classical_sr \
    --scale 4 \
    --training_patch_size 48 \
    --model_path experiments/finetune_SwinIR_SRx4_SEM/models/net_g_best.pth \
    --folder_lq ../datasets/eval/LR \
    --folder_gt ../datasets/eval/SEMimg \
    --output ../outputs/eval/SwinIR-M_finetuned
```

然后用评估脚本计算指标：

```bash
python ../LVSEM-ESRGAN/validate/calc_sr_metrics.py \
    --hr_dir ../datasets/eval/SEMimg \
    --lr_dir ../datasets/eval/LR \
    --sr_dir ../outputs/eval/SwinIR-M_finetuned \
    --suffix out
```

---

## 已修复的 BasicSR 问题

本分支包含以下修复（相对于 upstream master）：

1. **中文路径兼容** (`basicsr/utils/options.py`): YAML 读写添加 `encoding='utf-8'`
2. **中文路径写图** (`basicsr/utils/img_util.py`): `imwrite` 添加 temp+move fallback
3. **auto_resume 归档目录** (`basicsr/train.py`): 自动检查 `_archived_` 目录中的 checkpoint
4. **auto_resume 配置覆盖** (`basicsr/utils/options.py`): 命令行默认值不再覆盖 YAML 配置

---

## 第二轮训练计划（可选）

本轮训练结束后，可以尝试：

1. **去除标注文字** — 裁掉 SEM 图中的批注区域，避免模型学超分文字
2. **自定义退化模型** — 用 RealESRGAN 风格的在线退化替代 bicubic，更接近真实 LVSEM 退化
3. **学习率调度** — 尝试 cosine annealing 替代 multistep
4. **混合损失** — L1 + perceptual loss（VGG）提升感知质量

---

## 关键文件索引

| 文件 | 用途 |
|------|------|
| `options/train/SwinIR/finetune_SwinIR_SRx4_SEM.yml` | 训练配置 |
| `basicsr/train.py` | 训练入口（含 auto_resume 修复） |
| `basicsr/utils/options.py` | 配置解析（含中文路径修复） |
| `basicsr/utils/img_util.py` | 图像 IO（含中文路径 fallback） |
| `basicsr/archs/swinir_arch.py` | SwinIR 网络定义 |
| `basicsr/models/sr_model.py` | SR 训练模型 |

---

## 注意事项

1. **checkpoint 设备兼容**：checkpoint 是在 XPU 上保存的，加载时 BasicSR 会自动 `map_location` 到 CUDA，不需要手动转换
2. **多卡训练**：BasicSR 使用 `DistributedDataParallel`，2×RTX 30 会自动分配 batch
3. **TensorBoard**：训练日志在 `tb_logger/finetune_SwinIR_SRx4_SEM/`，可用 `tensorboard --logdir tb_logger` 查看
4. **中途停止**：随时 Ctrl+C 停止，下次运行同样命令会自动从最新 checkpoint 续训
