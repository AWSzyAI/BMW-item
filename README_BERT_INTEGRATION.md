### 1. 确保本地BERT模型已下载

```bash
# 检查模型是否存在
ls -la models/

# 如果不存在，下载模型
modelscope download --model 'google-bert/bert-base-chinese' --local_dir './models'
```

### 2. 准备训练数据

确保你的数据目录包含以下文件：
- `train.csv` - 训练数据
- `eval.csv` - 评估数据

数据格式示例：
```csv
case_title,performed_work,linked_items
发动机异响,检查发动机,发动机故障
刹车失灵,更换刹车片,刹车系统故障
空调不制冷,添加制冷剂,空调系统故障
```

### 3. 训练BERT模型

```bash
# 基本训练命令
python src/train_bert.py \
    --bert-model ./models \
    --allow-online False \
    --num-train-epochs 3.0 \
    --train-batch-size 16 \
    --eval-batch-size 32 \
    --max-length 256

# 高级训练命令（带早停和不平衡处理）
python src/train_bert.py \
    --bert-model ./models \
    --allow-online False \
    --num-train-epochs 10.0 \
    --train-batch-size 16 \
    --eval-batch-size 32 \
    --max-length 256 \
    --early-stopping-patience 3 \
    --learning-rate 5e-5 \
    --resample-method ros \
    --outmodel bert_model.joblib
```

### 4. 使用模型进行预测

```bash
# 单样本预测（两种入口等价，predict.py 现已支持 BERT 训练产生的 .joblib）
python src/predict.py \
    --modelsdir ./models \
    --model bert_model.joblib \
    --outdir ./output \
    --infile eval.csv

# 或使用专用入口
python src/predict_bert.py \
    --modelsdir ./models \
    --model bert_model.joblib \
    --outdir ./output \
    --infile eval.csv
```

```python
# 批量预测（在Python代码中）
from src.predict import predict  # predict.py 同样支持 BERT/TF-IDF 两种 bundle

results = predict(
    texts=["发动机有异响", "刹车不灵敏"], 
    model_path="./models/bert_model.joblib",
    top_k=5
)
print(results)
```

### 5. 评估模型性能

```bash
# 基本评估
python src/eval_bert.py \
    --modeldir ./models \
    --model bert_model.joblib \
    --outdir ./output \
    --path eval.csv \
    --mode new

# 开放集评估（带拒判阈值）
python src/eval_bert.py \
    --modeldir ./models \
    --model bert_model.joblib \
    --outdir ./output \
    --path eval.csv \
    --mode new \
    --reject-threshold 0.7

# 阈值扫描评估
python src/eval_bert.py \
    --modeldir ./models \
    --model bert_model.joblib \
    --outdir ./output \
    --path eval.csv \
    --mode new \
    --sweep-thresholds "0.5:0.9:0.05"
```

## 🔧 参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bert-model` | `./models` | BERT模型路径或名称 |
| `--allow-online` | `False` | 是否允许在线下载模型 |
| `--num-train-epochs` | `3.0` | 训练轮数 |
| `--train-batch-size` | `16` | 训练批次大小 |
| `--eval-batch-size` | `32` | 评估批次大小 |
| `--max-length` | `256` | 最大序列长度 |
| `--learning-rate` | `5e-5` | 学习率 |
| `--early-stopping-patience` | `3` | 早停耐心值 |
| `--resample-method` | `none` | 不平衡处理方法 |
| `--fp16` | `False` | 是否启用混合精度训练 |

### 预测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--modelsdir` | `./models` | 模型目录 |
| `--model` | `7.joblib` | 模型文件名 |
| `--outdir` | `./output/2025_up_to_month_7` | 输出目录 |
| `--infile` | `eval.csv` | 输入文件 |
| `--reject-threshold` | `None` | 拒判阈值 |

### 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `new` | 评估模式：new/clean/dirty |
| `--reject-threshold` | `None` | 拒判阈值 |
| `--sweep-thresholds` | `None` | 阈值扫描范围 |
| `--unknown-policy` | `tag-not-in-train` | 未知标签处理策略 |

## 📊 输出文件说明

### 训练输出

- `loss_curve.png` - 训练损失曲线图
- `loss_data.json` - 损失数据（JSON格式）
- `metrics_eval.csv` - 评估指标
- `log/{model}_train.txt` - 训练日志
- `models/{model}_bert/` - BERT模型目录
- `models/{model}.joblib` - 兼容格式的模型文件

### 评估输出

- `predictions_{file}.csv` - 逐样本预测结果
- `threshold_sweep.csv` - 阈值扫描结果（如果启用）
- `metrics_best_model_all_splits.csv` - 所有分割的评估指标

## 🔄 与原有TF-IDF流程的兼容性

### 数据格式兼容

BERT版本完全兼容原有的数据格式：
- 支持X/Y分离文件（`train_X.csv` + `train_y.csv`）
- 支持单表文件（`train.csv`）
- 支持多种标签列名（`linked_items`, `extend_id`, `item_title`）

### 接口兼容

所有脚本都保持与原有版本相同的命令行接口：
```bash
# TF-IDF版本
python src/train.py --train-file train.csv --eval-file eval.csv
python src/predict.py --model 7.joblib
python src/eval.py --model 7.joblib

# BERT版本（完全兼容）
python src/train_bert.py --train-file train.csv --eval-file eval.csv
python src/predict_bert.py --model bert_model.joblib
python src/eval_bert.py --model bert_model.joblib
```

### 输出格式兼容

- 模型文件使用相同的joblib格式
- 评估指标CSV格式保持一致
- 预测结果CSV格式保持一致

## 🎯 性能优化建议

### 1. 硬件优化

```bash
# 启用GPU训练（如果有CUDA）
python src/train_bert.py --fp16 --train-batch-size 32

# 使用更大的批次大小（如果内存充足）
python src/train_bert.py --train-batch-size 64 --eval-batch-size 128
```

### 2. 训练策略优化

```bash
# 使用早停避免过拟合
python src/train_bert.py --early-stopping-patience 5

# 处理类别不平衡
python src/train_bert.py --resample-method smote

# 调整学习率
python src/train_bert.py --learning-rate 3e-5 --weight-decay 0.01
```

### 3. 序列长度优化

```bash
# 根据数据特点调整最大长度
python src/train_bert.py --max-length 128  # 短文本
python src/train_bert.py --max-length 512  # 长文本
```

## 🐛 常见问题

### 1. 模型加载失败

**问题**：`ModuleNotFoundError: No module named 'transformers'`

**解决**：
```bash
pip install transformers torch
```

### 2. 内存不足

**问题**：`CUDA out of memory`

**解决**：
```bash
# 减小批次大小
python src/train_bert.py --train-batch-size 8 --eval-batch-size 16

# 启用梯度累积
python src/train_bert.py --train-batch-size 8 --grad-accum-steps 4
```

### 3. 训练速度慢

**问题**：训练时间过长

**解决**：
```bash
# 启用混合精度
python src/train_bert.py --fp16

# 使用多GPU（如果可用）
export CUDA_VISIBLE_DEVICES=0,1
python src/train_bert.py
```

### 4. 预测结果不一致

**问题**：BERT预测结果与TF-IDF不同

**说明**：这是正常的，BERT和TF-IDF是不同的模型架构，预测结果会有差异。BERT通常在理解语义方面表现更好。

## 📈 性能对比

| 指标 | TF-IDF | BERT | 提升 |
|------|--------|------|------|
| 准确率 | ~85% | ~92% | +7% |
| F1-macro | ~82% | ~90% | +8% |
| Hit@3 | ~88% | ~95% | +7% |

*注：以上为示例数据，实际性能取决于具体任务和数据质量。*

## 🔮 高级用法

### 1. 自定义模型架构

```python
# 修改train_bert.py中的模型加载部分
from transformers import AutoModelForSequenceClassification, AutoConfig

config = AutoConfig.from_pretrained(
    "./models",
    num_labels=num_labels,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)
model = AutoModelForSequenceClassification.from_pretrained(
    "./models",
    config=config
)
```

### 2. 自定义训练回调

```python
# 添加自定义回调
class CustomCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # 自定义逻辑
        pass

# 在训练中使用
trainer.add_callback(CustomCallback())
```

### 3. 模型集成

```python
# 结合BERT和TF-IDF的预测
from src.train_bert import BERTModelWrapper
from src.train import main as tfidf_train

# 训练两个模型
# ...

# 集成预测
def ensemble_predict(texts):
    bert_probs = bert_model.predict_proba(texts)
    tfidf_probs = tfidf_model.predict_proba(texts)
    # 加权平均
    ensemble_probs = 0.7 * bert_probs + 0.3 * tfidf_probs
    return ensemble_probs
```

## 📚 参考资料

- [Hugging Face Transformers文档](https://huggingface.co/docs/transformers/)
- [BERT论文](https://arxiv.org/abs/1810.04805)
- [文本分类最佳实践](https://huggingface.co/docs/transformers/tasks/sequence_classification)

## 🤝 贡献指南

如果你发现问题或有改进建议，请：

1. 检查现有的Issues
2. 创建新的Issue描述问题
3. 提交Pull Request

---

🎉 **恭喜！** 你现在已经掌握了BERT模型集成的完整使用方法。开始你的文本分类之旅吧！