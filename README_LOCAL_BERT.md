# 使用本地下载的BERT模型

你已经成功使用 `modelscope download --model 'google-bert/bert-base-chinese' --local_dir './models'` 下载了BERT模型到本地。下面是如何使用这个本地模型的指南。

## 模型文件结构

下载的模型位于 `./models/` 目录，包含以下重要文件：
- `config.json` - 模型配置文件
- `tokenizer.json` - 分词器配置
- `vocab.txt` - 词汇表
- `pytorch_model.bin` - PyTorch模型权重

## 使用方法

### 方法1: 使用提供的示例脚本

#### 1. 基础使用示例
```bash
python test_local_bert.py
```
这个脚本演示了如何：
- 加载本地BERT模型和分词器
- 对文本进行预处理
- 进行预测
- 解析预测结果

#### 2. 集成到现有训练流程
```bash
python use_local_bert.py
```
这个脚本展示了如何：
- 使用现有的 `src/BERT.py` 代码
- 指定本地模型路径
- 进行模型训练

### 方法2: 在代码中直接使用

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载本地分词器
tokenizer = AutoTokenizer.from_pretrained("./models", local_files_only=True)

# 加载本地模型
model = AutoModelForSequenceClassification.from_pretrained(
    "./models",
    num_labels=3,  # 根据你的任务调整
    local_files_only=True,
    ignore_mismatched_sizes=True
)

# 使用模型进行预测
text = "这是一个测试文本"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

### 方法3: 修改现有训练脚本

如果你有自己的训练脚本，只需要修改以下参数：

```python
# 在你的训练脚本中，将模型路径指向本地目录
model_path = "./models"  # 你的本地模型目录

# 加载模型时指定 local_files_only=True
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    local_files_only=True,
    num_labels=num_classes,  # 你的类别数
    ignore_mismatched_sizes=True
)
```

## 关键参数说明

- `local_files_only=True`: 强制只使用本地文件，不尝试在线下载
- `ignore_mismatched_sizes=True`: 允许分类头尺寸不匹配（因为原始模型是预训练模型，不是分类模型）
- `num_labels`: 指定你的分类任务的类别数量

## 常见问题

### 1. 模型加载失败
确保以下文件存在于 `./models/` 目录：
- `config.json`
- `tokenizer.json` 或 `vocab.txt`
- `pytorch_model.bin`

### 2. 分类头尺寸错误
使用 `ignore_mismatched_sizes=True` 参数，这会自动重新创建适合你任务的分类头。

### 3. 内存不足
- 减小 `batch_size`
- 使用 `fp16=True`（如果有CUDA）
- 减小 `max_length`

### 4. 训练数据格式
确保你的训练数据包含以下列：
- `case_title`: 案例标题
- `performed_work`: 执行的工作
- `linked_items`: 链接的项目（标签）

## 示例数据格式

```csv
case_title,performed_work,linked_items
发动机异响,检查发动机,发动机故障
刹车失灵,更换刹车片,刹车系统故障
空调不制冷,添加制冷剂,空调系统故障
```

## 训练命令示例

```bash
# 使用本地模型进行训练
python src/BERT.py \
    --train-file train.csv \
    --eval-file eval.csv \
    --outdir ./output \
    --modelsdir ./models \
    --outmodel my_bert_model.joblib \
    --bert-model ./models \
    --allow-online False \
    --num-train-epochs 3 \
    --train-batch-size 16 \
    --max-length 256
```

## 注意事项

1. **离线使用**: 设置 `--allow-online False` 确保不会尝试在线下载
2. **模型路径**: 使用 `--bert-model ./models` 或 `--init-hf-dir ./models` 指定本地模型
3. **分类任务**: 原始BERT是预训练模型，需要根据你的分类任务进行微调
4. **GPU加速**: 如果有CUDA，BERT训练会自动使用GPU加速

## 下一步

1. 准备你的训练数据
2. 运行示例脚本测试模型加载
3. 根据你的任务调整模型参数
4. 进行模型训练和评估