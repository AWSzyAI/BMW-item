import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os
import argparse

# 仅本地模式（无网络）
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

model_path = 'models/google-bert/bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=3,  # 假设3分类任务，根据你的实际任务调整
        local_files_only=True,
        ignore_mismatched_sizes=True  # 允许分类头尺寸不匹配
    )
device = torch.device(
    "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )
)
model.to(device)


# Load data
def load_data(outdir: str):
    X_train = pd.read_csv(os.path.join(outdir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(outdir, 'y_train.csv'))
    X_test = pd.read_csv(os.path.join(outdir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(outdir, 'y_test.csv'))
    label_mapping = pd.read_csv(os.path.join(outdir, 'label_mapping.csv'))
    
    # 期望列：y_* 至少包含 linked_items；若存在 item_title 列则一并使用
    required_cols = ['linked_items']
    for name, y in [('y_train', y_train), ('y_test', y_test)]:
        for col in required_cols:
            if col not in y.columns:
                raise ValueError(
                    f"{name} 缺少必需列: {col}，当前列为: {list(y.columns)}"
                )

    # 如果缺少 label 列，则基于 label_mapping 或数据联合集生成
    if 'label' not in y_train.columns or 'label' not in y_test.columns:
        # 标准化为字符串，避免类别不一致
        y_train_li = y_train['linked_items'].astype(str)
        y_test_li = y_test['linked_items'].astype(str)

        mapping_df = None
        if isinstance(label_mapping, pd.DataFrame):
            cols = set(label_mapping.columns)
            if {'linked_items', 'label'}.issubset(cols):
                # 使用现有映射
                mapping_df = label_mapping[['linked_items', 'label']].copy()
                mapping_df['linked_items'] = mapping_df['linked_items'].astype(str)
                mapping_df['label'] = mapping_df['label'].astype(int)
            elif 'linked_items' in cols and 'label' not in cols:
                # 只有类别名，没有数值 id，则按出现顺序生成
                unique_items = label_mapping['linked_items'].astype(str).unique().tolist()
                mapping_df = pd.DataFrame({
                    'linked_items': unique_items,
                    'label': list(range(len(unique_items)))
                })

        # 如果 mapping_df 仍为空，则基于训练+测试构建
        if mapping_df is None:
            all_items = pd.Index(y_train_li).append(pd.Index(y_test_li)).unique().tolist()
            mapping_df = pd.DataFrame({
                'linked_items': all_items,
                'label': list(range(len(all_items)))
            })

        # 覆盖不全时，补全未出现的类别
        known = set(mapping_df['linked_items'].astype(str))
        union_items = pd.Index(y_train_li).append(pd.Index(y_test_li)).unique().tolist()
        missing = [it for it in union_items if it not in known]
        if missing:
            start_id = int(mapping_df['label'].max()) + 1 if not mapping_df.empty else 0
            mapping_df = pd.concat([
                mapping_df,
                pd.DataFrame({'linked_items': missing, 'label': list(range(start_id, start_id + len(missing)))})
            ], ignore_index=True)

        # 应用映射，确保为 int 类型
        map_dict = dict(zip(mapping_df['linked_items'].astype(str), mapping_df['label'].astype(int)))
        y_train['label'] = y_train_li.map(map_dict)
        y_test['label'] = y_test_li.map(map_dict)

        # 检查是否仍有缺失映射
        if y_train['label'].isna().any() or y_test['label'].isna().any():
            raise ValueError("无法为部分 linked_items 生成 label，请检查数据与映射是否一致。")

        y_train['label'] = y_train['label'].astype(int)
        y_test['label'] = y_test['label'].astype(int)
        # 用扩展后的映射替换 label_mapping，后续用到类别数
        label_mapping = mapping_df.copy()

    # 若存在 item_title 列，为 label_mapping 增补 item_title（按出现频次最高的一个或保持原有）
    if 'item_title' in y_train.columns or 'item_title' in y_test.columns:
        # 合并 train+test 中的 (linked_items, item_title)
        concat_df = []
        if 'item_title' in y_train.columns:
            concat_df.append(y_train[['linked_items', 'item_title']].dropna())
        if 'item_title' in y_test.columns:
            concat_df.append(y_test[['linked_items', 'item_title']].dropna())
        if concat_df:
            li_it_df = pd.concat(concat_df, ignore_index=True)
            # 统计每个 linked_items 下最常见的 item_title
            freq_df = (
                li_it_df.groupby('linked_items')['item_title']
                .agg(lambda s: s.value_counts().index[0])  # 取出现次数最高的
                .reset_index()
            )
            # 若 label_mapping 已经有 item_title 列则只填充缺失；否则新增
            if 'item_title' in label_mapping.columns:
                # 仅对缺失值进行填充
                lm = label_mapping.merge(freq_df, on='linked_items', how='left', suffixes=('', '_freq'))
                lm['item_title'] = lm['item_title'].fillna(lm['item_title_freq'])
                lm = lm.drop(columns=['item_title_freq'])
                label_mapping = lm
            else:
                label_mapping = label_mapping.merge(freq_df, on='linked_items', how='left')
        # 若合并后仍不存在 item_title 列，则构造空列方便后续统一处理
        if 'item_title' not in label_mapping.columns:
            label_mapping['item_title'] = ''
    else:
        # 保证结构稳定，若需要也可添加空列
        if 'item_title' not in label_mapping.columns:
            label_mapping['item_title'] = ''

    return X_train, y_train, X_test, y_test, label_mapping

# 兼容原单任务数据集（保留）
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# CoT 多任务数据集：输入 case_title + performed_work，输出两种标签
class CoTDataset(Dataset):
    def __init__(self, X_df, labels_linked, labels_item, tokenizer, max_length=128):
        self.texts = (
            X_df.iloc[:, 0].astype(str) + ' [SEP] ' + X_df.iloc[:, 1].astype(str)
        ).tolist()
        self.labels_linked = labels_linked
        self.labels_item = labels_item
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_linked = int(self.labels_linked[idx])
        label_item = int(self.labels_item[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels_linked': torch.tensor(label_linked, dtype=torch.long),
            'labels_item': torch.tensor(label_item, dtype=torch.long)
        }

# 多任务 BERT：同时预测 linked_items 与 item_title
class MultiTaskBert(nn.Module):
    def __init__(self, model_path, num_labels_linked, num_labels_item):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path, local_files_only=True)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier_linked = nn.Linear(hidden, num_labels_linked)
        self.classifier_item = nn.Linear(hidden, num_labels_item)

    def forward(self, input_ids, attention_mask, labels_linked=None, labels_item=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        logits_linked = self.classifier_linked(cls)
        logits_item = self.classifier_item(cls)

        loss = None
        if labels_linked is not None and labels_item is not None:
            ce = nn.CrossEntropyLoss()
            loss = ce(logits_linked, labels_linked) + 0.3 * ce(logits_item, labels_item)
        return {
            'loss': loss,
            'logits_linked': logits_linked,
            'logits_item': logits_item
        }

# 训练（CoT）
def train_model_cot(model, train_loader, val_loader, optimizer, num_epochs=5, device='cuda'):
    model.train()
    train_losses, val_losses, val_acc_linked = [], [], []
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        model.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_linked = batch['labels_linked'].to(device)
            labels_item = batch['labels_item'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels_linked=labels_linked, labels_item=labels_item)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item())
        avg_train = total_train_loss / max(len(train_loader), 1)
        train_losses.append(avg_train)

        # 验证
        model.eval()
        total_val_loss = 0.0
        all_preds_linked, all_labels_linked = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_linked = batch['labels_linked'].to(device)
                labels_item = batch['labels_item'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                labels_linked=labels_linked, labels_item=labels_item)
                loss = outputs['loss']
                total_val_loss += float(loss.item())
                preds_linked = torch.argmax(outputs['logits_linked'], dim=1)
                all_preds_linked.extend(preds_linked.cpu().numpy())
                all_labels_linked.extend(labels_linked.cpu().numpy())
        avg_val = total_val_loss / max(len(val_loader), 1)
        val_losses.append(avg_val)
        val_acc = accuracy_score(all_labels_linked, all_preds_linked) if all_labels_linked else 0.0
        val_acc_linked.append(val_acc)
        print(f'Epoch {epoch+1}: Train {avg_train:.4f} | Val {avg_val:.4f} | ValAcc(linked) {val_acc:.4f}')
    return train_losses, val_losses, val_acc_linked

# 测试（主任务）
def evaluate_model_cot(model, test_loader, device='cuda'):
    model.eval()
    all_preds_linked, all_labels_linked = [], []
    all_preds_item, all_labels_item = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_linked = batch['labels_linked'].to(device)
            labels_item = batch['labels_item'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels_linked=labels_linked, labels_item=labels_item)
            preds_linked = torch.argmax(outputs['logits_linked'], dim=1)
            preds_item = torch.argmax(outputs['logits_item'], dim=1)
            all_preds_linked.extend(preds_linked.cpu().numpy())
            all_labels_linked.extend(labels_linked.cpu().numpy())
            all_preds_item.extend(preds_item.cpu().numpy())
            all_labels_item.extend(labels_item.cpu().numpy())
    acc_linked = accuracy_score(all_labels_linked, all_preds_linked) if all_labels_linked else 0.0
    acc_item = accuracy_score(all_labels_item, all_preds_item) if all_labels_item else 0.0
    print(f'Test Accuracy (linked_items): {acc_linked:.4f}')
    print(f'Test Accuracy (item_title):  {acc_item:.4f}')
    return acc_linked, acc_item, all_preds_linked, all_labels_linked



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='output/2025_up_to_month_2', help='数据输出目录（包含 X_*.csv / y_*.csv / label_mapping.csv）')
    args = parser.parse_args()
    # Load data
    print("Loading data...")
    X_train, y_train, X_test, y_test, label_mapping = load_data(args.outdir)

    print(f"Train data shape: {X_train.shape}")
    print(f"Train labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Number of unique linked_items: {len(label_mapping)}")
    if 'item_title' in y_train.columns:
        print(f"Number of unique item_title (train): {y_train['item_title'].nunique()}")

    # Extract labels
    train_labels_linked = y_train['label'].astype(int).tolist()
    test_labels_linked = y_test['label'].astype(int).tolist()

    # Build item_title encoder for CoT
    if 'item_title' in y_train.columns:
        item_le = LabelEncoder()
        y_train_item_ids = item_le.fit_transform(y_train['item_title'].astype(str).fillna(''))
        # safe transform for test
        classes_set = set(item_le.classes_)
        y_test_item_ids = [int(item_le.transform([v])[0]) if str(v) in classes_set else 0 for v in y_test['item_title'].astype(str)]
        num_item_classes = len(item_le.classes_)
    else:
        # 若不存在 item_title，则退化为单任务（item 分支只有 1 类）
        y_train_item_ids = [0] * len(y_train)
        y_test_item_ids = [0] * len(y_test)
        num_item_classes = 1

    # Compose input texts: case_title + performed_work
    train_texts = (X_train.iloc[:, 0].astype(str) + ' [SEP] ' + X_train.iloc[:, 1].astype(str)).tolist()
    test_texts = (X_test.iloc[:, 0].astype(str) + ' [SEP] ' + X_test.iloc[:, 1].astype(str)).tolist()

    # Device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else (
            'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'
        )
    )
    print(f"Using device: {device}")

    # Load local tokenizer and build multitask model
    local_model_path = './models/google-bert/bert-base-chinese'
    if not os.path.isdir(local_model_path):
        raise FileNotFoundError(f"未找到本地模型目录: {local_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    num_linked_classes = len(label_mapping)
    model = MultiTaskBert(local_model_path, num_labels_linked=num_linked_classes, num_labels_item=num_item_classes)
    model = model.to(device)

    # Datasets & loaders
    batch_size = 16
    max_length = 128
    train_dataset = CoTDataset(X_train, train_labels_linked, y_train_item_ids, tokenizer, max_length)
    test_dataset = CoTDataset(X_test, test_labels_linked, y_test_item_ids, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Train
    print("\nTraining model (CoT)...")
    num_epochs = 5
    train_losses, val_losses, val_accuracies = train_model_cot(
        model, train_loader, test_loader, optimizer, num_epochs=num_epochs, device=device
    )

    # Plot curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training/Validation Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Acc (linked)')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Validation Accuracy (linked)'); plt.legend()
    plt.tight_layout(); plt.savefig('bert_training_history_cot.png'); plt.show()

    # Evaluate
    print("\nEvaluating on test set...")
    acc_linked, acc_item, preds_linked, labels_linked = evaluate_model_cot(model, test_loader, device=device)

    # Save model and tokenizer
    save_path = 'bert_chinese_classifier_local_cot'
    os.makedirs(save_path, exist_ok=True)
    # 保存 backbone + heads
    try:
        # 仅保存 backbone 权重
        model.bert.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    except Exception:
        pass

    print(f"\nModel backbone and tokenizer saved to '{save_path}'")

    # Confusion matrix for linked_items (top 20)
    from collections import Counter
    # 使用完整标签空间，避免测试集中未出现类别导致的索引错位
    all_labels = list(range(num_linked_classes))
    cm = confusion_matrix(labels_linked, preds_linked, labels=all_labels)
    # 依据真实标签在测试集中的出现频次，选取前20个最常见类别
    counts = np.bincount(np.array(labels_linked, dtype=int), minlength=num_linked_classes)
    k = min(20, num_linked_classes)
    top_classes = np.argsort(counts)[-k:]
    cm_top = cm[np.ix_(top_classes, top_classes)]

    # 使用原始 linked_items 名称作为坐标轴标签
    id2name = None
    try:
        if isinstance(label_mapping, pd.DataFrame) and {'label', 'linked_items'}.issubset(set(label_mapping.columns)):
            id2name = dict(zip(label_mapping['label'].astype(int), label_mapping['linked_items'].astype(str)))
    except Exception:
        id2name = None
    if id2name is None:
        id2name = {i: str(i) for i in range(num_linked_classes)}
    # 截断过长标签，避免拥挤；保留前 30 个字符
    tick_labels = [id2name.get(int(i), str(i))[:30] for i in top_classes]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_top, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title('Confusion Matrix (Top Classes by Frequency, linked_items)')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('bert_confusion_matrix_cot.png'); plt.show()

    print(f"Test Accuracy (linked_items): {acc_linked:.4f}")
    print(f"Test Accuracy (item_title):  {acc_item:.4f}")

    return model, acc_linked


if __name__ == '__main__':
    model, accuracy = main()
    print(f"\nFinal test accuracy: {accuracy:.4f}")