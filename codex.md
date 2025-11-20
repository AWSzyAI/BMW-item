分析一下普遍分类错的案例都是什么问题，heatmap




超参网格（优先）：lr∈{3e-5,5e-5} × max_len∈{256,384} × epoch∈{5,8}，warmup_ratio=0.06，weight_decay=0.01，dropout=0.2。

打开 AMP + 梯度累积（有效 batch≥64）；eval 用小 batch，避免 OOM。