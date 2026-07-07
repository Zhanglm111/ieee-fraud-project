# 原始数据目录

请将 Kaggle IEEE-CIS Fraud Detection 数据集的原始 CSV 文件放在：

```text
data/raw/ieee-fraud-detection/
```

训练流程至少需要：

- `train_transaction.csv`
- `train_identity.csv`

如果需要使用官方测试集，也可以放入（但是测试集暂无标签）：

- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

原始数据体积较大，因此本目录下的数据文件会被 `.gitignore` 排除。
