# 原始交易数据目录

本目录用于存放在线交易欺诈风险建模所需的原始 CSV 数据文件。

项目当前实验使用 IEEE-CIS Fraud Detection 数据格式。请将原始文件放在：

```text
data/raw/ieee-fraud-detection/
```

流程至少需要：

- `train_transaction.csv`
- `train_identity.csv`

如果需要使用官方测试集，也可以放入（但是测试集无标签）：

- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

原始数据体积较大，因此本目录下的数据文件会被 `.gitignore` 排除。
