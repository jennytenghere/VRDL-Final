import pandas as pd

# 讀取 CSV
train_df = pd.read_csv("./train.csv")  # 包含 image_id, label
val_df = pd.read_csv("./validation_data.csv")  # 包含 image_id, fold

# 合併（根據 image_id）
merged_df = pd.merge(train_df, val_df, on="image_id")

# 儲存成新的 data.csv
merged_df.to_csv("data.csv", index=False)