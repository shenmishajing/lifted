import pandas as pd

data_root = "data/labeling/"
train_rf = pd.read_csv("data/labeling/vs_top/train_rf.csv")
# valid_rf = pd.read_csv("data/labeling/vs_top/valid_rf.csv")
# all_df = pd.concat([train_rf, valid_rf], ignore_index=True)
# all_df.to_csv("data/labeling/vs_top/all_rf.csv", index=False)

# for i,j in zip(['1', '2', '3'],["I","II","III"]):
#     phase_train_df = train_rf[train_rf['phase'].str.contains(i)]
#     phase_valid_df = valid_rf[valid_rf['phase'].str.contains(i)]
#     phase_train_df.to_csv(data_root + "phase_"+j+"_train.csv", index=False)
#     phase_valid_df.to_csv(data_root + "phase_"+j+"_valid.csv", index=False)

print(train_rf.columns.to_list())
