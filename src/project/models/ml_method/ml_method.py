# loading the zipped data
import numpy as np
import pandas as pd
import glob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

train_data_mode = "TOP"  # in ['CTO', 'TOP']

# we always test on supervised TOP labels
test_df = pd.concat(
    (
        pd.read_csv(f)
        for f in glob.glob("data/clinical-trial-outcome-prediction/data/phase*test.csv")
    )
)

if train_data_mode == "TOP":
    train_df = pd.concat(
        (
            pd.read_csv(f)
            for f in glob.glob(
                "data/clinical-trial-outcome-prediction/data/phase*train.csv"
            )
        )
    )
    valid_df = pd.concat(
        (
            pd.read_csv(f)
            for f in glob.glob(
                "data/clinical-trial-outcome-prediction/data/phase*valid.csv"
            )
        )
    )

elif train_data_mode == "CTO":
    train_df = pd.concat(
        (pd.read_csv(f) for f in glob.glob("data/labeling/phase*train.csv"))
    )
    valid_df = pd.concat(
        (pd.read_csv(f) for f in glob.glob("data/labeling/phase*valid.csv"))
    )

# ============ preprocess by filling NAs and dropping duplocates ============
train_df = pd.concat([train_df, valid_df])
train_df.fillna("", inplace=True)
train_df.drop_duplicates(subset=["nctid"], inplace=True)
test_df.fillna("", inplace=True)
test_df.drop_duplicates(subset=["nctid"], inplace=True)

# ============ set features to phase + diseases + icdcodes + drugs + inclusion / exclusion criteria ============
train_df["features"] = (
    train_df["diseases"]
    + " "
    + train_df["icdcodes"]
    + " "
    + train_df["drugs"]
    + " "
    + train_df["criteria"]
)
test_df["features"] = (
    test_df["diseases"]
    + " "
    + test_df["icdcodes"]
    + " "
    + test_df["drugs"]
    + " "
    + test_df["criteria"]
)

# featurize the data
tfidf = TfidfVectorizer(max_features=2048, stop_words="english")
X_train = tfidf.fit_transform(train_df["features"])
X_test = tfidf.transform(test_df["features"])


def bootstrap_eval(y_true, y_pred, y_prob, num_samples=100):
    f1s = []
    aps = []
    rocs = []
    for _ in range(num_samples):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        f1s.append(f1_score(y_true[indices], y_pred[indices]))
        aps.append(average_precision_score(y_true[indices], y_prob[indices]))
        rocs.append(roc_auc_score(y_true[indices], y_prob[indices]))
    return (
        np.mean(f1s),
        np.std(f1s),
        np.mean(aps),
        np.std(aps),
        np.mean(rocs),
        np.std(rocs),
    )


print("Phase, Model, F1, AP, ROC")
# for model_name in ['svm', 'xgboost', 'mlp', 'rf', 'lr', ]:
for model_name in ["svm", "lr", "xgboost", "rf"]:  # use fastest models for testing
    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=300, random_state=0, max_depth=10, n_jobs=4
        )
    elif model_name == "lr":
        model = LogisticRegression(max_iter=1000, random_state=0)
    elif model_name == "svm":
        model = LinearSVC(dual="auto", max_iter=10000, random_state=0)
        model = CalibratedClassifierCV(model)
        # model = SVC(kernel='linear', probability=True, random_state=0) # performs worse than the above
    elif model_name == "xgboost":
        model = XGBClassifier(n_estimators=300, random_state=0, max_depth=10, n_jobs=4)
    elif model_name == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(64, 64), max_iter=2000, random_state=0
        )
    else:
        raise ValueError("Unknown model name")

    model.fit(X_train, train_df["label"])
    test_df["pred"] = model.predict(X_test)
    test_df["prob"] = model.predict_proba(X_test)[:, 1]

    for phase in ["phase 1"]:  # , 'phase 2', 'phase 3']:
        test_df_subset = test_df[test_df["phase"].str.lower().str.contains(phase)]
        f1_mean, f1_std, ap_mean, ap_std, roc_mean, roc_std = bootstrap_eval(
            test_df_subset["label"].values,
            test_df_subset["pred"].values,
            test_df_subset["prob"].values,
        )
        print(
            f"{phase}, {model_name}, {f1_mean:.3f}, {f1_std:.3f}, {ap_mean:.3f}, {ap_std:.3f}, {roc_mean:.3f}, {roc_std:.3f}"
        )
