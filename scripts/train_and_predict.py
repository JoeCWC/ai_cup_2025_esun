import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib

# =========================
# 0. 設定 input 檔案路徑
# =========================
acct_transaction_csv =  "/home/joe/ai_cup_2025/bank/data_set/acct_transaction.csv"
acct_alert_csv =        "/home/joe/ai_cup_2025/bank/data_set/acct_alert.csv"
acct_predict_csv =      "/home/joe/ai_cup_2025/bank/data_set/acct_predict.csv"

# =========================
# 0. 設定 output 檔案路徑
# =========================
acct_predict_result_csv =   "/home/joe/ai_cup_2025/bank/trained_model_and_prediction/acct_predict_result.csv"
saved_model_path =          "/home/joe/ai_cup_2025/bank/trained_model_and_prediction/xgb_acctlevel_model.joblib"

# =========================
# 0. 檢查 input 檔案是否存在，不存在就退出
# =========================
for path in [acct_transaction_csv, acct_alert_csv, acct_predict_csv]:
    if not os.path.exists(path):
        print(f"[ERROR] 找不到必要的輸入檔案: {path}")
        sys.exit(1)  # 直接退出程式

# =========================
# 0. 檢查 output 目錄是否存在，若不存在則建立
# =========================
for path in [acct_predict_result_csv, saved_model_path]:
    out_dir = os.path.dirname(path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"[INFO] 建立目錄: {out_dir}")

# =========================
# 1. 載入資料
# =========================
print("[INFO] 載入資料...")
txn_df = pd.read_csv(acct_transaction_csv)
alert_df = pd.read_csv(acct_alert_csv)

# =========================
# 2. 建立帳戶層級特徵

# 「帳戶層級特徵表」
# - 時間特徵（交易頻率、夜間比例）
# - 金額特徵（平均、最大、標準差）
# - 網路特徵（出度、入度、對手數量）

# 透過 時間線拆解 + 金額分析 + 交易對手網路，去找出「未來可能被判定為警示」的帳戶。這個需求其實就是 反洗錢 (AML) / 可疑交易偵測 的典型任務。

# 時間線拆解 (Temporal Features)
# - 交易頻率：計算帳戶在不同時間窗內的交易次數（近 1 天、7 天、30 天）。
# - 交易間隔：平均交易間隔時間、最短/最長間隔。
# - 時間分布：夜間交易比例（22:00–06:00）、尖峰時段交易比例。
# - 異常模式：短時間內多筆大額交易、連續轉帳到不同帳戶。
# 👉 這些特徵能捕捉「洗錢常見的時間異常行為」

# 金額分析 (Amount Features)
# - 統計特徵：平均金額、最大金額、最小金額、標準差。
# - 大額比例：大於某門檻（例如 50 萬）的交易比例。
# - 金額分布：小額高頻 vs. 大額低頻。
# - 幣別異常：是否頻繁使用外幣（USD、JPY…）轉帳。
# 👉 這些特徵能捕捉「小額切割 (structuring)」或「大額異常」的模式

# 交易對手網路 (Graph Features)
# - 出度 (out-degree)：該帳戶轉出給多少不同帳戶。
# - 入度 (in-degree)：該帳戶收過多少不同帳戶。
# - 雙向交易比例：是否存在「互相轉帳」的關係。
# - 與警示帳戶的關聯：是否直接或間接與已知警示帳戶有交易。
# - 社群特徵：透過圖演算法（PageRank、Connected Components）找出可疑集團。
# 👉 這些特徵能捕捉「人頭帳戶集團」或「資金洗白網路」

# 不合理交易模式的判斷邏輯
# 除了模型預測，你也可以設計 規則檢測 (rule-based features)，例如：
# - 單日交易次數 > 50 且金額總和 > 100 萬。
# - 夜間交易比例 > 80%。
# - 出度 > 20 且交易對手多為新帳戶。
# - 與已知警示帳戶有直接交易。
# 這些規則可以和 XGBoost 模型結合，形成 Hybrid System，提升可解釋性。
# =========================
print("[INFO] 建立帳戶層級特徵...")

# (a) 金額特徵
amt_stats = txn_df.groupby("from_acct")["txn_amt"].agg(
    txn_amt_mean="mean",
    txn_amt_max="max",
    txn_amt_std="std",
    txn_count="count"
).reset_index().rename(columns={"from_acct": "acct"})

# (b) 時間特徵
txn_df["txn_hour"] = pd.to_datetime(txn_df["txn_time"], format="%H:%M:%S", errors="coerce").dt.hour
txn_df["is_night"] = txn_df["txn_hour"].apply(lambda h: 1 if pd.notnull(h) and (h < 6 or h >= 22) else 0)

time_stats = txn_df.groupby("from_acct").agg(
    night_ratio=("is_night", "mean"),
    txn_per_day=("txn_date", lambda x: len(x) / (x.max() - x.min() + 1))
).reset_index().rename(columns={"from_acct": "acct"})

# (c) 網路特徵
out_degree = txn_df.groupby("from_acct")["to_acct"].nunique().reset_index()
out_degree = out_degree.rename(columns={"from_acct": "acct", "to_acct": "out_degree"})

in_degree = txn_df.groupby("to_acct")["from_acct"].nunique().reset_index()
in_degree = in_degree.rename(columns={"to_acct": "acct", "from_acct": "in_degree"})

# (d) 合併所有特徵
acct_features = amt_stats.merge(time_stats, on="acct", how="left")
acct_features = acct_features.merge(out_degree, on="acct", how="left")
acct_features = acct_features.merge(in_degree, on="acct", how="left")

# 缺失值補 0
acct_features = acct_features.fillna(0)

# =========================
# 3. 建立標籤
# =========================
print("[INFO] 建立標籤...")
alert_df["label"] = 1
acct_features = acct_features.merge(alert_df[["acct", "label"]], on="acct", how="left")
acct_features["label"] = acct_features["label"].fillna(0).astype(int)

# =========================
# 4. 切分資料
# =========================
print("[INFO] 切分資料...")
X = acct_features.drop(columns=["acct", "label"])
y = acct_features["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# 5. 建立 pipeline
# =========================
print("[INFO] 建立前處理 pipeline...")
numeric_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features)
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor)
])

# =========================
# 6. 前處理
# =========================
print("[INFO] 前處理資料...")
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)


# =========================
# 7. 處理樣本不平衡：scale_pos_weight
#   設為 (負樣本數 / 正樣本數)
# =========================
neg, pos = np.bincount(y_train)
scale_pos_weight = (neg / pos) if pos > 0 else 1.0
print(f"正樣本數={pos}, 負樣本數={neg}, scale_pos_weight={scale_pos_weight:.2f}")


# =========================
# 8. 訓練 XGBoost
# =========================
model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,  # 類別不平衡處理，通常設為 neg/pos，讓少數類別權重更高
    max_depth=5,                        # 每棵樹的最大深度，控制模型複雜度，過大容易 overfitting，過小可能 underfitting，常見範圍：3–8
    learning_rate=0.02,                 # 學習率 (shrinkage)，步伐小，收斂慢但更穩定，但需要更多樹 (n_estimators)
                                        # 通常固定一個較小的 learning_rate（如 0.05），再用 early_stopping_rounds 自動挑選最佳樹數。
    n_estimators=5000,                  # 最大樹數 (boosting rounds)，搭配 early stopping
    subsample=0.9,                      # 控制每棵樹用多少樣本，<1.0 時能增加隨機性，降低過擬合
    colsample_bytree=0.9,               # 控制每棵樹用多少特徵，和 subsample 搭配調整，常見組合：0.7–0.9
    reg_lambda=1.5,                     # L2 正則化 (Ridge)，抑制權重過大，提升模型穩定性
    reg_alpha=0.3,                      # L1 正則化 (Lasso)，鼓勵稀疏性，有助於特徵選擇
    early_stopping_rounds=100,          # 給模型充足時間停止 設為 None 可關閉 early stopping，完整觀察 learning curve
    eval_metric="logloss",              # 評估指標（logloss、auc、aucpr），對極度不平衡的資料，aucpr 通常比 logloss 更敏感
    objective="binary:logistic",        # 二元分類
    #tree_method="gpu_hist",            # 如果資料量大， tree_method="hist"，能顯著加速訓練；如果資料量小，不加也沒差，auto 會自動選擇
    device="cuda",                      # 如果有 GPU ，停用tree_method，改用 device="cuda"
    random_state=42,                    # 固定隨機種子，確保結果可重現
    n_jobs=-1,                          # 使用所有 CPU 核心加速訓練
    verbosity=1                         # 顯示訓練資訊
)

model.fit(
    X_train_processed, y_train,
    eval_set=[(X_train_processed, y_train), (X_test_processed, y_test)],
    verbose=True
    )

# =========================
# 9. 評估
# =========================
y_pred = model.predict(X_test_processed)
y_proba = model.predict_proba(X_test_processed)[:, 1]

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

try:
    auc = roc_auc_score(y_test, y_proba) # 計算 ROC 曲線下面積，衡量分類器的區分能力
    print(f"AUC: {auc:.4f}")
except ValueError:
    print("AUC 無法計算（可能全為單一類別）")

# =========================
# 10. Evaluate multiple thresholds
# =========================
thresholds = np.linspace(0.1, 0.9, 17)  # (起始, 結束, 個數)
# 初始化最佳閾值與最佳 F1 分數
best_threshold = 0.5 
best_f1 = 0

print("\nThreshold evaluation:")
for t in thresholds:
    y_pred_thresh = (y_proba >= t).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    recall = recall_score(y_test, y_pred_thresh, zero_division=0)
    print(f"Threshold={t:.2f} | F1={f1:.4f} | Precision={precision:.4f} | Recall={recall:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\n最佳 Threshold: {best_threshold:.2f}")
print(f"對應的 F1: {best_f1:.4f}\n")


# =========================
# 11. 儲存模型
# =========================
joblib.dump(
    {
        "pipeline": pipeline,
        "model": model,
        "best_threshold": best_threshold
    },
    saved_model_path
    )

print(f"[INFO] 帳戶層級模型已存檔: {saved_model_path}")


# =========================
# 12. 載入模型
# =========================
print("[INFO] 載入模型進行預測...")
saved = joblib.load(saved_model_path)
pipeline = saved["pipeline"]
model = saved["model"]

# =========================
# 13. 載入 acct_predict.csv
# =========================
predict_df = pd.read_csv(acct_predict_csv)

# =========================
# 14. 合併帳戶特徵
# =========================
predict_df = predict_df.merge(acct_features, on="acct", how="left").fillna(0)

X_pred = predict_df.drop(columns=["acct", "label"], errors="ignore")
X_pred_processed = pipeline.transform(X_pred)

# =========================
# 15. 預測
# =========================
y_pred = model.predict(X_pred_processed)

result_df = pd.DataFrame({
    "acct": predict_df["acct"],
    "label": y_pred
})

#print(result_df.head())
result_df.to_csv(acct_predict_result_csv, index=False)
print(f"[INFO] 預測結果已輸出到 {acct_predict_result_csv}")