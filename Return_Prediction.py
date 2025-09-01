import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# =====================
# 1. Load data
# =====================
df = pd.read_csv("Alalyzed_Data.csv")

# =====================
# 2. Create numeric target  
# =====================
df["Return_Flag"] = df["Returned"].map({"Yes": 1, "No": 0})

# Fill return-related columns for "No"
df.loc[df["Returned"] == "No", ["Return_Status","Return_Reason","Return_Type"]] = ["No","N/A","N/A"]
df.loc[df["Returned"] == "No", ["Return_Date","Return_Delay_Days"]] = 0

# =====================
# 3. One-hot encode categorical columns
# =====================
categorical_cols = ["Region","Category","Ship Mode","Segment","Sub-Category","Country"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# =====================
# 4. Define features
# =====================
features = [
    "Sales", "Quantity", "Discount", "Price", "Profit",
    'Region_East', 'Region_South', 'Region_West',
    'Category_Office Supplies', 'Category_Technology',
    'Ship Mode_Same Day', 'Ship Mode_Second Class', 'Ship Mode_Standard Class',
    'Segment_Corporate', 'Segment_Home Office', 
    'Sub-Category_Appliances', 'Sub-Category_Art', 'Sub-Category_Binders',
    'Sub-Category_Bookcases', 'Sub-Category_Chairs', 'Sub-Category_Copiers',
    'Sub-Category_Envelopes', 'Sub-Category_Fasteners', 'Sub-Category_Furnishings',
    'Sub-Category_Labels', 'Sub-Category_Machines', 'Sub-Category_Paper', 
    'Sub-Category_Phones', 'Sub-Category_Storage', 'Sub-Category_Supplies', 'Sub-Category_Tables'
]

X = df_encoded[features]
y = df_encoded["Return_Flag"]

# =====================
# 5. Scale features
# =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================
# 6. Train-test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =====================
# 7. Apply SMOTE on training set only
# =====================
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("After SMOTE:", X_train_res.shape, y_train_res.shape)

# =====================
# 8. Train Random Forest
# =====================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced_subsample",
    max_depth=10
)
rf.fit(X_train_res, y_train_res)

# =====================
# 9. Predictions on test set
# =====================
y_prob_test = rf.predict_proba(X_test)[:,1]

# Threshold tuning to match actual return rate
target_return_rate = y_test.mean()
threshold = np.percentile(y_prob_test, 100*(1-target_return_rate))
y_pred_test = (y_prob_test >= threshold).astype(int)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1:", f1_score(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

# =====================
# 10. Predictions on all data
# =====================
y_prob_all = rf.predict_proba(X_scaled)[:,1]
y_pred_all = (y_prob_all >= threshold).astype(int)

df["Predicted_Return_Flag"] = y_pred_all
df["Predicted_Return_Prob"] = y_prob_all

# Risk level for Power BI
def risk_level(prob):
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    else:
        return "Low"

df["Return_Risk"] = df["Predicted_Return_Prob"].apply(risk_level)

# =====================
# 11. Save predictions
# =====================
df.to_csv("Return_Predictions_PowerBI03.csv", index=False)
print("✅ File saved with predictions + risk levels for Power BI.")
