# =============================================================================
#        LOAN DEFAULT PREDICTION - COMPLETE MACHINE LEARNING PROJECT
#        Dataset: Home Equity Loan (HMEQ) | Target: BAD (1=Default, 0=No Default)
#        Covers: Unit I → Unit V of ML Syllabus
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Unit II – Regression & Regularization
from sklearn.linear_model import LogisticRegression, Lasso, Ridge

# Unit III – Classification Algorithms
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    RandomForestClassifier
)

# Unit IV – Unsupervised / Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

# Unit V – Neural Network
from sklearn.neural_network import MLPClassifier

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("   LOAN DEFAULT PREDICTION — COMPLETE ML PIPELINE")
print("="*70)


# =============================================================================
# UNIT I ─ MACHINE LEARNING FUNDAMENTALS & DATA UNDERSTANDING
# =============================================================================
print("\n" + "─"*70)
print("UNIT I: ML FUNDAMENTALS — Data Loading & Understanding")
print("─"*70)

# ── 1.1  Load the dataset ────────────────────────────────────────────────────
df = pd.read_csv("hmeq.csv")
print(f"\n✔ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nColumn descriptions:")
col_desc = {
    "BAD":     "Target — 1 = Loan Defaulted, 0 = Loan Repaid",
    "LOAN":    "Loan amount requested ($)",
    "MORTDUE": "Amount due on existing mortgage ($)",
    "VALUE":   "Current value of the property ($)",
    "REASON":  "Reason for loan (HomeImp / DebtCon)",
    "JOB":     "Applicant's occupation category",
    "YOJ":     "Years at present job",
    "DEROG":   "Number of major derogatory reports",
    "DELINQ":  "Number of delinquent credit lines",
    "CLAGE":   "Age of oldest credit line (months)",
    "NINQ":    "Number of recent credit inquiries",
    "CLNO":    "Number of existing credit lines",
    "DEBTINC": "Debt-to-income ratio (%)",
}
for col, desc in col_desc.items():
    print(f"  {col:<10} → {desc}")

print("\nFirst 5 rows:")
print(df.head())

print("\nClass distribution (Supervised Learning — Binary Classification):")
counts = df['BAD'].value_counts()
print(f"  Non-default (0): {counts[0]} ({counts[0]/len(df)*100:.1f}%)")
print(f"  Default    (1): {counts[1]} ({counts[1]/len(df)*100:.1f}%)")

# ── 1.2  ML Pipeline overview ────────────────────────────────────────────────
print("""
Machine Learning Pipeline (Unit I):
  Step 1 → Feature Selection & Engineering
  Step 2 → Data Preprocessing (missing values, encoding, scaling)
  Step 3 → Train/Test Split
  Step 4 → Model Training (Supervised Learning)
  Step 5 → Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
  Step 6 → Model Comparison & Best Model Selection
""")


# =============================================================================
# UNIT II ─ REGRESSION BASICS & COVARIANCE / CORRELATION
# =============================================================================
print("─"*70)
print("UNIT II: REGRESSION BASICS — Correlation & Feature Relationships")
print("─"*70)

# ── 2.1  Feature Engineering & Missing Value Handling ────────────────────────
# Fill numeric missing values with median (robust to outliers)
num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols.remove('BAD')  # keep target untouched

for col in num_cols:
    missing = df[col].isna().sum()
    if missing > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled {missing} missing values in '{col}' with median={median_val:.2f}")

# Fill categorical missing values with mode
for col in ['REASON', 'JOB']:
    missing = df[col].isna().sum()
    if missing > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  Filled {missing} missing values in '{col}' with mode='{mode_val}'")

# ── 2.2  Encode categorical columns ──────────────────────────────────────────
le = LabelEncoder()
df['REASON_ENC'] = le.fit_transform(df['REASON'])
df['JOB_ENC']    = le.fit_transform(df['JOB'])

feature_cols = num_cols + ['REASON_ENC', 'JOB_ENC']
X = df[feature_cols].copy()
# Final safety pass: fill any remaining NaNs with column median
X = X.fillna(X.median())
y = df['BAD']

# ── 2.3  Covariance & Correlation (Unit II core concept) ─────────────────────
print("\nCorrelation of each feature WITH the target (BAD):")
correlations = X.corrwith(y).sort_values(ascending=False)
for feat, val in correlations.items():
    bar = "█" * int(abs(val) * 30)
    sign = "+" if val > 0 else "-"
    print(f"  {feat:<12} {sign}{abs(val):.4f}  {bar}")

print("\nTop positively correlated features increase default risk.")
print("Top negatively correlated features decrease default risk.")

# ── 2.4  Train / Test Split ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 2.5  Feature Scaling (important for distance-based models) ────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit on train only → no data leakage
X_test_sc  = scaler.transform(X_test)

# ── 2.6  Regularization Demo: Lasso & Ridge (Unit II) ────────────────────────
print("\nRegularization Methods (Lasso / Ridge) — effect on feature coefficients:")
print("  • Lasso (L1)  → Can zero-out unimportant features (feature selection)")
print("  • Ridge (L2)  → Shrinks coefficients, keeps all features")

lasso_model = Lasso(alpha=0.01, max_iter=5000)
lasso_model.fit(X_train_sc, y_train)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_sc, y_train)

lasso_coefs = pd.Series(lasso_model.coef_, index=feature_cols)
print("\n  Lasso coefficients (0 = feature eliminated):")
for feat, coef in lasso_coefs.items():
    print(f"    {feat:<12} {coef:+.4f}")


# =============================================================================
# UNIT III ─ CLASSIFICATION ALGORITHMS
# =============================================================================
print("\n" + "─"*70)
print("UNIT III: CLASSIFICATION ALGORITHMS")
print("─"*70)

# Helper function to evaluate any classifier
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, scaled=True):
    """
    Train a model, predict, and return all key metrics.
    Returns a dict with accuracy, precision, recall, f1, roc_auc.
    """
    Xtr = X_tr if not scaled else X_tr
    Xte = X_te if not scaled else X_te

    model.fit(Xtr, y_tr)
    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1] if hasattr(model, 'predict_proba') else None

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_proba) if y_proba is not None else 0.0

    print(f"\n  [{name}]")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    Precision: {prec:.4f}  (of all predicted defaults, how many were real)")
    print(f"    Recall   : {rec:.4f}  (of all real defaults, how many did we catch)")
    print(f"    F1-Score : {f1:.4f}  (harmonic mean of Precision & Recall)")
    print(f"    ROC-AUC  : {auc:.4f}  (overall discriminating power)")

    return {"Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "ROC_AUC": auc, "object": model,
            "y_pred": y_pred, "y_proba": y_proba}

results = []

# ── 3.1  Logistic Regression ──────────────────────────────────────────────────
print("\n── 3.1  Logistic Regression (Linear probabilistic classifier) ──")
print("  Maps a linear combination of features to a probability via sigmoid.")
r = evaluate_model("Logistic Regression",
                   LogisticRegression(max_iter=500, random_state=42),
                   X_train_sc, y_train, X_test_sc, y_test)
results.append(r)

# ── 3.2  Decision Tree ────────────────────────────────────────────────────────
print("\n── 3.2  Decision Tree Classifier ──")
print("  Recursively splits data by the feature that best separates classes.")
r = evaluate_model("Decision Tree",
                   DecisionTreeClassifier(max_depth=5, random_state=42),
                   X_train_sc, y_train, X_test_sc, y_test)
results.append(r)

# ── 3.3  Naïve Bayes ─────────────────────────────────────────────────────────
print("\n── 3.3  Naïve Bayesian Classifier ──")
print("  Applies Bayes' theorem assuming features are independent of each other.")
r = evaluate_model("Naive Bayes",
                   GaussianNB(),
                   X_train_sc, y_train, X_test_sc, y_test)
results.append(r)

# ── 3.4  k-Nearest Neighbours ────────────────────────────────────────────────
print("\n── 3.4  k-Nearest Neighbor (k=7) ──")
print("  Classifies a point by majority vote among its k nearest neighbours.")
r = evaluate_model("KNN (k=7)",
                   KNeighborsClassifier(n_neighbors=7),
                   X_train_sc, y_train, X_test_sc, y_test)
results.append(r)

# ── 3.5  Support Vector Machine ──────────────────────────────────────────────
print("\n── 3.5  Support Vector Machine (RBF kernel) ──")
print("  Finds the maximum-margin hyperplane separating the two classes.")
svm = SVC(kernel='rbf', probability=True, random_state=42)
r = evaluate_model("SVM (RBF)",
                   svm,
                   X_train_sc, y_train, X_test_sc, y_test)
results.append(r)

# ── 3.6  Linear Discriminant Analysis ────────────────────────────────────────
print("\n── 3.6  Linear Discriminant Analysis ──")
print("  Projects features into a lower-dimensional space maximising class separation.")
r = evaluate_model("LDA",
                   LinearDiscriminantAnalysis(),
                   X_train_sc, y_train, X_test_sc, y_test)
results.append(r)

# ── 3.7  Ensemble Methods ─────────────────────────────────────────────────────
print("\n── 3.7  Ensemble Methods (Bagging / Boosting / Random Forest) ──")

print("\n  [Bagging] — trains many independent trees on random data subsets")
r = evaluate_model("Bagging",
                   BaggingClassifier(n_estimators=50, random_state=42),
                   X_train_sc, y_train, X_test_sc, y_test)
results.append(r)

print("\n  [AdaBoost] — sequential; each tree corrects predecessor's errors")
r = evaluate_model("AdaBoost",
                   AdaBoostClassifier(n_estimators=100, random_state=42),
                   X_train_sc, y_train, X_test_sc, y_test)
results.append(r)

print("\n  [Gradient Boosting] — builds trees by minimising a loss function")
r = evaluate_model("Gradient Boosting",
                   GradientBoostingClassifier(n_estimators=100, random_state=42),
                   X_train_sc, y_train, X_test_sc, y_test)
results.append(r)

print("\n  [Random Forest] — bagging + random feature selection at each split")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
r = evaluate_model("Random Forest",
                   rf_model,
                   X_train_sc, y_train, X_test_sc, y_test)
results.append(r)


# =============================================================================
# UNIT IV ─ UNSUPERVISED LEARNING & CLUSTERING
# =============================================================================
print("\n" + "─"*70)
print("UNIT IV: UNSUPERVISED LEARNING — Clustering the Loan Applicants")
print("─"*70)
print("Goal: Group applicants into risk segments WITHOUT using the label (BAD).")

# Reduce dimensions with PCA for clustering visualisation
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train_sc)

# ── 4.1  K-Means ──────────────────────────────────────────────────────────────
print("\n── 4.1  K-Means Clustering (Partitioning Method, k=3) ──")
print("  Iteratively assigns points to the nearest centroid and re-computes centroids.")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X_train_sc)
print(f"  Cluster sizes → {np.bincount(km_labels)}")
print("  Inertia (within-cluster sum of squares):", round(kmeans.inertia_, 2))

# ── 4.2  Hierarchical / Agglomerative Clustering ──────────────────────────────
print("\n── 4.2  Agglomerative Hierarchical Clustering (k=3) ──")
print("  Merges closest pairs of clusters bottom-up until k clusters remain.")
agg = AgglomerativeClustering(n_clusters=3)
agg_labels = agg.fit_predict(X_train_sc)
print(f"  Cluster sizes → {np.bincount(agg_labels)}")

# ── 4.3  DBSCAN (Density-Based) ───────────────────────────────────────────────
print("\n── 4.3  DBSCAN — Density-Based Clustering ──")
print("  Groups densely packed points; labels sparse outlier points as noise (-1).")
dbscan = DBSCAN(eps=1.5, min_samples=10)
db_labels = dbscan.fit_predict(X_train_sc)
unique_labels, counts_db = np.unique(db_labels, return_counts=True)
for lbl, cnt in zip(unique_labels, counts_db):
    tag = "Noise" if lbl == -1 else f"Cluster {lbl}"
    print(f"  {tag}: {cnt} samples")


# =============================================================================
# UNIT V ─ NEURAL NETWORKS
# =============================================================================
print("\n" + "─"*70)
print("UNIT V: NEURAL NETWORKS — Multi-Layer Perceptron")
print("─"*70)
print("Architecture: Input(12) → Hidden(64, ReLU) → Hidden(32, ReLU) → Output(Sigmoid)")
print("Training: Back-Propagation with Adam optimiser")

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),   # two hidden layers
    activation='relu',             # ReLU activation (avoids vanishing gradient)
    solver='adam',                 # adaptive learning rate (modern standard)
    max_iter=300,
    random_state=42,
    early_stopping=True,           # stop if validation loss stops improving
    validation_fraction=0.1
)
r = evaluate_model("Neural Network (MLP)",
                   mlp,
                   X_train_sc, y_train, X_test_sc, y_test)
results.append(r)

print("\n  Key Unit V concepts applied:")
print("  ✔ Back-Propagation  — gradients flow backward to update weights")
print("  ✔ ReLU activation   — avoids vanishing gradient problem")
print("  ✔ Early Stopping    — prevents overfitting (Unit I: bias-variance)")
print("  ✔ Adam optimiser    — adaptive learning rate per parameter")


# =============================================================================
# FINAL MODEL COMPARISON
# =============================================================================
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)

results_clean = [{k: v for k, v in r.items() if k not in ("object","y_pred","y_proba")}
                 for r in results]
df_results = pd.DataFrame(results_clean).set_index("Model")
print(df_results.round(4).to_string())

best_model_name = df_results['ROC_AUC'].idxmax()
best_auc        = df_results['ROC_AUC'].max()
print(f"\n🏆 Best Model by ROC-AUC: {best_model_name} (AUC = {best_auc:.4f})")

best_r = next(r for r in results if r["Model"] == best_model_name)
print("\nDetailed Classification Report for Best Model:")
print(classification_report(y_test, best_r["y_pred"],
                             target_names=["No Default", "Default"]))


# =============================================================================
# VISUALISATIONS — one large figure with all key plots
# =============================================================================
print("\nGenerating visualisation dashboard …")

fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor('#0d1117')
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

DARK  = '#0d1117'
CARD  = '#161b22'
ACCENT= '#58a6ff'
GREEN = '#3fb950'
RED   = '#f85149'
TEXT  = '#e6edf3'
MUTED = '#8b949e'

def card_ax(ax, title):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(ACCENT)
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

# ── Plot 1: Class Distribution ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
card_ax(ax1, "Unit I — Class Distribution")
vals  = y.value_counts()
bars  = ax1.bar(["No Default", "Default"], vals.values,
                color=[GREEN, RED], edgecolor='#30363d', linewidth=0.8)
for bar, val in zip(bars, vals.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             str(val), ha='center', color=TEXT, fontsize=9, fontweight='bold')
ax1.set_ylabel("Count", color=TEXT)

# ── Plot 2: Correlation Heatmap ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
card_ax(ax2, "Unit II — Feature Correlation")
corr_mat = df[num_cols + ['BAD']].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.heatmap(corr_mat, ax=ax2, cmap=cmap, center=0,
            linewidths=0.3, linecolor='#0d1117',
            annot=True, fmt=".1f", annot_kws={"size": 6},
            cbar_kws={"shrink": 0.8})
ax2.tick_params(labelsize=6, colors=TEXT)

# ── Plot 3: Feature Importances (Random Forest) ───────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
card_ax(ax3, "Unit III — Feature Importance (RF)")
rf_imp = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
colors_imp = [ACCENT if v > rf_imp.median() else MUTED for v in rf_imp.values]
ax3.barh(rf_imp.index, rf_imp.values, color=colors_imp, edgecolor='#30363d')
ax3.set_xlabel("Importance", color=TEXT)

# ── Plot 4: ROC Curves for all models ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
card_ax(ax4, "Unit III — ROC Curves (All Models)")
palette = plt.cm.tab10(np.linspace(0, 1, len(results)))
for r_item, col in zip(results, palette):
    if r_item["y_proba"] is not None:
        fpr, tpr, _ = roc_curve(y_test, r_item["y_proba"])
        ax4.plot(fpr, tpr, color=col, lw=1.5,
                 label=f"{r_item['Model']} (AUC={r_item['ROC_AUC']:.3f})")
ax4.plot([0,1],[0,1], '--', color=MUTED, lw=0.8, label='Random Classifier')
ax4.set_xlabel("False Positive Rate", color=TEXT)
ax4.set_ylabel("True Positive Rate", color=TEXT)
ax4.legend(loc='lower right', fontsize=6.5,
           facecolor=CARD, edgecolor='#30363d', labelcolor=TEXT)

# ── Plot 5: Confusion Matrix (Best Model) ────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
card_ax(ax5, f"Unit III — Confusion Matrix\n({best_model_name})")
cm = confusion_matrix(y_test, best_r["y_pred"])
sns.heatmap(cm, annot=True, fmt='d', ax=ax5,
            cmap=sns.light_palette(ACCENT, as_cmap=True),
            linewidths=1, linecolor=DARK,
            xticklabels=["No Default","Default"],
            yticklabels=["No Default","Default"])
ax5.tick_params(labelsize=7, colors=TEXT)

# ── Plot 6: Metric Bar Chart ──────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, :])
card_ax(ax6, "Unit I — Model Performance Comparison (All Metrics)")
metrics  = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
n_models = len(df_results)
x        = np.arange(n_models)
width    = 0.15
met_cols = [ACCENT, GREEN, RED, '#d2a8ff', '#ffa657']
for i, (met, col) in enumerate(zip(metrics, met_cols)):
    ax6.bar(x + i*width, df_results[met].values, width,
            label=met, color=col, edgecolor=DARK, linewidth=0.5)
ax6.set_xticks(x + width*2)
ax6.set_xticklabels(df_results.index, rotation=25, ha='right', fontsize=7.5)
ax6.set_ylim(0, 1.05)
ax6.legend(loc='upper left', fontsize=8,
           facecolor=CARD, edgecolor='#30363d', labelcolor=TEXT)
ax6.set_ylabel("Score", color=TEXT)

# ── Plot 7: K-Means Clusters (PCA) ───────────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 0])
card_ax(ax7, "Unit IV — K-Means Clusters (PCA 2D)")
cluster_colors = [GREEN, ACCENT, RED]
for k in range(3):
    mask_k = km_labels == k
    ax7.scatter(X_pca[mask_k, 0], X_pca[mask_k, 1],
                s=5, alpha=0.5, color=cluster_colors[k], label=f"Cluster {k}")
ax7.legend(fontsize=7, facecolor=CARD, edgecolor='#30363d', labelcolor=TEXT)
ax7.set_xlabel("PC 1", color=TEXT); ax7.set_ylabel("PC 2", color=TEXT)

# ── Plot 8: DBSCAN (PCA) ─────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[3, 1])
card_ax(ax8, "Unit IV — DBSCAN Density Clusters (PCA 2D)")
unique_db = np.unique(db_labels)
db_palette = [RED if l == -1 else plt.cm.cool(l / max(unique_db.max(), 1))
              for l in unique_db]
for lbl, col in zip(unique_db, db_palette):
    mask_db = db_labels == lbl
    tag = "Noise" if lbl == -1 else f"Cluster {lbl}"
    ax8.scatter(X_pca[mask_db, 0], X_pca[mask_db, 1],
                s=5, alpha=0.6, color=col, label=tag)
ax8.legend(fontsize=6, facecolor=CARD, edgecolor='#30363d', labelcolor=TEXT)
ax8.set_xlabel("PC 1", color=TEXT); ax8.set_ylabel("PC 2", color=TEXT)

# ── Plot 9: Neural Network Training Loss ─────────────────────────────────────
ax9 = fig.add_subplot(gs[3, 2])
card_ax(ax9, "Unit V — Neural Network Loss Curve")
loss_curve = mlp.loss_curve_
ax9.plot(loss_curve, color=ACCENT, lw=2)
ax9.set_xlabel("Epoch", color=TEXT)
ax9.set_ylabel("Training Loss", color=TEXT)
ax9.text(len(loss_curve)*0.6, max(loss_curve)*0.85,
         "Back-Propagation\nconverging →",
         color=GREEN, fontsize=7.5)

# ── Super Title ───────────────────────────────────────────────────────────────
fig.suptitle("Loan Default Prediction — Complete ML Pipeline Dashboard",
             fontsize=16, fontweight='bold', color=TEXT, y=0.98)

plt.savefig("/mnt/user-data/outputs/loan_default_dashboard.png",
            dpi=150, bbox_inches='tight', facecolor=DARK)
print("✔ Dashboard saved → loan_default_dashboard.png")

# =============================================================================
# PREDICT A NEW APPLICANT (Example usage)
# =============================================================================
print("\n" + "─"*70)
print("EXAMPLE PREDICTION — New Loan Applicant")
print("─"*70)

new_applicant = pd.DataFrame([{
    "LOAN":    8000,    # loan amount
    "MORTDUE": 40000,   # mortgage due
    "VALUE":   60000,   # property value
    "YOJ":     5,       # years at job
    "DEROG":   1,       # derogatory reports
    "DELINQ":  2,       # delinquent lines
    "CLAGE":   80,      # age of oldest credit line
    "NINQ":    3,       # recent inquiries
    "CLNO":    12,      # number of credit lines
    "DEBTINC": 42,      # debt-to-income ratio
    "REASON_ENC": 1,    # HomeImp
    "JOB_ENC":    3,    # Other
}])

new_scaled = scaler.transform(new_applicant[feature_cols])
best_obj   = best_r["object"]
pred_label = best_obj.predict(new_scaled)[0]
pred_proba = best_obj.predict_proba(new_scaled)[0][1]

print(f"\n  Model Used    : {best_model_name}")
print(f"  Default Prob  : {pred_proba*100:.1f}%")
print(f"  Decision      : {'⚠ LOAN DEFAULT PREDICTED' if pred_label==1 else '✔ LOAN LIKELY TO BE REPAID'}")

print("\n" + "="*70)
print("ALL UNITS COVERED:")
print("  Unit I   → ML fundamentals, pipeline, accuracy/precision/recall/F1")
print("  Unit II  → Correlation, Lasso, Ridge regularization, train-test split")
print("  Unit III → LR, DT, NB, KNN, SVM, LDA, Bagging, Boosting, RF")
print("  Unit IV  → K-Means, Agglomerative, DBSCAN, PCA visualization")
print("  Unit V   → MLP Neural Network with back-propagation, ReLU, Adam")
print("="*70)
