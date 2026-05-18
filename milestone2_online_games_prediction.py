 #----------------------------------------------------------------------------------------------------------------------#
 #code block num_1 : Imports
 #----------------------------------------------------------------------------------------------------------------------#
import json
import os
import pickle
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

#-----------------------------------------------------------------------------------------------------------------------#
#block 2 - Configurations:
#-----------------------------------------------------------------------------------------------------------------------#
DATA_PATH = "data/train_data.csv"
PLOTS_DIR = "artifacts/plots"
MODELS_DIR = "artifacts/models"
METRICS_PATH = "artifacts/metrics.json"
TARGET = "GamePopularity"
RANDOM_STATE = 42
TEST_SIZE = 0.20 #to ensure that only 20% is used as test data

#-----------------------------------------------------------------------------------------------------------------------#
#Game popularity threshold are :
#RECS = Recommendations
# LOW : 0 RECS. to 100 RECS.
# Medium = 101 RECS. to 1000 RECS.
# High = 1001+ RECS.
#these thresholds are according to the dataset
#-----------------------------------------------------------------------------------------------------------------------#

LOW_THRESH= 100
HIGH_THRESH = 1000

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

CLASS_ORDER =["Low", "Medium", "High"] #consistent ordering

#-----------------------------------------------------------------------------------------------------------------------#
#Block 3 : Preprocessing
#-----------------------------------------------------------------------------------------------------------------------#
def preprocessing(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    #Naming the Columns
    df["QueryName"]     = df["QueryName"].fillna(df["ResponseName"]).str.strip()
    df["ResponseName"]  = df["ResponseName"].str.strip()

    #Release Dates
    df["ReleaseDate"] = pd.to_datetime(df["ReleaseDate"], errors="coerce")

    #By this we give numeric values to all so we avoid any possible NULL values
    #RequiredAge, DemoCount, DeveloperCount, PublisherCount, DLCCount,
    #MovieCount, ScreenshotCount, PackageCount, AchievementCount,
    #AchievementHighlightedCount, Metacritic: leave as-is

    #to remove anything lower than 0 " removes negatives"
    for col in ["SteamSpyOwners", "SteamSpyOwnersVariance","SteamSpyPlayersEstimate", "SteamSpyPlayersVariance"]:
        df[col] = df[col].clip(lower=0)

    #Deciding price columns
    df["PriceInitial"] = df["PriceInitial"].clip(lower=0)
    df["PriceFinal"] = df["PriceFinal"].clip(lower=0)

    #Price currency
    df["PriceCurrency"] = df["PriceCurrency"].replace("", "Unknown").fillna("Unknown")


    #Conversion of Boolean Columns into integers.
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    str_cols = [
        "SupportEmail", "SupportURL", "Website", "AboutText", "ShortDescrip", "DetailedDescrip",
        "PCMinReqsText", "PCRecReqsText", "LinuxMinReqsText", "LinuxRecReqsText", "MacMinReqsText", "MacRecReqsText",
        "Reviews", "SupportedLanguages", "LegalNotice", "DRMNotice", "ExtUserAcctNotice", "Background", "HeaderImage"
    ]
    for col in str_cols:
        df[col]= df[col].fillna("").str.strip()

    return df

#-----------------------------------------------------------------------------------------------------------------------#
# 4th Block new column creation
#-----------------------------------------------------------------------------------------------------------------------#

def create_tgt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rc = df["RecommendationCount"]
    df[TARGET] = pd.cut(
        rc,
        bins=[-1 , LOW_THRESH, HIGH_THRESH, float("inf")],
        labels=CLASS_ORDER,
    )
    return df

#-----------------------------------------------------------------------------------------------------------------------#
#5th Block : Feature Engineering
#-----------------------------------------------------------------------------------------------------------------------#
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    parts = []

    #Date Features
    ref = df["ReleaseDate"].dropna().max()
    parts.append(pd.DataFrame( {
        "Release_Year":         df["ReleaseDate"].dt.year,
        "Release_Month":        df["ReleaseDate"].dt.month,
        "Release_WeekDay":    df["ReleaseDate"].dt.day_of_week,
        "ReleaseAgeDays":       (ref - df["ReleaseDate"]).dt.days
    }, index=df.index))

    #Pricing Features
    init = df["PriceInitial"]
    final= df["PriceFinal"]
    ratio = np.where(init > 0 , (init - final) / init , 0.0)
    parts.append(pd.DataFrame({
        "LoggedFinalPrice":     np.log1p(final),
        "LoggedInitialPrice": np.log1p(init),
        "PriceDiscountRatio": np.clip(ratio, 0.0, 1.0),
        "IsFreePrice":  (final == 0).astype(int),
        "IsPremium":    (final <= 20).astype(int),
        "IsFreePrice_Mismatch":     ((final == 0) & (df["IsFree"] == 0)).astype(int),
    }, index=df.index))

    #SteamSpy (users & employees & owners
    owners = df["SteamSpyOwners"]
    players = df["SteamSpyPlayersEstimate"]
    own_var = df["SteamSpyOwnersVariance"]
    play_var = df["SteamSpyPlayersVariance"]
    parts.append(pd.DataFrame({
        "Logged_SteamSpy_Owners": np.log1p(owners),
        "Logged_SteamSpy_Players": np.log1p(players),
        "Logged_SteamSpy_Player_Variance": np.log1p(play_var),
        "Logged_SteamSpy_Owner_Variance": np.log1p(own_var),
        "Logged_OwnerShip_Lower": np.log1p(np.maximum(0 , owners - own_var)),
        "Logged_OwnerShip_Upper": np.log1p(owners + own_var),
        "Owners_Rank": owners.rank(pct=True),
        "Players_Rank": players.rank(pct=True),
        "Engagement_Ratio": np.where(owners > 0, players / owners, 0.0),
    },index=df.index))

    #metacritic scores
    score = df["Metacritic"]
    parts.append(pd.DataFrame({
        "Metacritic_Score": score,
        "Has_Metacritic": (score > 0).astype(int)
    }, index = df.index))

    #richness of content
    parts.append(pd.DataFrame({
        "LoggedMovieCount": np.log1p(df["MovieCount"]),
        "LoggedScreenshotCount": np.log1p(df["ScreenshotCount"]),
        "LoggedDLCCount": np.log1p(df["DLCCount"]),
        "LoggedAchievementCount": np.log1p(df["AchievementCount"]),
        "LoggedAchievementHL": np.log1p(df["AchievementHighlightedCount"]),
        "LoggedPackageCount": np.log1p(df["PackageCount"]),
        "LoggedDeveloperCount": np.log1p(df["DeveloperCount"]),
        "LoggedPublisherCount": np.log1p(df["PublisherCount"]),
        "LoggedDemoCount": np.log1p(df["DemoCount"]),
        "LoggedContentScore": np.log1p(df["MovieCount"]) + np.log1p(df["ScreenshotCount"]) + np.log1p(df["DLCCount"]),
    }, index=df.index))

    #Language Features
    cleaned = df["SupportedLanguages"].str.replace(
             r"\*[^*]*\*?languages[^*]*", "", regex=True
    )
    lang_count = cleaned.str.split().apply(len)
    parts.append(pd.DataFrame({
        "language_count": lang_count,
        "Logged_language_count": np.log1p(lang_count),
        "Multilingual": (lang_count > 1).astype(int),
    }, index=df.index))

    #Platform Features
    parts.append(pd.DataFrame({
        "PlatformCount": (df["PlatformWindows"] + df["PlatformLinux"] + df["PlatformMac"]),
        "SupportsLinux": df["PlatformLinux"],
        "SupportsMac":   df["PlatformMac"],
    },index=df.index))

    #MultiPlayer Features
    is_multi = (
        df["CategoryMultiplayer"] | df["CategoryCoop"] | df["CategoryMMO"]).astype(int)
    log_owners = np.log1p(df["SteamSpyOwners"])
    parts.append(pd.DataFrame({
        "IsMultiplayer": is_multi,
        "Language_X_Multiplayer": lang_count * is_multi,
        "Owners_X_Multiplayer": log_owners * is_multi,
        "Action_X_Multiplayer": df["GenreIsAction"] * is_multi,
    },index=df.index))

    #Age/Maturitiy Features
    age = df["RequiredAge"]
    parts.append(pd.DataFrame({
        "RequiredAge": age,
        "IsMature": (age >=17).astype(int),
    }, index=df.index))

    #Text Richness
    text_out ={}
    for col in ["AboutText", "ShortDescrip", "DetailedDescrip", "PCMinReqsText", "PCRecReqsText", "Reviews"]:
        text_out[f"{col}_len"] = np.log1p(df[col].str.len())
        text_out[f"{col}_has"] = (df[col] != "").astype(int)
    for col in ["LinuxMinReqsText", "LinuxRecReqsText",
               "MacMinReqsText", "MacRecReqsText"]:
        text_out[f"has_{col}"] = (df[col] != "").astype(int)


    for col in ["SupportEmail", "SupportURL", "Website",
             "LegalNotice", "DRMNotice", "ExtUserAcctNotice",
             "Background", "HeaderImage"]:
        text_out[f"has_{col}"] = (df[col] != "").astype(int)
    parts.append(pd.DataFrame(text_out, index=df.index))

    #Interaction Features
    log_players = np.log1p(df["SteamSpyPlayersEstimate"])
    log_own_var = np.log1p(df["SteamSpyOwnersVariance"])
    has_meta    = (df["Metacritic"] > 0).astype(float)
    parts.append(pd.DataFrame({
        "Owners_X_Metacritic": log_owners * df["Metacritic"],
        "Platers_X_Metacritic": log_players * df["Metacritic"],
        "Owners_X_has_Metacritic": log_owners * has_meta,
        "Var_X_owners": log_own_var * log_owners,
    }, index=df.index))

    #Passing through: all boolean Category/Genre/Platform columns that already are INTs
    bool_int_cols = [
        c for c in df.select_dtypes(include=[int , "int64"]).columns
        if c.startswith((
            "Category",
            "Genre",
            "Platform",
            "PCReqs",
            "LinuxReqs",
            "MacReqs",
            "IsFree",
            "FreeVer"
            "Purchase",
            "Subscription",
            "Controller"
        ))

    ]
    if bool_int_cols:
        parts.append(df[bool_int_cols])

    already = {
        "PriceInitial", "priceFinal", "RequiredAge", "SteamSpyOwners","SteamSpyOwnersVariance", "SteamSpyPlayersEstimate",
        "SteamSpyPlayersVariance","Metacritic","MovieCount","ScreenshotCount","DLCCount","AchievementCount","AchievementHighlightedCount",
        "PackageCount", "DeveloperCount","PublisherCount","DemoCount", "RecommendationCount"
    }
    extra = [ c for c in df.select_dtypes(include=[np.number]).columns
              if c not in already]

    if extra:
        parts.append(df[extra])

    #Currency
    parts.append(pd.get_dummies(df["PriceCurrency"], prefix="currency"))

    X = pd.concat(parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X

#-----------------------------------------------------------------------------------------------------------------------#
#6th Block : Helper for Metrics
#-----------------------------------------------------------------------------------------------------------------------#
def compute_clf_metrics(y_true, y_pred, label_encoder, name, train_time, test_time):
    acc= accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
     )
    return{
        "model": name,
        "accuracy": round(float(acc), 4),
        "accuracy_%": round(float(acc)* 100 , 2),
        "train_time": round(train_time, 4),
        "test_time": round(test_time, 6),
        "report": report,
    }

#-----------------------------------------------------------------------------------------------------------------------#
#7th Block : Helpers for Plotting
#-----------------------------------------------------------------------------------------------------------------------#
def plot_class_distribution(y_series, path):
    counts = y_series.value_counts().reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=["#ef4444", "#f59e0b", "#22c55e"], edgecolor="white", width=0.55)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 80,
                f"{int(bar.get_height()):,}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("GamePopularity Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("GamePopularity")
    ax.set_ylabel("Number of Games")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_accuracy_bar(results, path):
    names  = [r["model"] for r in results]
    accs   = [r["accuracy_%"] for r in results]
    colors = ["#3b82f6", "#16a34a", "#b45309"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, accs, color=colors[:len(names)], edgecolor="white", width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.2f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_title("Classification Accuracy — All Models (Test Set)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Model")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_train_time_bar(results, path):
    names  = [r["model"] for r in results]
    times  = [r["train_time"] for r in results]
    colors = ["#3b82f6", "#16a34a", "#b45309"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, times, color=colors[:len(names)], edgecolor="white", width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}s",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("Total Training Time — All Models", fontsize=14, fontweight="bold")
    ax.set_ylabel("Time (seconds)")
    ax.set_xlabel("Model")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_test_time_bar(results, path):
    names  = [r["model"] for r in results]
    times  = [r["test_time"] * 1000 for r in results]   # convert to ms
    colors = ["#3b82f6", "#16a34a", "#b45309"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, times, color=colors[:len(names)], edgecolor="white", width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}ms",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("Total Test (Inference) Time — All Models", fontsize=14, fontweight="bold")
    ax.set_ylabel("Time (milliseconds)")
    ax.set_xlabel("Model")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, le, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=le.transform(CLASS_ORDER))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{title} — Confusion Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_hyperparameter_effect(param_name, param_values, cv_scores, model_name, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([str(v) for v in param_values], cv_scores,
            marker="o", linewidth=2.5, markersize=9, color="#2563eb")
    for x, y in zip(range(len(param_values)), cv_scores):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10)
    ax.set_title(f"{model_name}: Effect of '{param_name}' on CV Accuracy\n"
                 f"(all other hyperparameters fixed)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Mean CV Accuracy (5-fold)")
    ax.set_ylim(max(0, min(cv_scores) - 0.05), min(1, max(cv_scores) + 0.05))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_feature_importance_clf(importances, feature_names, title, color, path):
    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).nlargest(20, "importance").iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, color=color)
    ax.set_title(f"{title} — Top 20 Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {path}")


#-----------------------------------------------------------------------------------------------------------------------#
#8th Block : Main Pipeline
#-----------------------------------------------------------------------------------------------------------------------#

def main():
    #Loading of raw data
    print("=" * 65)
    print("Online Games Popularity Classification Prediction")
    print("=" * 65)
    df_raw = pd.read_csv(DATA_PATH)
    print(f"\n[1] Loaded dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    #------------------------------------#
    #1st Step: preprocessing of columns
    #------------------------------------#
    print("\n[2] Preprocessing all columns...")
    df = preprocessing(df_raw)
    print(f"Done, Shape: {df.shape}")

    #-----------------------------------------------------------------------#
    #2nd Step: Creation of target column. Classification: low, medium, high.
    #-----------------------------------------------------------------------#
    print(f"\n[3] Creating '{TARGET}' from RecommendationCount...")
    df= create_tgt(df)

    #Drop of unneeded rows *NAN* , this ensures always higher accuracy
    df = df.dropna(subset=[TARGET])
    dist = df[TARGET].value_counts().reindex(CLASS_ORDER)
    print(f"Distribution:\n {dist.to_string()}")
    plot_class_distribution(df[TARGET], f"{PLOTS_DIR}/Class_Distribution.png")

    #Encoding
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y = le.transform(df[TARGET])
    print(f" Label Encoding: {dict(zip(CLASS_ORDER , le.transform(CLASS_ORDER)))}")

    #Feature Engineering
    print("\n[4] Feature Engineering...")
    X= feature_engineering(df)
    print(f"Feature matrix: {X.shape}")

    #Training and Testing Data splitting *80% for training and 20% for testing*
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[5] Training_Data/Testing_Data splitting: {len(X_train_raw)} training | {len(X_test_raw)} testing")

    #Imputation
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train_raw),
        columns=X_train_raw.columns, index=X_train_raw.index
    )

    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test_raw),
        columns=X_test_raw.columns, index=X_test_raw.index
    )

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    all_results = []
    tuning_records = {}

    #-------------------------------------------------------------------------------------------------------------------#
    #Model 1 - LGBM
    #-------------------------------------------------------------------------------------------------------------------#
    print("\n" + "=" * 65)
    print("Model 1: LGBM...")
    print("=" * 65)

    FIXED_LGB_LEAVES = 63

    #We give fixed number of leavers to tune
    lgb_nest_values =[200, 400, 600]
    lgb_nest_cv= []
    print(f"\n Tuning of N_Estimators (num_leaves_fixed = {FIXED_LGB_LEAVES}):")
    for n  in lgb_nest_values:
        lgb =LGBMClassifier(
            n_estimators=n, num_leaves=FIXED_LGB_LEAVES,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
            n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
        )
        score = cross_val_score(lgb, X_train_imputed, y_train, cv=kfold, scoring="accuracy").mean()

        lgb_nest_cv.append(score)
        print(f" n_estimators={n:=<5} CV Accuracy = { score:.4}")
    best_lgb_nest = lgb_nest_values[int(np.argmax(lgb_nest_cv))]
    plot_hyperparameter_effect(
        "n_estimators", lgb_nest_values, lgb_nest_cv,
        "LightGBM",
        f"{PLOTS_DIR}/lgb_tune_n_estimators.png"
    )
    tuning_records["LightGBM_n_estimators"]= {
        "values": lgb_nest_values, "cv_acc": lgb_nest_cv, "best": best_lgb_nest
    }

    #n_estimators *leaves* fixed at the best values that gived the best results for faster execution
    leaf_values = [31, 62, 127] #doubled or round doubled every time
    leaf_cv = []
    print(f"\n Tuning Number_ofLeaves (n_estimators fixed = {FIXED_LGB_LEAVES}):")
    for nl in leaf_values:
        lgb =LGBMClassifier(
        n_estimators=best_lgb_nest, num_leaves=nl,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
        )
        score = cross_val_score(lgb, X_train_imputed, y_train, cv=kfold, scoring="accuracy").mean()
        leaf_cv.append(score)
        print(f"Number_of_leaves={nl:<4} CV Accuracy = { score:.4}")
    best_lgb_leaves = leaf_values[int(np.argmax(leaf_cv))]
    plot_hyperparameter_effect(
        "Number_of_Leaves", leaf_values, leaf_cv,
        "LightGBM",
        f"{PLOTS_DIR}/lgb_tune_n_leaves.png"
    )
    tuning_records["LightGBM_n_leaves"]= {
        "values": leaf_values, "cv_acc": leaf_cv, "best": best_lgb_leaves
    }

    #LightGBM Model finalization:

    print(f"\n  Best hyperparams: n_estimators={best_lgb_nest}, num_leaves={best_lgb_leaves}")
    lgb_model = LGBMClassifier(
        n_estimators=best_lgb_nest, num_leaves=best_lgb_leaves,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
    )
    t0 = time.time()
    lgb_model.fit(X_train_imputed, y_train)
    lgb_train_time = time.time() - t0

    t0 = time.time()
    lgb_pred = lgb_model.predict(X_test_imputed)
    lgb_test_time = time.time() - t0

    lgb_m = compute_clf_metrics(y_test, lgb_pred, le, "LightGBM",
                                 lgb_train_time, lgb_test_time)
    all_results.append(lgb_m)
    print(f"  Train time: {lgb_train_time:.4f}s  |  Test time: {lgb_test_time*1000:.3f}ms")
    print(f"  Test Accuracy: {lgb_m['accuracy_%']:.2f}%")
    print(classification_report(y_test, lgb_pred, target_names=CLASS_ORDER, zero_division=0))

    plot_confusion_matrix(y_test, lgb_pred, le,
                          "LightGBM",
                          f"{PLOTS_DIR}/lgb_confusion_matrix.png")
    plot_feature_importance_clf(
        lgb_model.feature_importances_, X_train_imputed.columns.tolist(),
        "LightGBM", "#b45309",
        f"{PLOTS_DIR}/lgb_feature_importance.png"
    )

    # Save model
    lgb_artifact = {"model": lgb_model, "imputer": imputer, "label_encoder": le}
    with open(f"{MODELS_DIR}/lightgbm.pkl", "wb") as f:
        pickle.dump(lgb_artifact, f)
    print(f"  Saved model -> {MODELS_DIR}/lightgbm.pkl")


    #-------------------------------------------------------------------------------------------------------------------#
    #Model 2 : Gradient Boosting
    #-------------------------------------------------------------------------------------------------------------------#
    print("\n" + "=" * 65)
    print("Model 2; GradientBoostign")
    print("=" * 65)

    FIXED_GB_DEPTH = 5

    #tune n_estimators (fixed max depth)
    gb_nest_values = [100, 200, 300]
    gb_nest_cv =[]
    print(f"\n Tuning of N_Estimators (Fixed_Max_Depth = {FIXED_GB_DEPTH}):")
    for n in gb_nest_values:
        gb = GradientBoostingClassifier(
            n_estimators=n, max_depth=FIXED_GB_DEPTH, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE,
        )
        score = cross_val_score(gb, X_train_imputed, y_train, cv=kfold, scoring="accuracy").mean()
        gb_nest_cv.append(score)
        print(f"N_Estimators={n:<5} CV Accuracy = { score:.4}")
    best_gb_nest = gb_nest_values[int(np.argmax(gb_nest_cv))]
    plot_hyperparameter_effect(
        "N_Estimators", gb_nest_values, gb_nest_cv,
        "GradientBoosting", f"{PLOTS_DIR}/gb_tune_n_estimators.png"
    )
    tuning_records["GradientBoosting_n_estimators"]= {
        "values": gb_nest_values, "cv_acc": gb_nest_cv, "best": best_gb_nest
    }

    #Max Depth Tuning
    gb_depth_values = [3 ,5 ,7]
    gb_depth_cv =[]
    print(f"\n Max_Depth_Tuning (N_Estimators Fixed = {best_gb_nest}):")
    for d in gb_depth_values:
        gb = GradientBoostingClassifier(
            n_estimators=best_gb_nest, max_depth=d, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE,
        )
        score = cross_val_score(gb, X_train_imputed, y_train, cv=kfold, scoring="accuracy").mean()

        gb_depth_cv.append(score)
        print(f"Max_Depth={d} CV Accuracy = { score:.4}")

    best_gb_depth = gb_depth_values[int(np.argmax(gb_depth_cv))]
    plot_hyperparameter_effect(
        "Max_Depth", gb_depth_values, gb_depth_cv,
        "Gradient Boosting", f"{PLOTS_DIR}/gb_tune_Max_Depth.png"
    )
    tuning_records["GradientBoosting_Max_Depth"]= {
        "values": gb_depth_values, "cv_acc": gb_depth_cv, "best": best_gb_depth
    }

    #Gradient Boost model Finalization
    print(f"\n Best Hyperparameters:  N_Estimators={best_gb_nest}, Max_Depth={best_gb_depth}")
    gb_model = GradientBoostingClassifier(
        n_estimators=best_gb_nest, max_depth=best_gb_depth, learning_rate=0.05, subsample=0.8,random_state=RANDOM_STATE
    )
    t0 = time.time()
    gb_model.fit(X_train_imputed, y_train)
    gb_train_time = time.time() - t0

    t0 = time.time()
    gb_pred = gb_model.predict(X_test_imputed)
    gb_test_time = time.time() - t0

    gb_m = compute_clf_metrics(y_test, gb_pred, le, "GradientBoosting", gb_train_time, gb_test_time)
    all_results.append(gb_m)
    print(f"  Train time: {gb_train_time:.2f}s  |  Test time: {gb_test_time * 1000:.3f}ms")
    print(f"  Test Accuracy: {gb_m['accuracy_%']:.2f}%")
    print(classification_report(y_test, gb_pred, target_names=CLASS_ORDER, zero_division=0))

    plot_confusion_matrix(y_test, gb_pred, le,
                          "Gradient Boosting",
                          f"{PLOTS_DIR}/gb_confusion_matrix.png")

    gb_artifact = {"model": gb_model, "imputer": imputer, "label_encoder": le}
    with open(f"{MODELS_DIR}/gradient_boosting.pkl", "wb") as f:
        pickle.dump(gb_artifact, f)
    print(f"  Saved model -> {MODELS_DIR}/gradient_boosting.pkl")

    #-------------------------------------------------------------------------------------------------------------------#
    #Model 3: Stacked Ensemble using (LGB , XGB , RF , and ET = XGBoost meta-learner)
    #-------------------------------------------------------------------------------------------------------------------#
    print("\n" + "=" * 65)
    print("  MODEL 3: Stacked Ensemble (LGB + XGB + RF + ET → XGB meta)")
    print("=" * 65)

    #Helper - OOF
    def get_oof_proba(model, X , y , kf):
        n_classes = len(np.unique(y))
        oof = np.zeros((len(y), n_classes))
        for tr_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr = y[tr_idx]
            model.fit(X_tr, y_tr)
            oof[val_idx, :] = model.predict_proba(X_val)
        return oof

    #Base models (LGB , XGB , RF , and ET)
    base_lgb = LGBMClassifier(
        n_estimators = best_lgb_nest , num_leaves=best_lgb_nest, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.01, reg_lambda=1.0, min_child_samples=20,
        n_jobs=-1, random_state=RANDOM_STATE,verbose=-1,
    )
    base_xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8 , colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE,
        verbosity=0,
    )
    base_rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    base_et = ExtraTreesClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    base_models =[
        ("LightGBM", base_lgb),
        ("XGBoost", base_xgb),
        ("RandomForest", base_rf),
        ("Extra Trees" , base_et),
    ]

    #Meta Features Building :
    print("\n  Building OOF predictions for each base model ...")
    t_stack_start = time.time()
    oof_train_parts = []
    test_pred_parts = []

    for bname, bmodel in base_models:
        print(f"    OOF → {bname} ...", end=" ", flush=True)
        t0 = time.time()
        oof_proba = get_oof_proba(bmodel, X_train_imputed, y_train, kfold)
        oof_train_parts.append(oof_proba)
        print(f"done ({time.time() - t0:.1f}s)")
        bmodel.fit(X_train_imputed, y_train)
        test_pred_parts.append(bmodel.predict_proba(X_test_imputed))

    meta_train_feats = np.hstack(oof_train_parts)  # (n_train, 12)
    meta_test_feats = np.hstack(test_pred_parts)  # (n_test,  12)
    print(f"  Meta-feature shape — train: {meta_train_feats.shape}  test: {meta_test_feats.shape}")

    #Meta Learner Tuning for N_estimators . fixed max depth of 3
    FIXED_META_DEPTH = 3
    meta_nest_values = [50, 100, 200]
    meta_nest_cv = []
    print(f"\n  Tuning meta XGB n_estimators (max_depth fixed = {FIXED_META_DEPTH}):")
    for n in meta_nest_values:
        meta_xgb = XGBClassifier(
            n_estimators=n, max_depth=FIXED_META_DEPTH, learning_rate=0.1,
            eval_metric="mlogloss", n_jobs=-1,
            random_state=RANDOM_STATE, verbosity=0,
        )
        score = cross_val_score(meta_xgb, meta_train_feats, y_train,
                                cv=kfold, scoring="accuracy").mean()
        meta_nest_cv.append(score)
        print(f"    n_estimators={n:<5}  CV Acc = {score:.4f}")
    best_meta_nest = meta_nest_values[int(np.argmax(meta_nest_cv))]
    plot_hyperparameter_effect(
        "meta n_estimators", meta_nest_values, meta_nest_cv,
        "Stacked Ensemble (XGB meta)", f"{PLOTS_DIR}/stack_tune_n_estimators.png"
    )
    tuning_records["StackedEnsemble_meta_n_estimators"] = {
        "values": meta_nest_values, "cv_acc": meta_nest_cv, "best": best_meta_nest
    }

    #Meta learner tuning for max depth which N_estimators are fixed at best
    meta_depth_values = [2, 3, 5]
    meta_depth_cv = []
    print(f"\n  Tuning meta XGB max_depth (n_estimators fixed = {best_meta_nest}):")
    for d in meta_depth_values:
        meta_xgb = XGBClassifier(
            n_estimators=best_meta_nest, max_depth=d, learning_rate=0.1,
            eval_metric="mlogloss", n_jobs=-1,
            random_state=RANDOM_STATE, verbosity=0,
        )
        score = cross_val_score(meta_xgb, meta_train_feats, y_train,
                                cv=kfold, scoring="accuracy").mean()
        meta_depth_cv.append(score)
        print(f"    max_depth={d}  CV Acc = {score:.4f}")
    best_meta_depth = meta_depth_values[int(np.argmax(meta_depth_cv))]
    plot_hyperparameter_effect(
        "meta max_depth", meta_depth_values, meta_depth_cv,
        "Stacked Ensemble (XGB meta)", f"{PLOTS_DIR}/stack_tune_max_depth.png"
    )
    tuning_records["StackedEnsemble_meta_max_depth"] = {
        "values": meta_depth_values, "cv_acc": meta_depth_cv, "best": best_meta_depth
    }

    #Meta learner Finalization
    print(f"\n  Best meta params: n_estimators={best_meta_nest}, max_depth={best_meta_depth}")
    meta_xgb_final = XGBClassifier(
        n_estimators=best_meta_nest, max_depth=best_meta_depth,
        learning_rate=0.1, eval_metric="logloss",
        n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
    )
    meta_xgb_final.fit(meta_train_feats, y_train)

    stack_train_time = time.time() - t_stack_start
    t0 = time.time()
    stack_pred = meta_xgb_final.predict(meta_test_feats)
    stack_test_time = time.time() - t0

    stack_m = compute_clf_metrics(y_test, stack_pred, le, "Stacked Ensemble",
                                  stack_train_time, stack_test_time)
    all_results.append(stack_m)
    print(f"  Total stack train time: {stack_train_time:.2f}s  |  Test time: {stack_test_time * 1000:.3f}ms")
    print(f"  Test Accuracy: {stack_m['accuracy_%']:.2f}%")
    print(classification_report(y_test, stack_pred, target_names=CLASS_ORDER, zero_division=0))

    plot_confusion_matrix(y_test, stack_pred, le,
                          "Stacked Ensemble",
                          f"{PLOTS_DIR}/stack_confusion_matrix.png")

    #Saving to be reloaded for test to work fel mona2sha
    stack_artifact = {
        "base_models": base_models,  # list of (name, fitted_model)
        "meta_model": meta_xgb_final,
        "imputer": imputer,
        "label_encoder": le,
    }
    with open(f"{MODELS_DIR}/stacked_ensemble.pkl", "wb") as f:
        pickle.dump(stack_artifact, f)
    print(f"  Saved model -> {MODELS_DIR}/stacked_ensemble.pkl")


    # -------------------------------------------------------------------------------------------------------------------#
    # Model 4 : Logistic Regression
    # -------------------------------------------------------------------------------------------------------------------#
    print("\n" + "=" * 65)
    print("  MODEL 4: Logistic Regression")
    print("=" * 65)

    # Logistic Regression needs scaled features — we scale the imputed matrix here.
    # StandardScaler is fit on train only to prevent leakage into test.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler_lr = StandardScaler()
    X_train_scaled = scaler_lr.fit_transform(X_train_imputed)
    X_test_scaled = scaler_lr.transform(X_test_imputed)

    # --- Hyperparameter 1: C (inverse regularization strength) ---
    # C controls how strongly regularization penalizes large coefficients.
    # Small C = strong regularization (underfits), Large C = weak regularization (overfits).
    # We fix solver='lbfgs' while varying C.
    FIXED_LR_SOLVER = "lbfgs"
    lr_C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    lr_C_cv = []
    print(f"\n  Tuning C (solver fixed = {FIXED_LR_SOLVER}):")
    for c in lr_C_values:
        lr = LogisticRegression(C=c, solver=FIXED_LR_SOLVER,
                                max_iter=1000, random_state=RANDOM_STATE)
        score = cross_val_score(lr, X_train_scaled, y_train,
                                cv=kfold, scoring="accuracy").mean()
        lr_C_cv.append(score)
        print(f"    C={c:<8}  CV Accuracy = {score:.4f}")
    best_lr_C = lr_C_values[int(np.argmax(lr_C_cv))]
    plot_hyperparameter_effect(
        "C (Regularization)", lr_C_values, lr_C_cv,
        "Logistic Regression", f"{PLOTS_DIR}/lr_tune_C.png"
    )
    tuning_records["LogisticRegression_C"] = {
        "values": lr_C_values, "cv_acc": lr_C_cv, "best": best_lr_C
    }

    # --- Hyperparameter 2: solver ---
    # Different solvers use different optimization algorithms.
    # lbfgs: limited-memory BFGS quasi-Newton method, good for multiclass.
    # saga: stochastic average gradient, supports L1/L2, fast on large datasets.
    # newton-cg: Newton conjugate gradient, good for multiclass.
    # We fix C at the best value found above while varying solver.
    lr_solver_values = ["lbfgs", "saga", "newton-cg"]
    lr_solver_cv = []
    print(f"\n  Tuning solver (C fixed = {best_lr_C}):")
    for solver in lr_solver_values:
        lr = LogisticRegression(C=best_lr_C, solver=solver,
                                max_iter=1000, random_state=RANDOM_STATE)
        score = cross_val_score(lr, X_train_scaled, y_train,
                                cv=kfold, scoring="accuracy").mean()
        lr_solver_cv.append(score)
        print(f"    solver={solver:<12}  CV Accuracy = {score:.4f}")
    best_lr_solver = lr_solver_values[int(np.argmax(lr_solver_cv))]
    plot_hyperparameter_effect(
        "solver", lr_solver_values, lr_solver_cv,
        "Logistic Regression", f"{PLOTS_DIR}/lr_tune_solver.png"
    )
    tuning_records["LogisticRegression_solver"] = {
        "values": lr_solver_values, "cv_acc": lr_solver_cv, "best": best_lr_solver
    }

    # --- Final Logistic Regression Model ---
    print(f"\n  Best hyperparams: C={best_lr_C}, solver={best_lr_solver}")
    lr_model = LogisticRegression(C=best_lr_C, solver=best_lr_solver,
                                  max_iter=1000, random_state=RANDOM_STATE)
    t0 = time.time()
    lr_model.fit(X_train_scaled, y_train)
    lr_train_time = time.time() - t0

    t0 = time.time()
    lr_pred = lr_model.predict(X_test_scaled)
    lr_test_time = time.time() - t0

    lr_m = compute_clf_metrics(y_test, lr_pred, le, "Logistic Regression",
                               lr_train_time, lr_test_time)
    all_results.append(lr_m)
    print(f"  Train time: {lr_train_time:.4f}s  |  Test time: {lr_test_time * 1000:.3f}ms")
    print(f"  Test Accuracy: {lr_m['accuracy_%']:.2f}%")
    print(classification_report(y_test, lr_pred, target_names=CLASS_ORDER, zero_division=0))

    plot_confusion_matrix(y_test, lr_pred, le,
                          "Logistic Regression",
                          f"{PLOTS_DIR}/lr_confusion_matrix.png")

    # Save model — include scaler because LR requires scaled input at test time
    lr_artifact = {
        "model": lr_model,
        "scaler": scaler_lr,
        "imputer": imputer,
        "label_encoder": le,
    }
    with open(f"{MODELS_DIR}/logistic_regression.pkl", "wb") as f:
        pickle.dump(lr_artifact, f)
    print(f"  Saved model -> {MODELS_DIR}/logistic_regression.pkl")

    # -------------------------------------------------------------------------------------------------------------------#
    # Model 5 : Support Vector Machine (SVM)
    # -------------------------------------------------------------------------------------------------------------------#
    print("\n" + "=" * 65)
    print("  MODEL 5: Support Vector Machine (SVM)")
    print("=" * 65)

    # SVM also requires scaled features.
    # We reuse X_train_scaled / X_test_scaled that were already created for LR above.
    # LinearSVC is used instead of SVC(kernel='rbf') because:
    #   - The dataset has ~9000 rows and 110 features → LinearSVC is much faster.
    #   - LinearSVC uses a one-vs-rest multiclass strategy by default.
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    # --- Hyperparameter 1: C (regularization) ---
    # Same interpretation as Logistic Regression: smaller C = more regularization.
    # We fix max_iter=3000 while varying C.
    FIXED_SVM_ITER = 3000
    svm_C_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    svm_C_cv = []
    print(f"\n  Tuning C (max_iter fixed = {FIXED_SVM_ITER}):")
    for c in svm_C_values:
        svm = LinearSVC(C=c, max_iter=FIXED_SVM_ITER, random_state=RANDOM_STATE)
        score = cross_val_score(svm, X_train_scaled, y_train,
                                cv=kfold, scoring="accuracy").mean()
        svm_C_cv.append(score)
        print(f"    C={c:<8}  CV Accuracy = {score:.4f}")
    best_svm_C = svm_C_values[int(np.argmax(svm_C_cv))]
    plot_hyperparameter_effect(
        "C (Regularization)", svm_C_values, svm_C_cv,
        "SVM (LinearSVC)", f"{PLOTS_DIR}/svm_tune_C.png"
    )
    tuning_records["SVM_C"] = {
        "values": svm_C_values, "cv_acc": svm_C_cv, "best": best_svm_C
    }

    # --- Hyperparameter 2: max_iter ---
    # max_iter controls how many optimization iterations the solver runs.
    # Too few → model may not converge. Too many → slower with no gain.
    # We fix C at best value while varying max_iter.
    svm_iter_values = [500, 1000, 3000]
    svm_iter_cv = []
    print(f"\n  Tuning max_iter (C fixed = {best_svm_C}):")
    for it in svm_iter_values:
        svm = LinearSVC(C=best_svm_C, max_iter=it, random_state=RANDOM_STATE)
        score = cross_val_score(svm, X_train_scaled, y_train,
                                cv=kfold, scoring="accuracy").mean()
        svm_iter_cv.append(score)
        print(f"    max_iter={it:<5}  CV Accuracy = {score:.4f}")
    best_svm_iter = svm_iter_values[int(np.argmax(svm_iter_cv))]
    plot_hyperparameter_effect(
        "max_iter", svm_iter_values, svm_iter_cv,
        "SVM (LinearSVC)", f"{PLOTS_DIR}/svm_tune_max_iter.png"
    )
    tuning_records["SVM_max_iter"] = {
        "values": svm_iter_values, "cv_acc": svm_iter_cv, "best": best_svm_iter
    }

    # --- Final SVM Model ---
    # CalibratedClassifierCV wraps LinearSVC to give it predict_proba() capability.
    # This is needed so the saved model can produce probabilities if required.
    print(f"\n  Best hyperparams: C={best_svm_C}, max_iter={best_svm_iter}")
    svm_base = LinearSVC(C=best_svm_C, max_iter=best_svm_iter, random_state=RANDOM_STATE)
    svm_model = CalibratedClassifierCV(svm_base, cv=3)

    t0 = time.time()
    svm_model.fit(X_train_scaled, y_train)
    svm_train_time = time.time() - t0

    t0 = time.time()
    svm_pred = svm_model.predict(X_test_scaled)
    svm_test_time = time.time() - t0

    svm_m = compute_clf_metrics(y_test, svm_pred, le, "SVM (LinearSVC)",
                                svm_train_time, svm_test_time)
    all_results.append(svm_m)
    print(f"  Train time: {svm_train_time:.4f}s  |  Test time: {svm_test_time * 1000:.3f}ms")
    print(f"  Test Accuracy: {svm_m['accuracy_%']:.2f}%")
    print(classification_report(y_test, svm_pred, target_names=CLASS_ORDER, zero_division=0))

    plot_confusion_matrix(y_test, svm_pred, le,
                          "SVM (LinearSVC)",
                          f"{PLOTS_DIR}/svm_confusion_matrix.png")

    # Save model — include scaler (same one used for LR, reuse is fine)
    svm_artifact = {
        "model": svm_model,
        "scaler": scaler_lr,  # same StandardScaler fitted on train
        "imputer": imputer,
        "label_encoder": le,
    }
    with open(f"{MODELS_DIR}/svm.pkl", "wb") as f:
        pickle.dump(svm_artifact, f)
    print(f"  Saved model -> {MODELS_DIR}/svm.pkl")

    #-------------------------------------------------------------------------------------------------------------------#
    #Bar Graphs
    #-------------------------------------------------------------------------------------------------------------------#
    print("\n[7] Generating required comparison charts ...")
    plot_accuracy_bar(all_results, f"{PLOTS_DIR}/comparison_accuracy.png")
    plot_train_time_bar(all_results, f"{PLOTS_DIR}/comparison_train_time.png")
    plot_test_time_bar(all_results, f"{PLOTS_DIR}/comparison_test_time.png")

    #-------------------------------------------------------------------------------------------------------------------#
    #Saving of metrics & tuning records
    #-------------------------------------------------------------------------------------------------------------------#

    print(f"\n[8] Saving metrics to {METRICS_PATH} ...")

    # Convert numpy types so json.dump works
    def to_python(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)):  return obj.tolist()
        return obj

    summary = {
        "dataset_shape": list(df_raw.shape),
        "target_thresholds": {"Low": f"0-{LOW_THRESH}",
                              "Medium": f"{LOW_THRESH + 1}-{HIGH_THRESH}",
                              "High": f"{HIGH_THRESH + 1}+"},
        "class_distribution": dist.to_dict(),
        "train_size": int(len(X_train_imputed)),
        "test_size": int(len(X_test_imputed)),
        "features_engineered": int(X.shape[1]),
        "hyperparameter_tuning": tuning_records,
        "models": {
            r["model"]: {k: to_python(v) for k, v in r.items() if k != "report"}
            for r in all_results
        },
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=to_python)

    #-------------------------------------------------------------------------------------------------------------------#
    #Final Result
    #-------------------------------------------------------------------------------------------------------------------#
    print("\n" + "=" * 65)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Model':<25} {'Accuracy':>10}  {'Train(s)':>10}  {'Test(ms)':>10}")
    print("  " + "-" * 60)
    for r in sorted(all_results, key=lambda x: -x["accuracy"]):
        print(f"  {r['model']:<25} {r['accuracy_%']:>9.2f}%  "
              f"{r['train_time']:>9.3f}s  {r['test_time'] * 1000:>9.3f}ms")

    print(f"\n  All plots saved in: {PLOTS_DIR}/")
    print(f"  All models saved in: {MODELS_DIR}/")
    print(f"  Metrics saved in:   {METRICS_PATH}")
    print("\n  Done!\n")

#-----------------------------------------------------------------------------------------------------------------------#
#hamdella 3al salama
#-----------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    main()





















