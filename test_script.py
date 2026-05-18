"""
test_script.py  —  Milestone 2 Test / Inference Script
=======================================================
Loads all 5 saved trained models from artifacts/models/ and makes
predictions on a new (unseen) test CSV file WITHOUT re-training.

Models supported:
    1. LightGBM               (artifacts/models/lightgbm.pkl)
    2. Gradient Boosting      (artifacts/models/gradient_boosting.pkl)
    3. Stacked Ensemble       (artifacts/models/stacked_ensemble.pkl)
       base: LightGBM + XGBoost + Random Forest + Extra Trees
       meta: XGBoost
    4. Logistic Regression    (artifacts/models/logistic_regression.pkl)
    5. SVM (LinearSVC)        (artifacts/models/svm.pkl)

Usage:
    python test_script.py  path/to/test_data.csv

Notes:
    - The test CSV does NOT need GamePopularity or RecommendationCount.
      If either is present the script will automatically evaluate accuracy.
    - Every feature name matches the training script exactly.
    - Missing value handling: any column absent from the test CSV is
      filled with 0 before imputation — no row is ever dropped.
    - LR and SVM require StandardScaler — the saved scaler from training
      is loaded from the .pkl artifact and applied automatically.

Outputs:
    - Console  : per-model accuracy + classification report (if labels exist)
    - predictions_output.csv : QueryID + one prediction column per model
"""

import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION  — must match milestone2_online_games_prediction.py exactly
# =============================================================================

MODELS_DIR  = "artifacts/models"
OUTPUT_CSV  = "predictions_output.csv"
CLASS_ORDER = ["Low", "Medium", "High"]

LOW_THRESH  = 100
HIGH_THRESH = 1000


# =============================================================================
# PREPROCESSING  (identical to training script)
# =============================================================================

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans every column the same way the training script does.
    Safe to call even when some columns are absent from the test CSV.
    """
    df = df.copy()

    # --- Name columns ---
    if "QueryName" in df.columns:
        df["QueryName"] = df["QueryName"].fillna(
            df.get("ResponseName", pd.Series("", index=df.index))
        ).str.strip()
    if "ResponseName" in df.columns:
        df["ResponseName"] = df["ResponseName"].str.strip()

    # --- ReleaseDate ---
    if "ReleaseDate" in df.columns:
        df["ReleaseDate"] = pd.to_datetime(df["ReleaseDate"], errors="coerce")

    # --- SteamSpy: clip negatives ---
    for col in ["SteamSpyOwners", "SteamSpyOwnersVariance",
                "SteamSpyPlayersEstimate", "SteamSpyPlayersVariance"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # --- Price columns ---
    if "PriceInitial"  in df.columns:
        df["PriceInitial"] = df["PriceInitial"].clip(lower=0)
    if "PriceFinal"    in df.columns:
        df["PriceFinal"]   = df["PriceFinal"].clip(lower=0)
    if "PriceCurrency" in df.columns:
        df["PriceCurrency"] = df["PriceCurrency"].replace("", "Unknown").fillna("Unknown")

    # --- Boolean columns → int ---
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)

    # --- String / text columns ---
    str_cols = [
        "SupportEmail", "SupportURL", "Website", "AboutText",
        "ShortDescrip", "DetailedDescrip", "PCMinReqsText", "PCRecReqsText",
        "LinuxMinReqsText", "LinuxRecReqsText", "MacMinReqsText", "MacRecReqsText",
        "Reviews", "SupportedLanguages", "LegalNotice", "DRMNotice",
        "ExtUserAcctNotice", "Background", "HeaderImage",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").str.strip()

    return df


# =============================================================================
# FEATURE ENGINEERING  (identical column names to training script)
# =============================================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the feature matrix with EXACTLY the same column names
    as the training script (milestone2_online_games_prediction.py).
    Uses .get() with safe defaults so missing columns never crash.
    """
    parts = []

    # ── Date features ─────────────────────────────────────────────────────────
    if "ReleaseDate" in df.columns:
        ref = df["ReleaseDate"].dropna().max()
        parts.append(pd.DataFrame({
            "Release_Year":    df["ReleaseDate"].dt.year,
            "Release_Month":   df["ReleaseDate"].dt.month,
            "Release_WeekDay": df["ReleaseDate"].dt.dayofweek,
            "ReleaseAgeDays":  (ref - df["ReleaseDate"]).dt.days,
        }, index=df.index))
    else:
        parts.append(pd.DataFrame({
            "Release_Year": np.nan, "Release_Month": np.nan,
            "Release_WeekDay": np.nan, "ReleaseAgeDays": np.nan,
        }, index=df.index))

    # ── Price features ────────────────────────────────────────────────────────
    init    = df.get("PriceInitial", pd.Series(0.0, index=df.index))
    final   = df.get("PriceFinal",   pd.Series(0.0, index=df.index))
    is_free = df.get("IsFree",       pd.Series(0,   index=df.index))
    ratio   = np.where(init > 0, (init - final) / init, 0.0)
    parts.append(pd.DataFrame({
        "LoggedFinalPrice":     np.log1p(final),
        "LoggedInitialPrice":   np.log1p(init),
        "PriceDiscountRatio":   np.clip(ratio, 0.0, 1.0),
        "IsFreePrice":          (final == 0).astype(int),
        "IsPremium":            (final >= 20).astype(int),
        "IsFreePrice_Mismatch": ((final == 0) & (is_free == 0)).astype(int),
    }, index=df.index))

    # ── SteamSpy features ─────────────────────────────────────────────────────
    owners   = df.get("SteamSpyOwners",          pd.Series(0, index=df.index))
    players  = df.get("SteamSpyPlayersEstimate",  pd.Series(0, index=df.index))
    own_var  = df.get("SteamSpyOwnersVariance",   pd.Series(0, index=df.index))
    play_var = df.get("SteamSpyPlayersVariance",  pd.Series(0, index=df.index))
    parts.append(pd.DataFrame({
        "Logged_SteamSpy_Owners":          np.log1p(owners),
        "Logged_SteamSpy_Players":         np.log1p(players),
        "Logged_SteamSpy_Player_Variance": np.log1p(play_var),
        "Logged_SteamSpy_Owner_Variance":  np.log1p(own_var),
        "Logged_OwnerShip_Lower":          np.log1p(np.maximum(0, owners - own_var)),
        "Logged_OwnerShip_Upper":          np.log1p(owners + own_var),
        "Owners_Rank":                     owners.rank(pct=True),
        "Players_Rank":                    players.rank(pct=True),
        "Engagement_Ratio":                np.where(owners > 0, players / owners, 0.0),
    }, index=df.index))

    # ── Metacritic features ───────────────────────────────────────────────────
    score = df.get("Metacritic", pd.Series(0, index=df.index))
    parts.append(pd.DataFrame({
        "Metacritic_Score": score,
        "Has_Metacritic":   (score > 0).astype(int),
    }, index=df.index))

    # ── Content richness ──────────────────────────────────────────────────────
    parts.append(pd.DataFrame({
        "LoggedMovieCount":       np.log1p(df.get("MovieCount",                  pd.Series(0, index=df.index))),
        "LoggedScreenshotCount":  np.log1p(df.get("ScreenshotCount",             pd.Series(0, index=df.index))),
        "LoggedDLCCount":         np.log1p(df.get("DLCCount",                    pd.Series(0, index=df.index))),
        "LoggedAchievementCount": np.log1p(df.get("AchievementCount",            pd.Series(0, index=df.index))),
        "LoggedAchievementHL":    np.log1p(df.get("AchievementHighlightedCount", pd.Series(0, index=df.index))),
        "LoggedPackageCount":     np.log1p(df.get("PackageCount",                pd.Series(0, index=df.index))),
        "LoggedDeveloperCount":   np.log1p(df.get("DeveloperCount",              pd.Series(0, index=df.index))),
        "LoggedPublisherCount":   np.log1p(df.get("PublisherCount",              pd.Series(0, index=df.index))),
        "LoggedDemoCount":        np.log1p(df.get("DemoCount",                   pd.Series(0, index=df.index))),
        "LoggedContentScore":     (
            np.log1p(df.get("MovieCount",        pd.Series(0, index=df.index)))
            + np.log1p(df.get("ScreenshotCount", pd.Series(0, index=df.index)))
            + np.log1p(df.get("DLCCount",        pd.Series(0, index=df.index)))
        ),
    }, index=df.index))

    # ── Language features ─────────────────────────────────────────────────────
    sl = df.get("SupportedLanguages", pd.Series("", index=df.index)).fillna("")
    cleaned    = sl.str.replace(r"\*[^*]*\*?languages[^*]*", "", regex=True)
    lang_count = cleaned.str.split().apply(len)
    parts.append(pd.DataFrame({
        "language_count":        lang_count,
        "Logged_language_count": np.log1p(lang_count),
        "Multilingual":          (lang_count > 1).astype(int),
    }, index=df.index))

    # ── Platform features ─────────────────────────────────────────────────────
    parts.append(pd.DataFrame({
        "PlatformCount": (
            df.get("PlatformWindows", pd.Series(0, index=df.index))
            + df.get("PlatformLinux", pd.Series(0, index=df.index))
            + df.get("PlatformMac",   pd.Series(0, index=df.index))
        ),
        "SupportsLinux": df.get("PlatformLinux", pd.Series(0, index=df.index)),
        "SupportsMac":   df.get("PlatformMac",   pd.Series(0, index=df.index)),
    }, index=df.index))

    # ── Multiplayer features ──────────────────────────────────────────────────
    is_multi = (
        df.get("CategoryMultiplayer", pd.Series(0, index=df.index)).astype(bool)
        | df.get("CategoryCoop",      pd.Series(0, index=df.index)).astype(bool)
        | df.get("CategoryMMO",       pd.Series(0, index=df.index)).astype(bool)
    ).astype(int)
    log_owners = np.log1p(owners)
    parts.append(pd.DataFrame({
        "IsMultiplayer":          is_multi,
        "Language_X_Multiplayer": lang_count * is_multi,
        "Owners_X_Multiplayer":   log_owners * is_multi,
        "Action_X_Multiplayer":   df.get("GenreIsAction", pd.Series(0, index=df.index)) * is_multi,
    }, index=df.index))

    # ── Age / maturity features ───────────────────────────────────────────────
    age = df.get("RequiredAge", pd.Series(0, index=df.index))
    parts.append(pd.DataFrame({
        "RequiredAge": age,
        "IsMature":    (age >= 17).astype(int),
    }, index=df.index))

    # ── Text richness features ────────────────────────────────────────────────
    text_out = {}
    for col in ["AboutText", "ShortDescrip", "DetailedDescrip",
                "PCMinReqsText", "PCRecReqsText", "Reviews"]:
        s = df.get(col, pd.Series("", index=df.index)).fillna("")
        text_out[f"{col}_len"] = np.log1p(s.str.len())
        text_out[f"{col}_has"] = (s != "").astype(int)
    for col in ["LinuxMinReqsText", "LinuxRecReqsText",
                "MacMinReqsText",   "MacRecReqsText"]:
        s = df.get(col, pd.Series("", index=df.index)).fillna("")
        text_out[f"has_{col}"] = (s != "").astype(int)
    for col in ["SupportEmail", "SupportURL", "Website", "LegalNotice",
                "DRMNotice", "ExtUserAcctNotice", "Background", "HeaderImage"]:
        s = df.get(col, pd.Series("", index=df.index)).fillna("")
        text_out[f"has_{col}"] = (s != "").astype(int)
    parts.append(pd.DataFrame(text_out, index=df.index))

    # ── Interaction features ──────────────────────────────────────────────────
    log_players = np.log1p(players)
    log_own_var = np.log1p(own_var)
    has_meta    = (score > 0).astype(float)
    parts.append(pd.DataFrame({
        "Owners_X_Metacritic":     log_owners * score,
        "Platers_X_Metacritic":    log_players * score,
        "Owners_X_has_Metacritic": log_owners * has_meta,
        "Var_X_owners":            log_own_var * log_owners,
    }, index=df.index))

    # ── Boolean Genre / Category columns (already int after preprocessing) ────
    bool_int_cols = [
        c for c in df.select_dtypes(include=[int, "int64"]).columns
        if c.startswith((
            "Category", "Genre", "Platform", "PCReqs",
            "LinuxReqs", "MacReqs", "IsFree", "FreeVer",
            "Purchase", "Subscription", "Controller",
        ))
    ]
    if bool_int_cols:
        parts.append(df[bool_int_cols])

    # ── Extra numeric columns not already handled ─────────────────────────────
    already = {
        "PriceInitial", "PriceFinal", "RequiredAge", "SteamSpyOwners",
        "SteamSpyOwnersVariance", "SteamSpyPlayersEstimate", "SteamSpyPlayersVariance",
        "Metacritic", "MovieCount", "ScreenshotCount", "DLCCount", "AchievementCount",
        "AchievementHighlightedCount", "PackageCount", "DeveloperCount", "PublisherCount",
        "DemoCount", "RecommendationCount",
    }
    extra = [c for c in df.select_dtypes(include=[np.number]).columns if c not in already]
    if extra:
        parts.append(df[extra])

    # ── PriceCurrency one-hot ─────────────────────────────────────────────────
    if "PriceCurrency" in df.columns:
        parts.append(pd.get_dummies(df["PriceCurrency"], prefix="currency"))

    X = pd.concat(parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X


# =============================================================================
# HELPER — align test features to training columns then impute
# =============================================================================

def align_and_impute(X_test: pd.DataFrame, imputer) -> pd.DataFrame:
    """
    Aligns the test feature matrix to the exact columns the imputer
    was fitted on during training. Any column completely missing from
    the test set is filled with 0 before the imputer applies medians.
    This guarantees no test row is ever dropped.
    """
    if hasattr(imputer, "feature_names_in_"):
        train_cols = list(imputer.feature_names_in_)
    else:
        train_cols = list(range(imputer.statistics_.shape[0]))

    X_aligned = X_test.reindex(columns=train_cols, fill_value=0)
    X_imp = pd.DataFrame(
        imputer.transform(X_aligned),
        columns=train_cols,
        index=X_test.index,
    )
    return X_imp


# =============================================================================
# HELPER — align imputed features to scaler columns then scale
# =============================================================================

def align_and_scale(X_imp: pd.DataFrame, scaler) -> np.ndarray:
    """
    Aligns the imputed feature matrix to the exact columns the scaler
    was fitted on during training, then applies transform().
    Used by Logistic Regression and SVM which require scaled input.
    """
    if hasattr(scaler, "feature_names_in_"):
        scaler_cols = list(scaler.feature_names_in_)
    else:
        # Fallback: scaler was fitted on a numpy array — column count must match
        scaler_cols = X_imp.columns.tolist()

    X_aligned = X_imp.reindex(columns=scaler_cols, fill_value=0)
    return scaler.transform(X_aligned)


# =============================================================================
# HELPER — derive ground truth from whichever column is available
# =============================================================================

def get_ground_truth(df_test: pd.DataFrame):
    """
    Returns (y_true_encoded, label_encoder, valid_mask).
    Returns (None, None, None) if no label column exists.
    """
    le_gt = LabelEncoder()
    le_gt.fit(CLASS_ORDER)

    if "GamePopularity" in df_test.columns:
        valid = df_test["GamePopularity"].isin(CLASS_ORDER)
        if valid.any():
            y = le_gt.transform(df_test.loc[valid, "GamePopularity"])
            print(f"  Ground truth found (GamePopularity column).")
            print(f"  Distribution: {dict(df_test['GamePopularity'].value_counts())}")
            return y, le_gt, valid.values

    if "RecommendationCount" in df_test.columns:
        rc = df_test["RecommendationCount"]
        y_labels = pd.cut(
            rc,
            bins=[-1, LOW_THRESH, HIGH_THRESH, float("inf")],
            labels=CLASS_ORDER,
        )
        valid = y_labels.notna()
        if valid.any():
            y = le_gt.transform(y_labels[valid])
            print(f"  Ground truth derived from RecommendationCount.")
            return y, le_gt, valid.values

    return None, None, None


# =============================================================================
# HELPER — shared evaluation and reporting logic
# =============================================================================

def evaluate_and_report(y_true, preds_enc, valid_mask, has_labels,
                        model_name, infer_time, all_summary):
    """
    Prints accuracy + classification report if ground truth is available.
    Appends a summary row to all_summary either way.
    """
    print(f"  Inference time : {infer_time*1000:.3f} ms")
    if has_labels:
        preds_for_eval = preds_enc[valid_mask] if valid_mask is not None else preds_enc
        acc = accuracy_score(y_true, preds_for_eval)
        print(f"  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
        print()
        print(classification_report(y_true, preds_for_eval,
                                    target_names=CLASS_ORDER, zero_division=0))
        all_summary.append({
            "Model":           model_name,
            "Accuracy":        f"{acc*100:.2f}%",
            "Inference (ms)":  f"{infer_time*1000:.3f}",
        })
    else:
        all_summary.append({
            "Model":           model_name,
            "Accuracy":        "N/A",
            "Inference (ms)":  f"{infer_time*1000:.3f}",
        })


# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

def predict_on_new_data(test_csv_path: str):

    print("=" * 65)
    print("  MILESTONE 2 — TEST / INFERENCE SCRIPT")
    print("  Models: LightGBM | Gradient Boosting | Stacked Ensemble")
    print("          Logistic Regression | SVM (LinearSVC)")
    print("=" * 65)

    # ── 1. Load & prepare test data ───────────────────────────────────────────
    print(f"\n[1] Loading test data: {test_csv_path}")
    df_test = pd.read_csv(test_csv_path)
    print(f"    Rows: {df_test.shape[0]}  |  Columns: {df_test.shape[1]}")

    df_test = preprocessing(df_test)
    X_test  = feature_engineering(df_test)
    print(f"    Feature matrix: {X_test.shape}")

    # ── 2. Ground truth (optional) ────────────────────────────────────────────
    print("\n[2] Checking for ground truth labels ...")
    y_true, le_gt, valid_mask = get_ground_truth(df_test)
    has_labels = y_true is not None

    if not has_labels:
        print("    No ground truth found — predictions only, no accuracy reported.")

    # ── 3. Output dataframe scaffold ──────────────────────────────────────────
    query_ids  = df_test["QueryID"].values if "QueryID" in df_test.columns else df_test.index
    results_df = pd.DataFrame({"QueryID": query_ids})
    all_summary = []

    # =========================================================================
    # MODEL 1 — LightGBM
    # =========================================================================
    lgb_path = os.path.join(MODELS_DIR, "lightgbm.pkl")
    print("\n" + "-" * 65)
    print("  [Model 1] LightGBM")
    print("-" * 65)

    if not os.path.exists(lgb_path):
        print(f"  File not found: {lgb_path}  — Skipping.")
    else:
        with open(lgb_path, "rb") as f:
            lgb_art = pickle.load(f)

        X_imp = align_and_impute(X_test, lgb_art["imputer"])

        t0            = time.time()
        lgb_preds_enc = lgb_art["model"].predict(X_imp)
        infer_time    = time.time() - t0

        lgb_preds_lbl = lgb_art["label_encoder"].inverse_transform(lgb_preds_enc)
        results_df["LightGBM_Prediction"] = lgb_preds_lbl

        evaluate_and_report(y_true, lgb_preds_enc, valid_mask,
                             has_labels, "LightGBM", infer_time, all_summary)

    # =========================================================================
    # MODEL 2 — Gradient Boosting
    # =========================================================================
    gb_path = os.path.join(MODELS_DIR, "gradient_boosting.pkl")
    print("\n" + "-" * 65)
    print("  [Model 2] Gradient Boosting")
    print("-" * 65)

    if not os.path.exists(gb_path):
        print(f"  File not found: {gb_path}  — Skipping.")
    else:
        with open(gb_path, "rb") as f:
            gb_art = pickle.load(f)

        X_imp = align_and_impute(X_test, gb_art["imputer"])

        t0           = time.time()
        gb_preds_enc = gb_art["model"].predict(X_imp)
        infer_time   = time.time() - t0

        gb_preds_lbl = gb_art["label_encoder"].inverse_transform(gb_preds_enc)
        results_df["GradientBoosting_Prediction"] = gb_preds_lbl

        evaluate_and_report(y_true, gb_preds_enc, valid_mask,
                             has_labels, "Gradient Boosting", infer_time, all_summary)

    # =========================================================================
    # MODEL 3 — Stacked Ensemble  (LGB + XGB + RF + ET → XGB meta-learner)
    # =========================================================================
    stack_path = os.path.join(MODELS_DIR, "stacked_ensemble.pkl")
    print("\n" + "-" * 65)
    print("  [Model 3] Stacked Ensemble (LGB + XGB + RF + ET → XGB meta)")
    print("-" * 65)

    if not os.path.exists(stack_path):
        print(f"  File not found: {stack_path}  — Skipping.")
    else:
        with open(stack_path, "rb") as f:
            stack_art = pickle.load(f)

        X_imp = align_and_impute(X_test, stack_art["imputer"])

        print("  Running base models ...")
        t0 = time.time()
        meta_feats = np.hstack([
            bm.predict_proba(X_imp)
            for _, bm in stack_art["base_models"]
        ])
        stack_preds_enc = stack_art["meta_model"].predict(meta_feats)
        infer_time      = time.time() - t0

        stack_preds_lbl = stack_art["label_encoder"].inverse_transform(stack_preds_enc)
        results_df["StackedEnsemble_Prediction"] = stack_preds_lbl

        print(f"  Inference time : {infer_time*1000:.3f} ms  (all 4 base + meta)")
        if has_labels:
            preds_for_eval = stack_preds_enc[valid_mask] if valid_mask is not None else stack_preds_enc
            acc = accuracy_score(y_true, preds_for_eval)
            print(f"  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
            print()
            print(classification_report(y_true, preds_for_eval,
                                        target_names=CLASS_ORDER, zero_division=0))
            all_summary.append({"Model": "Stacked Ensemble",
                                 "Accuracy": f"{acc*100:.2f}%",
                                 "Inference (ms)": f"{infer_time*1000:.3f}"})
        else:
            all_summary.append({"Model": "Stacked Ensemble", "Accuracy": "N/A",
                                 "Inference (ms)": f"{infer_time*1000:.3f}"})

    # =========================================================================
    # MODEL 4 — Logistic Regression
    # =========================================================================
    lr_path = os.path.join(MODELS_DIR, "logistic_regression.pkl")
    print("\n" + "-" * 65)
    print("  [Model 4] Logistic Regression")
    print("-" * 65)

    if not os.path.exists(lr_path):
        print(f"  File not found: {lr_path}  — Skipping.")
    else:
        with open(lr_path, "rb") as f:
            lr_art = pickle.load(f)

        # Step 1: impute using training medians
        X_imp = align_and_impute(X_test, lr_art["imputer"])

        # Step 2: scale using the StandardScaler fitted during training
        # LR requires standardised features — the scaler is saved in the artifact
        X_scaled = align_and_scale(X_imp, lr_art["scaler"])

        t0           = time.time()
        lr_preds_enc = lr_art["model"].predict(X_scaled)
        infer_time   = time.time() - t0

        lr_preds_lbl = lr_art["label_encoder"].inverse_transform(lr_preds_enc)
        results_df["LogisticRegression_Prediction"] = lr_preds_lbl

        evaluate_and_report(y_true, lr_preds_enc, valid_mask,
                             has_labels, "Logistic Regression", infer_time, all_summary)

    # =========================================================================
    # MODEL 5 — SVM (LinearSVC)
    # =========================================================================
    svm_path = os.path.join(MODELS_DIR, "svm.pkl")
    print("\n" + "-" * 65)
    print("  [Model 5] SVM (LinearSVC)")
    print("-" * 65)

    if not os.path.exists(svm_path):
        print(f"  File not found: {svm_path}  — Skipping.")
    else:
        with open(svm_path, "rb") as f:
            svm_art = pickle.load(f)

        # Step 1: impute using training medians
        X_imp = align_and_impute(X_test, svm_art["imputer"])

        # Step 2: scale using the same StandardScaler saved in the artifact
        # SVM requires standardised features for correct margin computation
        X_scaled = align_and_scale(X_imp, svm_art["scaler"])

        t0            = time.time()
        svm_preds_enc = svm_art["model"].predict(X_scaled)
        infer_time    = time.time() - t0

        svm_preds_lbl = svm_art["label_encoder"].inverse_transform(svm_preds_enc)
        results_df["SVM_Prediction"] = svm_preds_lbl

        evaluate_and_report(y_true, svm_preds_enc, valid_mask,
                             has_labels, "SVM (LinearSVC)", infer_time, all_summary)

    # =========================================================================
    # SUMMARY TABLE + SAVE PREDICTIONS
    # =========================================================================
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    if all_summary:
        summary_df = pd.DataFrame(all_summary)
        print(summary_df.to_string(index=False))

    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Predictions saved → {OUTPUT_CSV}")
    print(f"  Columns: {list(results_df.columns)}")
    print()
    print(results_df.head(10).to_string(index=False))
    print("\nDone!")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_script.py  path/to/test_data.csv")
        sys.exit(1)
    predict_on_new_data(sys.argv[1])