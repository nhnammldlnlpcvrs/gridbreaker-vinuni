"""
Production-grade modeling utilities for VinDatathon 2026.
Implements PLAN_TRAINING.md requirements:
  - Purged Walk-Forward CV (gap = horizon, prevents leakage)
  - Composite objective (0.4*MAE + 0.4*RMSE + 0.2*(1-R2))
  - Horizon-aware evaluation
  - Ensemble methods (weighted, stacking, rank-average)
  - Recursive forecasting engine
  - Post-processing (clipping, smoothing, residual correction)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from scipy.optimize import minimize
from scipy.stats import rankdata
import warnings

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# 1. VALIDATION STRATEGY — Purged Walk-Forward CV
# ═════════════════════════════════════════════════════════════════════════════

class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross-Validation for long-horizon forecasting.

    Design decisions (per PLAN_TRAINING.md §1):
    - Expanding training window (uses all history up to split point)
    - Purge gap between train/val prevents rolling feature leakage
    - Each validation fold = `horizon` days (realistic forecast simulation)
    - Min training size enforced for model stability

    Why gap=14 in original is broken:
    Features use lags up to 730 days. A 14-day gap allows rolling windows
    and EWM features to leak information from training into validation,
    producing overly optimistic CV scores that don't match leaderboard.
    """

    def __init__(self, n_splits=3, horizon=365, purge_days=90, min_train_days=730):
        self.n_splits = n_splits
        self.horizon = horizon
        self.purge_days = purge_days
        self.min_train_days = min_train_days

    def split(self, X, dates=None):
        """Generate (train_idx, val_idx) pairs.

        Parameters
        ----------
        X : array-like
            Feature matrix (used only for length).
        dates : array-like, optional
            Date labels for chronological ordering.

        Yields
        ------
        train_idx, val_idx : np.ndarray
        """
        n = len(X)
        if dates is not None:
            sort_idx = np.argsort(dates)
        else:
            sort_idx = np.arange(n)

        for i in range(self.n_splits):
            remaining = self.n_splits - i - 1
            val_end   = n - remaining * (self.horizon + self.purge_days)
            val_start = val_end - self.horizon
            train_end = val_start - self.purge_days

            if train_end < self.min_train_days:
                continue

            yield sort_idx[:train_end], sort_idx[val_start:val_end]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ═════════════════════════════════════════════════════════════════════════════
# 2. COMPOSITE OBJECTIVE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def composite_score(y_true, y_pred, baseline_mae=None, baseline_rmse=None):
    """
    Competition-grade composite: 0.4*MAE_norm + 0.4*RMSE_norm + 0.2*(1-R²)

    Lower is better. Each component normalized by mean-predictor baseline
    so MAE and RMSE (different scales) are comparable.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2   = r2_score(yt, yp)

    if baseline_mae is None:
        baseline_mae = mean_absolute_error(yt, np.full_like(yt, yt.mean()))
    if baseline_rmse is None:
        baseline_rmse = np.sqrt(mean_squared_error(yt, np.full_like(yt, yt.mean())))

    norm_mae  = mae / baseline_mae if baseline_mae > 0 else 0.0
    norm_rmse = rmse / baseline_rmse if baseline_rmse > 0 else 0.0

    return 0.4 * norm_mae + 0.4 * norm_rmse + 0.2 * (1.0 - r2)


def evaluate_predictions(y_true, y_pred, prefix=""):
    """Return full metrics dict for model evaluation."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2   = r2_score(yt, yp)
    mape = np.mean(np.abs((yt - yp) / (yt + 1e-9))) * 100.0

    base_mae  = mean_absolute_error(yt, np.full_like(yt, yt.mean()))
    base_rmse = np.sqrt(mean_squared_error(yt, np.full_like(yt, yt.mean())))

    return {
        f"{prefix}mae":       mae,
        f"{prefix}rmse":      rmse,
        f"{prefix}r2":        r2,
        f"{prefix}mape":      mape,
        f"{prefix}composite": composite_score(yt, yp, base_mae, base_rmse),
    }


def horizon_breakdown(dates, y_true, y_pred, horizons=None):
    """
    Evaluate forecast quality by horizon bucket.

    horizons: dict of {name: (start_day_1indexed, end_day_1indexed)}
    """
    if horizons is None:
        horizons = {
            "short":  (1, 30),
            "medium": (31, 180),
            "long":   (181, 548),
        }

    results = {}
    n = len(y_true)
    for name, (lo, hi) in horizons.items():
        mask = (np.arange(n) >= lo - 1) & (np.arange(n) <= min(hi - 1, n - 1))
        if mask.sum() > 0:
            results[name] = evaluate_predictions(
                np.array(y_true)[mask],
                np.array(y_pred)[mask],
                prefix=f"{name}_",
            )
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 3. ENSEMBLE METHODS
# ═════════════════════════════════════════════════════════════════════════════

def weighted_blend(predictions_dict, weights_dict):
    """Weighted average of model predictions."""
    result = np.zeros_like(next(iter(predictions_dict.values())), dtype=np.float64)
    total_w = sum(weights_dict.values())
    for name, preds in predictions_dict.items():
        w = weights_dict.get(name, 1.0)
        result += w * np.asarray(preds, dtype=np.float64)
    return result / total_w


def rank_average(predictions_dict):
    """Rank-average ensemble — robust to outlier predictions."""
    result = np.zeros_like(next(iter(predictions_dict.values())), dtype=np.float64)
    for preds in predictions_dict.values():
        result += rankdata(np.asarray(preds)) / len(preds)
    return result / len(predictions_dict)


def median_ensemble(predictions_dict):
    """Element-wise median — robust to extreme predictions."""
    stack = np.column_stack([np.asarray(p) for p in predictions_dict.values()])
    return np.median(stack, axis=1)


def optimize_ensemble_weights(predictions_dict, y_true, non_negative=True, metric="composite"):
    """
    Learn optimal blending weights via constrained optimization.

    Parameters
    ----------
    predictions_dict : dict of {name: predictions_array}
    y_true : array-like
    non_negative : bool
        If True, weights constrained to [0, 1].
    metric : str
        "composite" (default), "mae", or "rmse".

    Returns
    -------
    weights : dict of {name: weight}
    """
    names = list(predictions_dict.keys())
    preds_matrix = np.column_stack([np.asarray(predictions_dict[n], dtype=np.float64)
                                     for n in names])
    yt = np.asarray(y_true, dtype=np.float64)

    base_mae  = mean_absolute_error(yt, np.full_like(yt, yt.mean()))
    base_rmse = np.sqrt(mean_squared_error(yt, np.full_like(yt, yt.mean())))

    def _objective(w):
        blended = preds_matrix @ w
        if metric == "mae":
            return mean_absolute_error(yt, blended)
        elif metric == "rmse":
            return np.sqrt(mean_squared_error(yt, blended))
        else:
            return composite_score(yt, blended, base_mae, base_rmse)

    n = len(names)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0) for _ in range(n)] if non_negative else [(-0.5, 1.5) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        _objective, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return dict(zip(names, np.maximum(result.x, 0) / np.maximum(result.x, 0).sum()))


def horizon_specific_blend(predictions_dict, y_true, dates, horizons=None):
    """
    Learn different blending weights for each horizon bucket.
    Returns a function: f(step_index) -> weights_dict
    """
    if horizons is None:
        horizons = {"short": 30, "medium": 180, "long": 9999}

    horizon_weights = {}
    n = len(y_true)

    for name, cutoff in horizons.items():
        mask = np.arange(n) < cutoff
        if mask.sum() > 0:
            sub_preds = {k: np.array(v)[mask] for k, v in predictions_dict.items()}
            sub_y = np.array(y_true)[mask]
            w = optimize_ensemble_weights(sub_preds, sub_y)
            horizon_weights[name] = (cutoff, w)

    def get_weights(step_idx):
        for name, (cutoff, w) in sorted(horizon_weights.items(), key=lambda x: x[1][0]):
            if step_idx < cutoff:
                return w
        return list(horizon_weights.values())[-1][1]

    return get_weights, horizon_weights


# ═════════════════════════════════════════════════════════════════════════════
# 4. RECURSIVE FORECASTING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def recursive_forecast(
    models_dict,
    ensemble_weights,
    initial_df,
    test_dates,
    feature_cols,
    feature_recompute_fn,
    target_cols=("Revenue", "COGS"),
    post_process_fn=None,
    verbose=True,
):
    """
    Iterative day-by-day forecasting with full feature recomputation.

    For each test date:
      1. Extract feature row from current state
      2. Get prediction from each model
      3. Blend with ensemble weights
      4. Insert prediction back into dataframe
      5. Recompute ALL derived features (lag, rolling, EWM, cross-ratios)
      6. Continue to next date

    Parameters
    ----------
    models_dict : dict of {name: {"Revenue": model, "COGS": model}}
        Trained models for each target.
    ensemble_weights : dict of {name: weight}
    initial_df : pd.DataFrame
        Full feature dataframe with correct index (Date).
    test_dates : list of pd.Timestamp
    feature_cols : list of str
    feature_recompute_fn : callable(df) -> df
        Function that recomputes all derived features.
    target_cols : tuple
    post_process_fn : callable(predictions, step) -> predictions, optional
    verbose : bool

    Returns
    -------
    pd.DataFrame with predictions filled in.
    """
    df = initial_df.copy()

    for i, d in enumerate(test_dates):
        if verbose and i % 100 == 0:
            print(f"  Day {i+1:>3}/{len(test_dates)}  ({d.date()})")

        row = df.loc[[d], feature_cols].fillna(0)

        for tgt in target_cols:
            preds = {}
            for name, models in models_dict.items():
                if tgt in models and models[tgt] is not None:
                    p = float(models[tgt].predict(row)[0])
                    preds[name] = max(p, 0)

            if preds:
                blended = 0.0
                total_w = 0.0
                for name, p in preds.items():
                    w = ensemble_weights.get(name, 1.0 / len(preds))
                    blended += w * p
                    total_w += w
                df.loc[d, tgt] = blended / max(total_w, 1e-9)

        # Apply post-processing
        if post_process_fn is not None:
            for tgt in target_cols:
                if pd.notna(df.loc[d, tgt]):
                    df.loc[d, tgt] = post_process_fn(df.loc[d, tgt], i, tgt)

        # Recompute features for next iteration
        if i < len(test_dates) - 1:
            df = feature_recompute_fn(df)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# 5. POST-PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def build_historical_bounds(sales, target_col):
    """Extract daily bounds from historical data for clipping."""
    sales = sales.copy()
    sales["month"] = sales["Date"].dt.month
    sales["day"]   = sales["Date"].dt.day

    stats = sales.groupby(["month", "day"])[target_col].agg(
        p01="quantile", p05="quantile", p95="quantile", p99="quantile",
        mean="mean", std="std",
    )
    stats.columns = [f"{target_col}_{c}" for c in stats.columns]
    stats = stats.reset_index()
    return stats


def clip_to_historical(pred_series, date_series, bounds_df, target_col, multiplier=3.0):
    """
    Clip predictions to [p01, mean + multiplier * std] based on same calendar day.
    Prevents catastrophic RMSE spikes from runaway recursive forecasts.
    """
    result = pred_series.copy()
    for i, (d, p) in enumerate(zip(date_series, pred_series)):
        mo, day = d.month, d.day
        match = bounds_df[(bounds_df["month"] == mo) & (bounds_df["day"] == day)]
        if len(match) > 0:
            lo = match[f"{target_col}_p01"].values[0]
            hi = match[f"{target_col}_mean"].values[0] + multiplier * match[f"{target_col}_std"].values[0]
            result[i] = np.clip(p, max(0, lo), hi)
    return result


def exponential_smooth(predictions, alpha=0.3):
    """Apply exponential smoothing to forecast trajectory."""
    smoothed = np.array(predictions, dtype=np.float64).copy()
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha * smoothed[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def residual_correct(recent_actuals, recent_preds, future_preds, decay=0.9):
    """
    Apply residual correction: adjust future predictions based on recent bias.
    Uses exponentially decaying weight on recent residuals.
    """
    residuals = np.array(recent_actuals) - np.array(recent_preds)
    n = len(residuals)
    weights = decay ** np.arange(n)[::-1]
    weights /= weights.sum()
    bias = np.sum(residuals * weights)
    return np.array(future_preds) + bias


# ═════════════════════════════════════════════════════════════════════════════
# 6. DRIFT & STABILITY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def drift_analysis(y_true, y_pred, window=30):
    """
    Compute rolling error metrics to detect forecast drift.
    Returns DataFrame with rolling MAE, RMSE, bias for each window.
    """
    errors = np.array(y_true) - np.array(y_pred)
    n = len(errors)

    records = []
    for i in range(0, n - window + 1, window):
        e = errors[i:i + window]
        records.append({
            "start_idx": i,
            "end_idx": i + window,
            "rolling_mae": np.mean(np.abs(e)),
            "rolling_rmse": np.sqrt(np.mean(e ** 2)),
            "rolling_bias": np.mean(e),
            "rolling_std": np.std(e),
        })
    return pd.DataFrame(records)


def fold_stability_analysis(cv_results):
    """
    Analyze stability across CV folds.
    Returns coefficient of variation for each metric.
    """
    metrics = ["mae", "rmse", "r2"]
    stats = {}
    for m in metrics:
        key = f"fold_{m}"
        if key in cv_results:
            vals = cv_results[key]
            stats[f"{m}_mean"] = np.mean(vals)
            stats[f"{m}_std"]  = np.std(vals)
            stats[f"{m}_cv"]   = np.std(vals) / (np.mean(vals) + 1e-9)
    return stats


# ═════════════════════════════════════════════════════════════════════════════
# 7. OPTUNA INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════

def make_optuna_objective(
    X, y, model_class, cv, param_fn,
    baseline_mae=None, baseline_rmse=None,
):
    """
    Factory: create an Optuna objective function that minimizes composite_score.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    model_class : class
        LGBMRegressor, XGBRegressor, CatBoostRegressor, etc.
    cv : PurgedWalkForwardCV
    param_fn : callable(trial) -> dict
    baseline_mae, baseline_rmse : float, optional

    Returns
    -------
    objective(trial) -> float
    """
    if baseline_mae is None:
        baseline_mae = mean_absolute_error(y, np.full_like(y, y.mean()))
    if baseline_rmse is None:
        baseline_rmse = np.sqrt(mean_squared_error(y, np.full_like(y, y.mean())))

    dates = None
    if "Date" in X.index.names or hasattr(X.index, "name") and X.index.name == "Date":
        dates = X.index.values
    elif hasattr(X, "index"):
        try:
            # If X has a Date column preserved
            pass
        except:
            pass

    def objective(trial):
        params = param_fn(trial)
        scores = []

        for tr_idx, val_idx in cv.split(X):
            X_tr = X.iloc[tr_idx]
            X_val = X.iloc[val_idx]
            y_tr = y.iloc[tr_idx]
            y_val = y.iloc[val_idx]

            model = model_class(**params)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            s = composite_score(y_val, y_pred, baseline_mae, baseline_rmse)
            scores.append(s)

        return np.mean(scores)

    return objective


# ═════════════════════════════════════════════════════════════════════════════
# 8. CROSS-VALIDATION RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_purged_cv(X, y, model, cv, label="Model", verbose=True):
    """
    Run PurgedWalkForwardCV and return comprehensive results.
    """
    fold_mae, fold_rmse, fold_r2, fold_composite = [], [], [], []
    all_y_true, all_y_pred = [], []

    if verbose:
        print(f"\n{'Fold':>5} | {'Train size':>10} | {'Val size':>8} | {'MAE':>14} | {'RMSE':>13} | {'R2':>10} | {'Comp.':>8}")
        print("-" * 90)

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X)):
        X_tr = X.iloc[tr_idx]
        X_val = X.iloc[val_idx]
        y_tr = y.iloc[tr_idx]
        y_val = y.iloc[val_idx]

        model_clone = clone(model)
        model_clone.fit(X_tr, y_tr)
        y_pred = model_clone.predict(X_val)

        mae  = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2   = r2_score(y_val, y_pred)
        comp = composite_score(
            y_val, y_pred,
            mean_absolute_error(y_val, np.full_like(y_val, y_val.mean())),
            np.sqrt(mean_squared_error(y_val, np.full_like(y_val, y_val.mean()))),
        )

        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        fold_composite.append(comp)
        all_y_true.extend(y_val.values)
        all_y_pred.extend(y_pred)

        if verbose:
            print(f"  {fold+1:>3}  | {len(X_tr):>10,} | {len(X_val):>8,} | {mae:>14,.0f} | {rmse:>13,.0f} | {r2:>10.4f} | {comp:>8.4f}")

    if verbose:
        print("-" * 90)
        overall_mae  = mean_absolute_error(all_y_true, all_y_pred)
        overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
        overall_r2   = r2_score(all_y_true, all_y_pred)
        overall_comp = composite_score(
            all_y_true, all_y_pred,
            mean_absolute_error(all_y_true, np.full_like(all_y_true, np.mean(all_y_true))),
            np.sqrt(mean_squared_error(all_y_true, np.full_like(all_y_true, np.mean(all_y_true)))),
        )
        print(f"  OVERALL| {'':>10} | {'':>8} | {overall_mae:>14,.0f} | {overall_rmse:>13,.0f} | {overall_r2:>10.4f} | {overall_comp:>8.4f}")

    return {
        "fold_mae": fold_mae,
        "fold_rmse": fold_rmse,
        "fold_r2": fold_r2,
        "fold_composite": fold_composite,
        "mae_mean": np.mean(fold_mae),
        "mae_std": np.std(fold_mae),
        "overall_mae": mean_absolute_error(all_y_true, all_y_pred),
        "overall_rmse": np.sqrt(mean_squared_error(all_y_true, all_y_pred)),
        "overall_r2": r2_score(all_y_true, all_y_pred),
        "overall_composite": composite_score(
            all_y_true, all_y_pred,
            mean_absolute_error(all_y_true, np.full_like(all_y_true, np.mean(all_y_true))),
            np.sqrt(mean_squared_error(all_y_true, np.full_like(all_y_true, np.mean(all_y_true)))),
        ),
        "y_true": all_y_true,
        "y_pred": all_y_pred,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 9. LIGHTGBM PARAM SPACE (for Optuna)
# ═════════════════════════════════════════════════════════════════════════════

LGB_PARAM_SPACE = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

def suggest_lgb_params(trial):
    """Suggest LightGBM hyperparameters for Optuna tuning."""
    return {
        **LGB_PARAM_SPACE,
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 14),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "random_state": 42,
    }


XGB_PARAM_SPACE = {
    "verbosity": 0,
    "n_jobs": -1,
    "tree_method": "hist",
}

def suggest_xgb_params(trial):
    """Suggest XGBoost hyperparameters for Optuna tuning."""
    return {
        **XGB_PARAM_SPACE,
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
    }


def suggest_catboost_params(trial):
    """Suggest CatBoost hyperparameters for Optuna tuning."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "depth": trial.suggest_int("depth", 3, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_seed": 42,
        "verbose": 0,
        "thread_count": -1,
        "allow_writing_files": False,
    }


def suggest_rf_params(trial):
    """Suggest RandomForest/ExtraTrees hyperparameters for Optuna tuning."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "random_state": 42,
        "n_jobs": -1,
    }


def suggest_histgb_params(trial):
    """Suggest HistGradientBoosting hyperparameters for Optuna tuning."""
    return {
        "max_iter": trial.suggest_int("max_iter", 200, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 31, 255),
        "random_state": 42,
    }
