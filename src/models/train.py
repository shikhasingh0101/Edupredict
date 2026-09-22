"""Reproducible DVC + MLflow training stage for EduPredict.

Model selection uses only the train/validation splits.
The final test split is never used for model selection.

Each candidate model gets its own MLflow run. The model with the
lowest validation RMSE is selected as the champion, refit on the
complete development set, saved locally, logged to MLflow, registered
in the MLflow Model Registry, and assigned the 'champion' alias.
"""

from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/processed/student_performance_engineered.csv"
)

SELECTION_PATH = Path(
    "outputs/metrics/selected_features.json"
)

SUMMARY_PATH = Path(
    "outputs/metrics/model_experiment_summary.json"
)

RESULTS_PATH = Path(
    "outputs/metrics/model_experiment_results.csv"
)

MODEL_PATH = Path(
    "models/edupredict_final_pipeline.joblib"
)

TARGET = "Exam_Score"

RANDOM_STATE = 42

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

MLFLOW_EXPERIMENT = "EduPredict"

REGISTERED_MODEL_NAME = "EduPredict_Exam_Score_Model"

MODEL_ALIAS = "champion"


# ============================================================
# PREPROCESSOR
# ============================================================

def make_preprocessor(X):
    """Create preprocessing pipeline for numerical/categorical data."""

    numerical_columns = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    return ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_columns,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ),
    ])


# ============================================================
# MODEL FACTORY
# ============================================================

def make_model(name):
    """Return the requested regression model."""

    models = {
        "LinearRegression": LinearRegression(),

        "Ridge": Ridge(
            alpha=1.0,
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        ),
    }

    if name not in models:
        raise ValueError(
            f"Unknown model: {name}"
        )

    return models[name]


# ============================================================
# COMPLETE MODEL PIPELINE
# ============================================================

def build_pipeline(X, model_name):
    """Build preprocessing + model pipeline."""

    return Pipeline([
        (
            "preprocessor",
            make_preprocessor(X),
        ),
        (
            "model",
            make_model(model_name),
        ),
    ])


# ============================================================
# MLFLOW PARAMETER LOGGING
# ============================================================

def log_model_parameters(model_name, model):
    """Log safe scalar model parameters to MLflow."""

    # Model name is logged explicitly.
    mlflow.log_param(
        "model",
        model_name,
    )

    # Get model hyperparameters.
    params = model.get_params()

    safe_params = {}

    for key, value in params.items():

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            safe_params[key] = value

        elif value is None:
            safe_params[key] = "None"

    if safe_params:
        mlflow.log_params(safe_params)


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("EduPredict — DVC + MLflow Training")
    print("=" * 70)

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    print(
        f"MLflow tracking URI: {MLFLOW_TRACKING_URI}"
    )

    print(
        f"MLflow experiment:   {MLFLOW_EXPERIMENT}"
    )

    # --------------------------------------------------------
    # Load engineered dataset
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Load selected features
    # --------------------------------------------------------

    if not SELECTION_PATH.exists():
        raise FileNotFoundError(
            f"Feature selection file not found: "
            f"{SELECTION_PATH}"
        )

    with open(
        SELECTION_PATH,
        "r",
    ) as f:
        selection = json.load(f)

    selected_features = selection.get(
        "selected_features",
        [],
    )

    selected = [
        column
        for column in selected_features
        if column in df.columns
        and column != TARGET
    ]

    # Safety fallback.
    if not selected:
        selected = [
            column
            for column in df.columns
            if column != TARGET
        ]

    X = df[selected].copy()

    y = df[TARGET].copy()

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Selected features: {len(selected)}"
    )

    # --------------------------------------------------------
    # Development / test split
    # --------------------------------------------------------
    #
    # Test set is completely untouched during model selection.
    #
    # 80% development
    # 20% test
    # --------------------------------------------------------

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Training / validation split
    # --------------------------------------------------------
    #
    # 80% of development -> training
    # 20% of development -> validation
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    print(
        f"Development rows: {len(X_dev)}"
    )

    print(
        f"Training rows:    {len(X_train)}"
    )

    print(
        f"Validation rows:  {len(X_val)}"
    )

    print(
        f"Test rows:        {len(X_test)}"
    )

    # --------------------------------------------------------
    # Candidate models
    # --------------------------------------------------------

    model_names = [
        "LinearRegression",
        "Ridge",
        "RandomForest",
        "ExtraTrees",
        "GradientBoosting",
        "HistGradientBoosting",
    ]

    results = []

    # ========================================================
    # MODEL EXPERIMENT LOOP
    # ========================================================

    for model_name in model_names:

        print()
        print("-" * 70)
        print(
            f"Training: {model_name}"
        )
        print("-" * 70)

        pipeline = build_pipeline(
            X_train,
            model_name,
        )

        # ----------------------------------------------------
        # Start MLflow run
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name=model_name
        ) as run:

            # ------------------------------------------------
            # Tags
            # ------------------------------------------------

            mlflow.set_tags({
                "project": "EduPredict",
                "task": "exam_score_regression",
                "stage": "model_selection",
                "dataset": "StudentPerformanceFactors",
                "selection_metric": "validation_rmse",
            })

            # ------------------------------------------------
            # General experiment parameters
            # ------------------------------------------------

            mlflow.log_params({
                "selected_feature_count": len(selected),
                "development_rows": len(X_dev),
                "training_rows": len(X_train),
                "validation_rows": len(X_val),
                "test_rows": len(X_test),
                "target": TARGET,
                "random_state": RANDOM_STATE,
            })

            # ------------------------------------------------
            # Create model
            # ------------------------------------------------

            model = make_model(
                model_name
            )

            # Log model-specific parameters.
            #
            # random_state is intentionally NOT logged
            # separately here because it is already included
            # in the general experiment parameters above.
            #
            # This prevents MLflow's "Changing param values"
            # error for models such as Ridge.
            # ------------------------------------------------

            model_params = model.get_params()

            safe_model_params = {}

            for key, value in model_params.items():

                # Avoid duplicate random_state parameter.
                if key == "random_state":
                    continue

                if isinstance(
                    value,
                    (str, int, float, bool),
                ):
                    safe_model_params[key] = value

                elif value is None:
                    safe_model_params[key] = "None"

            if safe_model_params:
                mlflow.log_params(
                    safe_model_params
                )

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            pipeline.fit(
                X_train,
                y_train,
            )

            # ------------------------------------------------
            # Validation prediction
            # ------------------------------------------------

            predictions = pipeline.predict(
                X_val
            )

            # ------------------------------------------------
            # Validation metrics
            # ------------------------------------------------

            validation_rmse = float(
                mean_squared_error(
                    y_val,
                    predictions,
                ) ** 0.5
            )

            validation_mae = float(
                mean_absolute_error(
                    y_val,
                    predictions,
                )
            )

            validation_r2 = float(
                r2_score(
                    y_val,
                    predictions,
                )
            )

            # ------------------------------------------------
            # Log validation metrics
            # ------------------------------------------------

            mlflow.log_metrics({
                "validation_rmse": validation_rmse,
                "validation_mae": validation_mae,
                "validation_r2": validation_r2,
            })

            # ------------------------------------------------
            # Log candidate model
            # ------------------------------------------------
            #
            # MLflow 3.x + skops performs security validation.
            #
            # These are the specific sklearn/numpy types used
            # by the trusted models in this project.
            # ------------------------------------------------

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                    (
                        "sklearn.ensemble."
                        "_hist_gradient_boosting."
                        "predictor.TreePredictor"
                    ),
                ],
            )

            # ------------------------------------------------
            # Store experiment result
            # ------------------------------------------------

            results.append({
                "model": model_name,
                "validation_rmse": validation_rmse,
                "validation_mae": validation_mae,
                "validation_r2": validation_r2,
                "run_id": run.info.run_id,
            })

            print(
                f"RMSE: {validation_rmse:.6f}"
            )

            print(
                f"MAE : {validation_mae:.6f}"
            )

            print(
                f"R²  : {validation_r2:.6f}"
            )

            print(
                f"Run : {run.info.run_id}"
            )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "validation_rmse"
        )
        .reset_index(drop=True)
    )

    # Lowest validation RMSE = champion.
    champion_name = results_df.iloc[0]["model"]

    champion_run_id = results_df.iloc[0]["run_id"]

    champion_rmse = float(
        results_df.iloc[0]["validation_rmse"]
    )

    champion_mae = float(
        results_df.iloc[0]["validation_mae"]
    )

    champion_r2 = float(
        results_df.iloc[0]["validation_r2"]
    )

    print()
    print("=" * 70)
    print("MLflow Experiment Comparison")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Champion: {champion_name}"
    )

    print(
        f"Champion run: {champion_run_id}"
    )

    # ========================================================
    # REFIT CHAMPION
    # ========================================================
    #
    # IMPORTANT:
    # The test set is STILL untouched.
    #
    # We now train the selected champion on all development
    # data = train + validation.
    # ========================================================

    print()
    print(
        "Refitting champion on complete development set..."
    )

    final_pipeline = build_pipeline(
        X_dev,
        champion_name,
    )

    final_pipeline.fit(
        X_dev,
        y_dev,
    )

    # --------------------------------------------------------
    # Save production joblib artifact
    # --------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_pipeline,
        MODEL_PATH,
    )

    print(
        f"Saved: {MODEL_PATH}"
    )

    # ========================================================
    # REGISTER CHAMPION IN MLFLOW MODEL REGISTRY
    # ========================================================

    print()
    print(
        "Registering champion model in MLflow..."
    )

    champion_model_uri = (
        f"runs:/{champion_run_id}/model"
    )

    registered_model = mlflow.register_model(
        model_uri=champion_model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    registered_version = (
        registered_model.version
    )

    # --------------------------------------------------------
    # Assign champion alias
    # --------------------------------------------------------

    client = mlflow.MlflowClient()

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        MODEL_ALIAS,
        registered_version,
    )

    print(
        f"Registered model: "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"Registered version: "
        f"{registered_version}"
    )

    print(
        f"Alias: {MODEL_ALIAS}"
    )

    # ========================================================
    # SAVE EXPERIMENT RESULTS
    # ========================================================

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save experiment summary
    # --------------------------------------------------------

    summary = {
        "champion_model": champion_name,
        "champion_run_id": champion_run_id,
        "champion_validation_rmse": champion_rmse,
        "champion_validation_mae": champion_mae,
        "champion_validation_r2": champion_r2,
        "registered_model": REGISTERED_MODEL_NAME,
        "registered_version": int(
            registered_version
        ),
        "registered_alias": MODEL_ALIAS,
        "selected_features": selected,
        "development_rows": int(
            len(y_dev)
        ),
        "training_rows": int(
            len(y_train)
        ),
        "validation_rows": int(
            len(y_val)
        ),
        "test_rows": int(
            len(y_test)
        ),
        "note": (
            "The test set was not used for model selection. "
            "The champion was selected using validation RMSE "
            "and then refit on the complete development set."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Training completed successfully"
    )
    print("=" * 70)

    print(
        f"Champion: {champion_name}"
    )

    print(
        f"Validation RMSE: "
        f"{champion_rmse:.6f}"
    )

    print(
        f"Validation MAE:  "
        f"{champion_mae:.6f}"
    )

    print(
        f"Validation R²:   "
        f"{champion_r2:.6f}"
    )

    print(
        f"Model artifact:  "
        f"{MODEL_PATH}"
    )

    print(
        f"MLflow model:    "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"MLflow version:  "
        f"{registered_version}"
    )

    print(
        f"MLflow alias:    "
        f"{MODEL_ALIAS}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
