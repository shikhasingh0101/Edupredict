"""Training/serving preprocessing pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(
    numerical_columns: list[str],
    categorical_columns: list[str]
) -> ColumnTransformer:

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numerical_columns),
        ("categorical", categorical_pipe, categorical_columns),
    ])
