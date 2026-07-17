import pandas as pd
import numpy as np

from typing import Dict, Any, Tuple, List, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def split_data(raw_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split dataframe into train and validation sets.

    Args:
        raw_df (pd.DataFrame): Raw dataframe.

    Returns:
        Dict[str, pd.DataFrame]: Train and validation dataframes.
    """

    train_df, val_df = train_test_split(
        raw_df,
        test_size=0.2,
        random_state=42,
        stratify=raw_df["Exited"],
    )

    return {
        "train": train_df,
        "val": val_df,
    }


def create_inputs_targets(
    df_dict: Dict[str, pd.DataFrame],
    input_cols: List[str],
    target_col: str,
) -> Dict[str, Any]:
    """
    Create train and validation inputs and targets.
    """

    data = {}

    for split in df_dict:
        data[f"{split}_inputs"] = df_dict[split][input_cols].copy()
        data[f"{split}_targets"] = df_dict[split][target_col].copy()

    return data


def scale_numeric_features(
    data: Dict[str, Any],
    numeric_cols: List[str],
) -> MinMaxScaler:
    """
    Scale numeric features using MinMaxScaler.
    """

    scaler = MinMaxScaler().fit(data["train_inputs"][numeric_cols])

    for split in ["train", "val"]:
        data[f"{split}_inputs"][numeric_cols] = scaler.transform(
            data[f"{split}_inputs"][numeric_cols]
        )

    return scaler


def encode_categorical_features(
    data: Dict[str, Any],
    categorical_cols: List[str],
) -> OneHotEncoder:
    """
    One-hot encode categorical features.
    """

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    ).fit(data["train_inputs"][categorical_cols])

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    for split in ["train", "val"]:

        encoded = encoder.transform(
            data[f"{split}_inputs"][categorical_cols]
        )

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoded_cols,
            index=data[f"{split}_inputs"].index,
        )

        data[f"{split}_inputs"] = pd.concat(
            [data[f"{split}_inputs"], encoded_df],
            axis=1,
        )

        data[f"{split}_inputs"].drop(
            columns=categorical_cols,
            inplace=True,
        )

    data["encoded_cols"] = encoded_cols

    return encoder


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = True,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    List[str],
    Optional[MinMaxScaler],
    OneHotEncoder,
]:
    """
    Complete preprocessing of training data.

    Returns:
        X_train, train_targets, X_val, val_targets,
        input_cols, scaler, encoder
    """

    input_cols = list(raw_df.columns[:-1])
    target_col = "Exited"

    split_dfs = split_data(raw_df)

    data = create_inputs_targets(
        split_dfs,
        input_cols,
        target_col,
    )

    numeric_cols = (
        data["train_inputs"]
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    categorical_cols = (
        data["train_inputs"]
        .select_dtypes(include="object")
        .columns
        .tolist()
    )

    scaler = None

    if scaler_numeric:
        scaler = scale_numeric_features(
            data,
            numeric_cols,
        )

    encoder = encode_categorical_features(
        data,
        categorical_cols,
    )

    feature_cols = numeric_cols + data["encoded_cols"]

    X_train = data["train_inputs"][feature_cols]
    X_val = data["val_inputs"][feature_cols]

    return {
        'train_X': X_train,
        'train_y': data['train_targets'],
        'val_X': X_val,
        'val_y': data['val_targets'],
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder
    }


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: list,
    scaler: Optional[MinMaxScaler],
    encoder: OneHotEncoder,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess new data using fitted scaler and encoder.

    Args:
        new_df (pd.DataFrame): New data.
        input_cols (list): List of feature columns used during training.
        scaler (MinMaxScaler | None): Fitted scaler.
        encoder (OneHotEncoder): Fitted encoder.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            Processed dataframe and ids.
    """

    # Сохраняем id для submission
    ids = new_df["id"].copy()

    # Берем только те признаки, которые были при обучении
    X = new_df[input_cols].copy()

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    if scaler is not None:
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    encoded = encoder.transform(X[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_cols,
        index=X.index,
    )

    X = X.drop(columns=categorical_cols)

    X = pd.concat([X, encoded_df], axis=1)

    return X, ids