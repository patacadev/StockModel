import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from .build_features import calc_RSI, calc_MACD, calc_relative_volume

##### PREPARE DATA #####
def _extract_features(data, feature_columns, target_name="Close"):
    """
    Creates new features to feed the model.
    Parameters:
        data (pd.DataFrame): Original dataframe (['Close', 'Open', 'Low', 'High', 'Volume'])
        feature_columns (List<str>): list of feature columns to be used
        target_name (str): column name to be used as target
    Returns:
        features (pd.DataFrame): pandas DataFrame containing extracted features
        target (np.ndarray): 1d array containing target data
    """
    # Primero extraemos las columnas indicadas en feature_columns que existan en data
    existing_columns = [col for col in feature_columns if col in data.columns]
    features = data.loc[:, existing_columns].copy()
    # Ahora calculamos las columnas extras
    if "Range" in feature_columns:
        features["Range"] = data["High"] - data["Low"]
    if "Gap" in feature_columns:
        features["Gap"] = data["Open"] - data["Close"].shift(1)
    if "RSI" in feature_columns:
        features = calc_RSI(features)
    if "MACD" in feature_columns:
        features = calc_MACD(features)
    if "Relative_Volume" in feature_columns:
        features = calc_relative_volume(features)
    if "ALL_EMA" in feature_columns:
        features['Close_EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
        features['Close_EMA_15'] = data['Close'].ewm(span=10, adjust=False).mean()
        features['Close_EMA_25'] = data['Close'].ewm(span=25, adjust=False).mean()
        features['Close_EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        features['Close_EMA_100'] = data['Close'].ewm(span=100, adjust=False).mean()
        features['Open_EMA_5'] = data['Open'].ewm(span=5, adjust=False).mean()
        features['Open_EMA_10'] = data['Open'].ewm(span=10, adjust=False).mean()
        features['Open_EMA_25'] = data['Open'].ewm(span=25, adjust=False).mean()
        features['Open_EMA_50'] = data['Open'].ewm(span=50, adjust=False).mean()
        features['Open_EMA_100'] = data['Open'].ewm(span=100, adjust=False).mean()

    features = features.dropna()

    # Seleccionamos columna que se usará como TARGET
    target = data[target_name].to_numpy().reshape(-1, 1)  # Convertir a una columna para la escala
    print(f"Feature columns: {list(features.columns)}")
    print(f"Target column: {target_name}")
    return features, target

def _create_dataset(features, target, window_size):
    """
    Creates dataset with moving window to feed the model.
    Parameters:
        features (np.numpy)
        target (np.numpy)
        window_size (int)
    Returns:
        X (np.numpy)
        Y (np.numpy)
    """
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])  # To predict i we get i-window to i-1
        y.append(target[i+window_size])
    return np.array(X), np.array(y)
    
# MAIN METHOD
def prepare_data_train(data, window_size, feature_columns, target_name):
    """
    Preprocess and prepare data for feeding the model
    Parameters:
        data (pd.DataFrame)
        window_size (int)
        train_partition_size (int)
    Returns:
        X_train (np.numpy)
        y_train (np.numpy)
        X_test (np.numpy)
        y_test (np.numpy)
        target_scaler (MinMaxScaler)
        train_dates (pd.DateTimeIndex)
        test_dates (pd.DateTimeIndex)
    """
    # Extraer las columnas X y y
    X_train, y_train = _extract_features(data, feature_columns, target_name=target_name)

    # Guardamos las fechas para la graficación (tras feature extraction ya que se pierden datos con las ventanas)
    train_dates = X_train.index

    ### NORMALIZACIÓN ###
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)

    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)

    X_train, y_train = _create_dataset(X_train_scaled, y_train_scaled, window_size=window_size)
    
    print(f"Feature min {feature_scaler.data_min_}")
    print(f"Feature max {feature_scaler.data_max_}")
    print(f"Target min {target_scaler.data_min_}")
    print(f"Target max {target_scaler.data_max_}")
    print(f"X_train shape: {X_train.shape}")
    print(f"Train_dates: {train_dates[0].strftime('%Y-%m-%d %H:%M:%S')} - {train_dates[-1].strftime('%Y-%m-%d %H:%M:%S')}")
    return X_train, y_train, train_dates

def prepare_data_predict(data, window_size, feature_columns, target_name):
    """
    Preprocess and prepare data for feeding the model
    Parameters:
        data (pd.DataFrame)
        window_size (int)
    Returns:
        X (np.numpy)
        y (np.numpy)
        dates (pd.DateTimeIndex)
    """

    # Extraer las columnas X y y
    X, y = _extract_features(data, feature_columns, target_name=target_name)

    # Guardamos las fechas para la graficación (tras feature extraction ya que se pierden datos con las ventanas)
    dates = X.index

    ### NORMALIZACIÓN ###
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)

    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y)

    X, y = _create_dataset(X_scaled, y_scaled, window_size=window_size)

    print(f"Feature min {feature_scaler.data_min_}")
    print(f"Feature max {feature_scaler.data_max_}")
    print(f"Target min {target_scaler.data_min_}")
    print(f"Target max {target_scaler.data_max_}")
    return X, y, dates, target_scaler