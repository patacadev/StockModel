import numpy as np
import pandas as pd


def calc_RSI(df, target_col="Close", window=14):
    """
    Calcula el Índice de Fuerza Relativa (RSI) y lo añade como una nueva columna al DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame con los datos de la acción, debe contener una columna 'Close'.
        window (int): Número de periodos a considerar para el cálculo del RSI (por defecto 14).
    
    Returns:
        pd.DataFrame: DataFrame original con una nueva columna 'RSI' añadida.
    """
    # Calcula los cambios en el precio de cierre entre cada día
    delta = df[target_col].diff(1)
    # Separamos los cambios positivos y negativos
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # Calculamos las medias móviles de las ganancias y pérdidas
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    # Calculamos el Relative Strength (RS)
    rs = avg_gain / avg_loss
    # Calculamos el RSI
    rsi = 100 - (100 / (1 + rs))
    # Añadimos el RSI como una nueva columna en el DataFrame
    df['RSI'] = rsi
    return df

def calc_MACD(df, ema_rapida=12, ema_lenta=26, señal=9):
    """
    Calcula el indicador MACD y añade las columnas correspondientes al DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame que contiene al menos una columna 'Close'.
        ema_rapida (int): Período para la EMA rápida (por defecto 12 días).
        ema_lenta (int): Período para la EMA lenta (por defecto 26 días).
        señal (int): Período para la línea de señal (por defecto 9 días).

    Returns:
        pd.DataFrame: DataFrame con columnas adicionales 'MACD', 'Signal_Line' y 'MACD_Hist'.
    """
    # Calcula las EMAs
    df['EMA_Rapida'] = df['Close'].ewm(span=ema_rapida, adjust=False).mean()
    df['EMA_Lenta'] = df['Close'].ewm(span=ema_lenta, adjust=False).mean()
    # Calcula el MACD
    df['MACD'] = df['EMA_Rapida'] - df['EMA_Lenta']
    # Calcula la Línea de Señal
    df['Signal_Line'] = df['MACD'].ewm(span=señal, adjust=False).mean()
    # Calcula el Histograma MACD
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    # Elimina las columnas intermedias de EMA
    df.drop(['EMA_Rapida', 'EMA_Lenta'], axis=1, inplace=True)
    return df

def calc_relative_volume(df):
    df['Volume_20'] = df['Volume'].rolling(window=20).mean()
    df['Relative_Volume'] =  df['Volume'] / df['Volume_20'] * 100
    # Borramos columna de volumen medio
    df.drop(['Volume_20'], axis=1, inplace=True)
    return df