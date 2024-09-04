import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

def grafica_train(history):
    """
    Función para graficar la pérdida de entrenamiento, la pérdida de validación,
    el MAPE de entrenamiento y el MAPE de validación a lo largo de las épocas.

    Parámetros:
    history : objeto History
        Un objeto History de Keras con el historial de entrenamiento. Se asume que
        tiene las claves 'loss', 'val_loss', 'mean_absolute_percentage_error' y
        'val_mean_absolute_percentage_error' en su diccionario `history`.

    Returns:
        None
    """
    # Extract data from the History object
    history_dict = history.history
    
    # Create interactive plots using Plotly
    fig = go.Figure()
    fig2 = go.Figure()
    # Plot training and validation loss
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(history_dict['loss']) + 1),
        y=history_dict['loss'],
        mode='lines+markers',
        name='Training Loss'
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(history_dict['val_loss']) + 1),
        y=history_dict['val_loss'],
        mode='lines+markers',
        name='Validation Loss'
    ))
    
    # Plot training and validation MAPE
    fig2.add_trace(go.Scatter(
        x=np.arange(1, len(history_dict['mape']) + 1),
        y=history_dict['mape'],
        mode='lines+markers',
        name='Training MAPE'
    ))
    fig2.add_trace(go.Scatter(
        x=np.arange(1, len(history_dict['val_mape']) + 1),
        y=history_dict['val_mape'],
        mode='lines+markers',
        name='Validation MAPE'
    ))
    
    # Update layout
    fig.update_layout(
        title='Training and Validation Loss',
        xaxis_title='Epoch',
        yaxis_title='Value',
        legend_title='Metrics',
        template='plotly_dark'
    )
    fig2.update_layout(
        title='Training and Validation MAPE',
        xaxis_title='Epoch',
        yaxis_title='Value',
        legend_title='Metrics',
        template='plotly_dark'
    )
    
    # Show the plot
    fig.show()
    fig2.show()
    
def _grafica_pred_real(predictions, ticker_name=None):
    """
    Args:
        predictions (pd.DataFrame): predictions DF with 'pred' and 'real' keys
    """
    fig = go.Figure()
    # Predicción
    fig.add_trace(go.Scatter(
        x=predictions.index,
        y=predictions["pred"],
        mode='lines+markers',
        name='Prediction'
    ))
    # Valor real
    fig.add_trace(go.Scatter(
        x=predictions.index,
        y=predictions["real"],
        mode='lines+markers',
        name='Real'
    ))
    if ticker_name is not None:
        # Añadir títulos y etiquetas
        fig.update_layout(
            title=ticker_name,
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Metrics',
            template='plotly_dark'
        )
    else:
        # Añadir títulos y etiquetas
        fig.update_layout(
            title="Prediction vs reality",
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Metrics',
            template='plotly_dark'
        )
    fig.show()

def _create_prediction_dataset(test_dates, predictions, real_data):
    """
        Ajustamos predicciones a ventana utilizada (recortamos las primeras x fechas)
        Parameters:
            test_dates (pd.DateTimeIndex): fechas correspondientes al dataset de test (-ventana feature)
            predictions (np.ndarray): predicciones salidas del modelo (len(predictions) = len(test_dates) - window_size)
            real_data (np.ndarray): datos reales
        Returns:
            data (pd.DataFrame): Dataframe with prediction dates, real data and prediction data
    """
    # Ajustamos fechas a predicciones (debido a la ventana)
    fechas = test_dates[-len(predictions):].to_numpy().reshape(-1)

    df_predicciones = pd.DataFrame({'pred': predictions.reshape(-1), 'real': real_data.reshape(-1)}, index=fechas)
    return df_predicciones

def grafica_corr(data):
    """
    Muestra matriz de correlaciones.
    Parameters:
        data (pd.DataFrame)
    Returns:
        None
    """
    # Compute the correlation matrix
    corr = data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix Heatmap', size=15)
    plt.show()

##### MAIN METHOD #####
def model_predict(model, X, y, dates, target_scaler, ticker_name=None, graph=False):
    """
        Evaluamos el modelo
        Parameters:
            model (tensorflow.keras.models)
            X (np.numpy)
            y (np.numpy)
            dates (pd.DateTimeIndex)
            target_scaler (MinMaxScaler)
            ticker_name (str): only for the graph
            graph (boolean): whether to show the graph or not
        Returns:
            df_predicciones (df.DataFrame): Date, pred, real
            loss (float)
            mape (float)
            r2 (float)
    """
    print(f"Target min {target_scaler.data_min_}")
    print(f"Target max {target_scaler.data_max_}")
    # Predict
    predictions = model.predict(X)

    # Desnormalizar las predicciones y las etiquetas verdaderas usando el scaler original
    y_rescaled = target_scaler.inverse_transform(y.reshape(-1, 1))
    predictions_rescaled = target_scaler.inverse_transform(predictions)

    # Calculating MSE, MAPE and R2
    loss = mean_squared_error(y_rescaled, predictions_rescaled)
    mape = mean_absolute_percentage_error(y_rescaled, predictions_rescaled)
    r2 = r2_score(y_rescaled, predictions_rescaled)

    print(f'Loss: {loss:.4f} MAPE: {mape:.4f} R2: {r2:.4f}')
    # Creamos dataset predicciones
    df_predicciones = _create_prediction_dataset(dates, predictions_rescaled, y_rescaled)

    # Graficar los resultados utilizando la función grafica_pred_real
    if graph:
        _grafica_pred_real(df_predicciones, ticker_name)

    return df_predicciones, loss, mape, r2