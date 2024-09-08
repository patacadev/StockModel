import numpy as np
import pandas as pd

from datetime import datetime
import json, os, time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from .preprocess import prepare_data_train, prepare_data_predict
from ..visualization.visualization import model_predict, grafica_train

def train_metrics_summary(history_dict):
    best_losses = []
    best_val_losses = []
    best_mapes = []
    best_val_mapes = []

    best_metrics_per_ticker = {}

    # Recopilar las mejores métricas por ticker y métricas globales
    for ticker, metrics in history_dict.items():
        best_loss = min(metrics['loss'])
        best_val_loss = min(metrics['val_loss'])
        best_mape = min(metrics['mape'])
        best_val_mape = min(metrics['val_mape'])

        best_losses.append(best_loss)
        best_val_losses.append(best_val_loss)
        best_mapes.append(best_mape)
        best_val_mapes.append(best_val_mape)

        best_metrics_per_ticker[ticker] = {
            "loss": best_loss,
            "val_loss": best_val_loss,
            "mape": best_mape,
            "val_mape": best_val_mape
        }

    # Cálculo de la media y mediana
    mean_dict = {
        "loss": np.mean(best_losses),
        "val_loss": np.mean(best_val_losses),
        "mape": np.mean(best_mapes),
        "val_mape": np.mean(best_val_mapes)
    }

    median_dict = {
        "loss": np.median(best_losses),
        "val_loss": np.median(best_val_losses),
        "mape": np.median(best_mapes),
        "val_mape": np.median(best_val_mapes)
    }

    return {
        "mean": mean_dict,
        "median": median_dict,
        "tickers": best_metrics_per_ticker
    }

class StockModel:
    def __init__(self, window_size, feature_columns, target_name, export=False):
        print(f"Initializing model:\n - Window size: {window_size}\n - Features: {str(feature_columns)}\n - Target: {target_name}")
        self.window_size = window_size
        self.feature_columns = feature_columns
        self.target_name = target_name
        self.model = None
        self.export = export
        if export:
            self.train_metadata = {"window_size": window_size, "target": target_name, "feature_columns": feature_columns}
            # Creamos path para guardar modelo
            fecha_hora_actual = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            nombre_directorio = f"{fecha_hora_actual}/"
            path = "trainings/"+nombre_directorio
            self.folder_path = path
        
    def _preprocess(self, combined_data):
        """
        Method to preprocess data for feeding the model.

        Parameters:
            data (pd_.DataFrame): dataframe indexed by date containing data and an extra column named "Ticker" indicating the stock name
        Returns:
            
        """
        # Diccionario para almacenar datos de cada empresa
        data_dict = {}
        
        # Separar y normalizar los datos por cada ticker
        for ticker in combined_data['Ticker'].unique():
            try:
                print(f"--- Preparing {ticker} data using {self.window_size} window---")
                data = combined_data[combined_data['Ticker'] == ticker]
                # Prepare data (feature extraction, dataset creation)
                X_train, y_train, train_dates = prepare_data_train(data, window_size=self.window_size,
                    feature_columns=self.feature_columns, target_name=self.target_name)
                # Save data in dict
                data_dict[ticker] = {
                    "X_train": X_train,
                    "y_train": y_train
                }
            except Exception as error:
                print(f"Error preparing {ticker} data. {error}")
        # Actualizamos metadata
        self.train_metadata["start_date"] = str(train_dates[0])
        self.train_metadata["end_date"] = str(train_dates[-1])
        return data_dict
    
    ## TRAIN
    def train_multi_model(self, data_dict, patience=10, batch_size=32, epochs=100, graph=False, layers=1, units_per_layer=128):
        """
        Train LSTM model and return itself and its history
        Parameters:
            X_train (np.numpy)
            y_train (np.numpy)
        Returns:
            model (tensorflow.keras.models)
            history (keras.callbacks.History)
        """
        first_dict_element = next(iter(data_dict))
        dict_size = len(data_dict)
        # LSTM model
        model = Sequential()
        model.add(Input(shape=(data_dict[first_dict_element]["X_train"].shape[1], data_dict[first_dict_element]["X_train"].shape[2])))
        for i in range(layers-1):
            model.add(LSTM(units_per_layer, return_sequences=True))
            model.add(Dropout(0.3))
        # Last LSTM layer
        model.add(LSTM(units_per_layer, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        
        model.summary()

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mape'])

        i = 0
        history_dict = {}
        for ticker, content in data_dict.items():
            i += 1
            print(f"--- {i}/{dict_size} Training model for {ticker} ---")
            X_train = content["X_train"]
            y_train = content["y_train"]
            # Definir el Early Stopping basado en la métrica de validación (para cada fit)
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, mode='min')
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, 
                                callbacks=[early_stopping], shuffle=False, verbose=1)
            # Save training metrics
            history_dict[ticker] = {"loss": history.history['loss'], "val_loss": history.history['val_loss'],
                                    "mape": history.history['mape'], "val_mape": history.history['val_mape']}
            if graph:
                grafica_train(history)

        return model, history_dict

    def train(self, combined_data, patience, epochs=100, graph=False, layers=1, units_per_layer=128):
        if layers > 0:
            # Save metadata
            self.train_metadata["patience"] = patience
            self.train_metadata["max_epochs"] = epochs
            self.train_metadata["layers"] = layers
            self.train_metadata["units_per_layer"] = units_per_layer
            self.train_metadata["number_tickers"] = len(combined_data['Ticker'].unique())
            data_dict = self._preprocess(combined_data)
            # Entrenamos y guardamos tiempo transcurrido
            inicio = time.time()
            self.model, history_dict = self.train_multi_model(data_dict, patience=patience, epochs=epochs, graph=graph, layers=layers, units_per_layer=units_per_layer)
            fin = time.time()
            tiempo = fin-inicio
            horas = int(tiempo // 3600)
            minutos = int((tiempo % 3600) // 60)
            segundos = int(tiempo % 60)
            if horas > 0:
                tiempo = f"{horas:02d}h {minutos:02d}min {segundos:02d}s"
            else:
                tiempo = f"{minutos:02d}min {segundos:02d}s"
            self.train_metadata["train_time"] = tiempo
            # Select best epoch of every ticker and get the mean
            summary = train_metrics_summary(history_dict)
            self.train_metadata["train_summary"] = summary
            self.train_metadata["history"] = history_dict
            if self.export:
                self.save()
            return history_dict
        else:
            raise Exception("Model has to have at least one LSTM layer")

    def predict(self, data, ticker_name=None, graph=False):
        """
        Prepares data for prediction and predicts
        Parameters:
            X
            y
            dates
            target_scaler
        """
        if self.model:
            X, y, dates, target_scaler = prepare_data_predict(data, window_size=self.window_size, feature_columns=self.feature_columns, 
                                                            target_name=self.target_name)
            print(f"Predict with X shape: {X.shape}")
            return model_predict(self.model, X, y, dates, target_scaler, ticker_name, graph=graph)
        else:
            raise Exception("Model was not found")

    def evaluate(self, X_test, y_test):
        """
        Evaluate model with test data
        Parameters:
            X_test (np.ndarray)
            y_test (np.ndarray)
        Returns:
            loss
            mape
        """
        # Evaluar el modelo
        print(f"Evaluating model with X_test shape: {X_test.shape}")

        loss, mape = self.model.evaluate(X_test, y_test)
        print(f'Loss: {loss:.4f} MAPE: {mape:.4f}')

        return loss, mape
    
    def evaluate_many(self, test_data, graph):
        """
        Evaluate model with test data
        Parameters:
            test_data (pd.DataFrame): dataframe containing "Ticker" column
        Returns:
            metrics_dict (dict): dict containing metrics for every Ticker
            mean_metrics (dict): dict containing 
        """
        metrics_dict = {}

        for ticker in test_data['Ticker'].unique():
            data = test_data[test_data['Ticker'] == ticker]
            _, loss, mape, r2 = self.predict(data, ticker, graph=graph)
            # Almacenamos metricas en dict
            metrics_dict[ticker] = {'loss': loss, 'mape': mape, 'r2': r2}
        # Calcular las medias de cada métrica usando los valores en metrics_dict
        mean_loss = np.mean([metrics['loss'] for metrics in metrics_dict.values()])
        mean_mape = np.mean([metrics['mape'] for metrics in metrics_dict.values()])
        mean_r2 = np.mean([metrics['r2'] for metrics in metrics_dict.values()])

        median_loss = np.median([metrics['loss'] for metrics in metrics_dict.values()])
        median_mape = np.median([metrics['mape'] for metrics in metrics_dict.values()])
        median_r2 = np.median([metrics['r2'] for metrics in metrics_dict.values()])


        mean_metrics = {'loss': mean_loss, 'mape': mean_mape, 'r2': mean_r2}
        median_metrics = {'loss': median_loss, 'mape': median_mape, 'r2': median_r2}
        summary = {"mean": mean_metrics, "median": median_metrics, "number_tickers": len(metrics_dict)}
        if self.export:
            json.dump(metrics_dict, open(self.folder_path+"evaluation.txt", 'w'), indent=4)
            json.dump(summary, open(self.folder_path+"evaluation_summary.txt", 'w'), indent=4)
        return metrics_dict, summary

    def save(self):
        """
        Saves model to disk
        """
        if self.model is not None:
            os.makedirs(self.folder_path, exist_ok=True)
            print(f"Saving model in {self.folder_path}...")
            self.model.save(self.folder_path+"model.keras")
            # Save metadata
            json.dump(self.train_metadata, open(self.folder_path+"train_metadata.txt", 'w'), indent=4)
        else:
            print("Error saving model: Model was not trained")

    def load(self, path):
        """
        Load model from disk
        """
        print(f"Saving model from {path}...")
        # Cargamos también los parámetros del optimizador para continuar entrenando si se desea
        self.model = load_model(path+"/model.keras")
        self.model.summary()