import numpy as np
import pandas as pd

from datetime import datetime
import json
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from .preprocess import prepare_data_train, prepare_data_predict
from ..visualization.visualization import model_predict, grafica_train

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
        self.train_metadata["end_date"] = str(train_dates[1])
        self.train_metadata["tickers"] = combined_data['Ticker'].unique().tolist()
        return data_dict
    
    ## TRAIN
    def train_multi_model(self, data_dict, patience=10, batch_size=32, epochs=100, graph=False):
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
        model = Sequential([
            Input(shape=(data_dict[first_dict_element]["X_train"].shape[1], data_dict[first_dict_element]["X_train"].shape[2])),
            # 1st LSTM layer
            LSTM(512, return_sequences=True),
            Dropout(0.3),
            LSTM(512, return_sequences=False),
            Dropout(0.3),
            Dense(1)
        ])
        
        model.summary()

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mape'])

        i = 0
        for ticker, content in data_dict.items():
            i += 1
            print(f"--- {i}/{dict_size} Training model for {ticker} ---")
            X_train = content["X_train"]
            y_train = content["y_train"]
            # Definir el Early Stopping basado en la métrica de validación (para cada fit)
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, mode='min')
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, 
                                callbacks=[early_stopping], shuffle=False, verbose=1)
            data_dict[ticker]["history"] = history
            if graph:
                grafica_train(history)
            
            #evaluate_model(model, content["X_test"], content["y_test"])

        return model

    def train(self, combined_data, patience, epochs=100, graph=False):
        # Save metadata
        self.train_metadata["patience"] = patience
        self.train_metadata["max_epochs"] = epochs
        data_dict = self._preprocess(combined_data)
        self.model = self.train_multi_model(data_dict, patience=patience, epochs=epochs, graph=graph)
        if self.export:
            self.save()

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

        mean_metrics = {'loss': mean_loss, 'mape': mean_mape, 'r2': mean_r2}
        if self.export:
            json.dump(metrics_dict, open(self.folder_path+"evaluation.txt", 'w'), indent=4)
            json.dump(mean_metrics, open(self.folder_path+"evaluation_mean.txt", 'w'), indent=4)
        return metrics_dict, mean_metrics
    
    def grid_search_window_size(self, combined_data, test_data, window_size_options, patience=5, epochs=100, graph=False):
        """
        Train but doing a grid search for the window size. Returns the best window size and the grid results.
        """
        best_window_size = None
        best_loss = float('inf')
        best_model = None

        # Diccionario para almacenar resultados del Grid Search
        grid_results = {}

        for window_size in window_size_options:
            print(f"\n--- Running Grid Search for window_size: {window_size} ---")
            self.window_size = window_size

            # Entrenar el modelo
            self.train(combined_data, patience=patience, epochs=epochs, graph=graph)

            mean_loss, mean_mape, mean_r2 = self.evaluate_many(test_data, graph)
            print(f"Evaluation result for window_size {window_size}: Loss = {mean_loss}, MAPE = {mean_mape}, R2 = {mean_r2}")
            
            grid_results[window_size] = (mean_loss, mean_mape, mean_r2)
            
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_window_size = window_size
                best_model = self.model

        print(f"\nBest window_size found: {best_window_size} with loss: {best_loss}")
        self.window_size = best_window_size
        self.model = best_model

        return best_window_size, grid_results

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

    def load(self, name):
        """
        Load model from disk
        """
        print(f"Saving model from {name}...")
        self.model = load_model("trainings/"+name)
        self.model.summary()