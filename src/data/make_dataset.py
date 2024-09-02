import numpy as np
import pandas as pd

import yfinance as yf
from datetime import datetime, timedelta

def save_list(stock_list, filename):
    with open("symbols/"+filename+".txt", 'w') as f:
        for item in stock_list:
            f.write(f"{item}\n")
def load_list(filename):
    with open("symbols/"+filename+".txt", 'r') as f:
        loaded_list = [line.strip() for line in f]
    return loaded_list

##### DESCARGA DATOS #####
def get_stock_symbols(name=None):
    # Obtener la lista de componentes del S&P 500 desde Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    lista_sp500 = table[0]["Symbol"].to_list()

    # Obtener la lista de componentes del NASDAQ-100 desde Wikipedia
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    table = pd.read_html(url)
    lista_nasdaq100 = table[4]["Ticker"].to_list()

    # Algunos simbolos tienen . en lugar de -
    for i, stock in enumerate(lista_nasdaq100):
        if "." in stock:
            lista_nasdaq100[i] = stock.replace(".","-")
    for i, stock in enumerate(lista_sp500):
        if "." in stock:
            lista_sp500[i] = stock.replace(".","-")

    lista_completa = lista_sp500 + lista_nasdaq100
    download_stock = set(lista_completa) # Eliminamos duplicados

    # Añadimos SP500, NASDAQ100, DOW Jones y BTC-USD
    extra_stock = ['^GSPC', '^IXIC' ,'^DJI', 'BTC-USD']
    download_stock.update(extra_stock)
    
    if name == "NASDAQ":
        print(f"{len(lista_nasdaq100)} símbolos de acciones")
        return list(set(lista_nasdaq100))
    elif name == "SP500":
        print(f"{len(lista_sp500)} símbolos de acciones")
        return list(set(lista_sp500))
    else:
        print(f"{len(download_stock)} símbolos de acciones")
        return list(download_stock)

def download_data(stock_list, folder_path="stock_data"):
    """
    Downloads stock data using yfinance API (1d, 1h and 5min aggregates)
    Parameters:
        stock_list (Set{}): List containing stock symbols to download
        folder_path (list): folder path to save the data in (default: stock_data)
    Returns:
        None
    """
    end_date=datetime.now()
    print(f"Downloading {stock_list} data...")
    stock_list_len = len(stock_list)
    for i, ticker in enumerate(stock_list):
        print(f"--- {i}/{stock_list_len} ---")
        # 1d data
        try:
            data = yf.download(ticker, start='2000-01-01', end=end_date, interval='1d', progress=False)
            data.to_csv(folder_path+"/"+ticker+'-1d.csv')
            print(f"{ticker} 1 day OK")
        except Exception as e:
            print(f"{ticker} 1 day ERROR")
            print(e)
        # 1h data
        try:
            start_date_1h = (end_date - timedelta(days=730)) # La API solo nos permite datos horarios de los ultimos 730 días
            data = yf.download(ticker, start=start_date_1h, end=end_date, interval='1h', progress=False)
            data.to_csv(folder_path+"/"+ticker+'-1h.csv')
            print(f"{ticker} 1 hour OK")
        except Exception as e:
            print(f"{ticker} 1 hour ERROR")
            print(e)
        # 5m data
        try:
            start_date_5m = (end_date - timedelta(days=60)) # La API solo nos permite datos horarios de los ultimos 60 días
            data = yf.download(ticker, start=start_date_5m, end=end_date, interval='5m', progress=False)
            data.to_csv(folder_path+"/"+ticker+'-5m.csv')
            print(f"{ticker} 5 min OK")
        except Exception as e:
            print(f"{ticker} 5 min ERROR")
            print(e)
    print("Download DONE")

def _get_data(ticker_name, interval="1d", start_date=None, end_date=None):
    """
    Function to get data in an online manner (directly from yfinance instead of downloading and loading)
    Parameters:
        ticker_name (str): Stock symbol to download
        interval (str): Aggregation interval to use (1d, 1h or 5m)
        start_date (str): Starting date of the data retrieval. Format YYYY-MM-DD
        end_date (str): Ending date of the data retrieval. Format YYYY-MM-DD
    Returns:
        ticker_historical (pd.DataFrame)
    """
    if start_date is None:
        start_date = "2000-01-01"
    if end_date is None:
        end_date = "2024-01-01"

    ticker = yf.Ticker(ticker_name)
    today = datetime.now()
    ticker_historical = None
    if interval == "1d":
        ticker_historical = ticker.history(start=start_date, end=end_date, interval="1d")
    elif interval == "1h":
        max_start = datetime.now() - timedelta(days=730)
        ticker_historical = ticker.history(start=max_start, end=today, interval="1h")
    elif interval == "5m":
        max_start = datetime.now() - timedelta(days=60)
        ticker_historical = ticker.history(start=max_start, end=today, interval="5m")
    else:
        print("ERROR: interval not supported")
    return ticker_historical

def get_stock_data(ticker_names, interval="1d", start_date=None, end_date=None):
    """
    Function to get data in an online manner (directly from yfinance instead of downloading and loading) 
    for multiple tickers
    Parameters:
        ticker_names (List <str>): Stock symbols to get
        interval (str): Aggregation interval to use (1d, 1h or 5m)
    Returns:
        combined_data (pd.DataFrame): dataframe with new column 'Ticker'
    """
    # Diccionario para almacenar los DataFrames de cada ticker
    data_frames = {}
    
    # Descargar datos de cada ticker y almacenarlos en el diccionario
    for ticker in ticker_names:
        data = _get_data(ticker, interval, start_date, end_date)
        # Ensure that dates match the required (will match if interval 1d)
        if start_date is not None and end_date is not None:
            data = data[(data.index >= start_date) & (data.index <= end_date)]
        elif start_date is not None and end_date is None:
            data = data[(data.index >= start_date)]
        elif start_date is None and end_date is not None:
            data = data[(data.index <= end_date)]
        data['Ticker'] = ticker  # Añadir una columna para identificar el ticker
        data_frames[ticker] = data
    
    # Concatenar todos los DataFrames en uno solo
    combined_data = pd.concat(data_frames.values())
    
    return combined_data

