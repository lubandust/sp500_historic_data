import yfinance as yf
from fredapi import Fred
import pandas as pd
import os
from datetime import datetime
import numpy as np
import logging

# Configure logging to display the time, log level, and message
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# FRED API key needed to access macroeconomic data from FRED
FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)


# Function to fetch historical stock and dividends data from Yahoo Finance
def fetch_yahoo_finance_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        historical_data = stock.history(start=start_date, end=end_date)
        dividends = stock.dividends

        # Adding Ticker column for identification
        historical_data["Ticker"] = ticker
        dividends = dividends.to_frame().reset_index()
        dividends["Ticker"] = ticker

        # Renaming columns for consistency
        dividends.columns = ["Date", "Dividends", "Ticker"]

        return historical_data.reset_index(), dividends
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame(), pd.DataFrame()


# Function to fetch earnings calendar data using yfinance package
def fetch_yahoo_earnings_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        earnings_calendar = stock.get_earnings_dates(limit=8)

        # Restructuring and adding Ticker column
        earnings_calendar = earnings_calendar.rename_axis("Date").reset_index()
        earnings_calendar["Ticker"] = ticker

        return earnings_calendar
    except Exception as e:
        logging.error(f"Error fetching earnings data for {ticker}: {e}")
        return pd.DataFrame()


# Function to fetch macroeconomic data series from FRED
def fetch_fred_data(series_id, start_date, end_date):
    try:
        data = fred.get_series(series_id, start_date, end_date)
        return data
    except Exception as e:
        logging.error(f"Error fetching FRED data for series_id {series_id}: {e}")
        return pd.Series()


# Function to fetch and merge historical stock, dividends, and earnings data for a list of tickers
def fetch_and_merge_data(tickers, start_date, end_date):
    all_stock_data = []
    all_dividends_data = []
    all_earnings_data = []

    # Loop through each ticker to fetch and store its data
    for ticker in tickers:
        stock_data, dividends_data = fetch_yahoo_finance_data(ticker, start_date, end_date)

        if not stock_data.empty:
            all_stock_data.append(stock_data)
        if not dividends_data.empty:
            all_dividends_data.append(dividends_data)

        earnings_data = fetch_yahoo_earnings_data(ticker)
        if not earnings_data.empty:
            all_earnings_data.append(earnings_data)

    # Concatenating and sorting data from all tickers
    merged_stock_data = pd.concat(all_stock_data).sort_values(by=["Date", "Ticker"]).reset_index(drop=True)
    merged_dividends_data = pd.concat(all_dividends_data).sort_values(by=["Date", "Ticker"]).reset_index(drop=True)
    merged_earnings_data = pd.concat(all_earnings_data).sort_values(by=["Date", "Ticker"]).reset_index(drop=True)

    return merged_stock_data, merged_dividends_data, merged_earnings_data


# Orchestrate fetching and saving of backtest data.
# Retrieves stock tickers, fetches financial and macroeconomic data, and saves data to CSV files.
def get_backtest_data():
    stock_tickers = get_tickers_list()
    fred_series_id = "GS10"  # 10-Year Treasury Constant Maturity Rate
    start_date = "2023-08-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    # Fetch and merge stock data
    stock_data, dividends_data, earnings_data = fetch_and_merge_data(stock_tickers, start_date, end_date)

    # Fetch macroeconomic data
    macro_data = fetch_fred_data(fred_series_id, start_date, end_date)

    # Create directory to save data
    os.makedirs("backtest_data", exist_ok=True)

    # Save each type of data to its respective CSV file
    if not stock_data.empty:
        stock_data.to_csv("backtest_data/stock_data.csv", index=False)
    if not dividends_data.empty:
        dividends_data.to_csv("backtest_data/dividends_data.csv", index=False)
    if not earnings_data.empty:
        earnings_data.to_csv("backtest_data/earnings_data.csv", index=False)
    if not macro_data.empty:
        macro_data.to_csv("backtest_data/macro_data.csv", header=True, index=True)


# Function to calculate the average annualized growth rate of quarterly earnings per share (EPS)
def calculate_growth_rate(earnings_df):
    try:
        # Ensure EPS is of float type for calculation
        earnings_df["Reported EPS"] = earnings_df["Reported EPS"].astype(float)

        # Calculate quarterly growth rate
        earnings_df["Quarterly Growth"] = earnings_df["Reported EPS"].pct_change()
        growth_rates = earnings_df["Quarterly Growth"].dropna()

        if growth_rates.empty:
            return None

        # Calculate average quarterly and then annualize the growth rate
        average_quarterly_growth_rate = np.mean(growth_rates)
        annualized_growth_rate = average_quarterly_growth_rate * 4

        return annualized_growth_rate
    except Exception as e:
        logging.error(f"Error calculating growth rate: {e}")
        return None


# Function to calculate intrinsic value of a stock using a simplified growth model
def calculate_intrinsic_value(eps, growth_rate):
    try:
        return eps * (7 + 2 * growth_rate)
    except Exception as e:
        logging.error(f"Error calculating intrinsic value: {e}")
        return None


# Function to read historical sp500 ticker list from a CSV file
def get_tickers_list(input_file="sp500_ticker_periods_tested.csv"):
    try:
        tickers_sp500_data = pd.read_csv(input_file)
        tickers_list = tickers_sp500_data["Ticker"].to_list()

        # Return a limited subset of tickers for testi
        return tickers_list[:5]
    except Exception as e:
        logging.error(f"Error reading tickers list: {e}")
        return []


def save_to_csv(ticker, earnings_data, annualized_growth_rate):
    try:
        # Adding the calculated annualized growth rate to the data
        earnings_data["Annualized Growth Rate"] = annualized_growth_rate
        csv_filename = f"backtest_data/{ticker}_modelling_data.csv"

        # Save the data to a CSV file
        earnings_data.to_csv(csv_filename, index=False)
        logging.info(f"Data saved to {csv_filename}")
    except Exception as e:
        logging.error(f"Error saving data for {ticker}: {e}")


# Fetch and save modelling data for a list of tickers.
# Retrieves earnings data, calculates growth rates, and saves the data for each ticker.
def get_modelling_data():
    ticker_list = get_tickers_list()

    # Create directory to save data
    os.makedirs("backtest_data", exist_ok=True)

    for ticker in ticker_list:
        try:
            # Fetch earnings data for the ticker
            earnings_data = fetch_yahoo_earnings_data(ticker)
            if earnings_data.empty:
                continue

            # Calculate annualized growth rate
            annualized_growth_rate = calculate_growth_rate(earnings_data)
            if annualized_growth_rate is not None:
                # Save the data if growth rate is calculated successfully
                save_to_csv(ticker, earnings_data, annualized_growth_rate)
        except Exception as e:
            logging.error(f"Error processing data for {ticker}: {e}")
