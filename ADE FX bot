import pandas as pd
import MetaTrader5 as mt5
from dotenv import load_dotenv
import numpy as np
import datetime
import logging
import time
import pytz
import os

# Function to get the trading session based on WAT (West Africa Time)
def get_trading_session():
    wat = pytz.timezone('Africa/Lagos')  # West Africa Time (GMT+1)
    current_time = datetime.datetime.now(wat)
    current_hour = current_time.hour
    current_weekday = current_time.weekday()

    # Check if it's within trading hours (7:00 AM to 5:00 PM WAT)
    if current_weekday < 5 and 7 <= current_hour < 17:
        # Define trading sessions
        if 7 <= current_hour < 10:  # Tokyo session (7:00 AM to 10:00 AM WAT)
            return 'Tokyo'
        elif 8 <= current_hour < 16:  # Frankfurt session (8:00 AM to 4:00 PM WAT)
            return 'Frankfurt'
        elif 8 <= current_hour < 17:  # London session (8:00 AM to 5:00 PM WAT)
            return 'London'
        elif 14 <= current_hour < 17:  # New York session (2:00 PM to 5:00 PM WAT)
            return 'New York'
    return None
        
# Function to generate the performance report
def generate_report():
    # Example code to generate the report (you can customize this as needed)
    print("Generating performance report...")
    # Placeholder for report generation logic (you'll customize this)
    report = {
        "Total Trades": 10,  # Example statistic
        "Win Rate": "80%",   # Example statistic
        "Total Profit": 500  # Example statistic
    }
    # Print or save the report as needed
    print(report)
    # You may want to log the report details to the log file
    logging.info(f"Performance Report: {report}")

# Logging setup code starts here
# Function to set up logging
def setup_logging():
    logging.basicConfig(
        filename="ADE_FX.log",  # Log file name
        level=logging.INFO,  # Logging level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        datefmt="%Y-%m-%d %H:%M:%S"  # Date format
    )
    logging.info("Logging system initialized.")

# Function to log account status
def log_account_status(balance, equity, margin):
    logging.info(f"Account Balance: {balance}, Equity: {equity}, Margin: {margin}")

# Function to log trade execution details
def log_trade_execution(order_type, symbol, lot_size, take_profit, stop_loss, result):
    if result is None:
        logging.error(f"{order_type} order for {symbol} failed. No result received.")
    elif result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"{order_type} order for {symbol} failed. Error code: {result.retcode}")
    else:
        logging.info(f"{order_type} order for {symbol} executed successfully. "
                     f"Lot Size: {lot_size}, Take Profit: {take_profit}, Stop Loss: {stop_loss}")

# Function to handle errors gracefully
def handle_error(exception):
    logging.error(f"An unexpected error occurred: {exception}")
    print(f"An unexpected error occurred: {exception}")

# Initialize the logging system
setup_logging()

# Function to print the name with a yellow background and black text, and pause for 5 seconds after showing the message
def print_initialization_message():
    print("\033[43m\033[30mADE FX Bot Initialized\033[0m")  # Yellow background, black text
    time.sleep(5)  # Pausing for 5 seconds

# Call the function to print the initialization message
print_initialization_message()

# Initialize the MetaTrader 5 terminal
if not mt5.initialize():
    print("Failed to initialize MT5")
    print("Error code:", mt5.last_error())
    exit()  # Exit if initialization fails

# Load environment variables
load_dotenv()

login = os.getenv("207293298")
password = os.getenv("Ola-jire1")
server = os.getenv("Exness-MT5Trial9")

if not login or not password or not server:
    print("Error: Missing login credentials.")
    exit()  # Exit if credentials are missing

# Define the login_to_account function here
def login_to_account(login, password, server):
    authorized = mt5.login(login, password, server)
    if authorized:
        print(f"Logged in successfully to account {login}")
    else:
        print(f"Login failed. Error code: {mt5.last_error()}")
        exit()

# Call the login function after loading credentials
login_to_account(login, password, server)

# Continue with the rest of your bot logic
print("Credentials loaded successfully")

# Define timeframes (15-minute, 30-minute, and 1-hour)
timeframes = {
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1
}

# ATR Calculation
def calculate_atr(data, period=14):
    data['high-low'] = data['high'] - data['low']
    data['high-close'] = np.abs(data['high'] - data['close'].shift(1))
    data['low-close'] = np.abs(data['low'] - data['close'].shift(1))
    tr = data[['high-low', 'high-close', 'low-close']].max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.iloc[-1]

# RSI Calculation
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)  # Handle NaN values by filling with 0
    
# Function to get the account balance
def get_balance():
    account_info = mt5.account_info()
    if account_info is not None:
        return account_info.balance
    return 0
    
    # Function to calculate the lot size (5% of balance)
def calculate_lot_size(balance, stop_loss_pips, symbol):
    risk_percentage = 5 / 100
    risk_amount = balance * risk_percentage
    pip_value = get_pip_value(symbol)  # Get pip value for the symbol
    lot_size = risk_amount / (stop_loss_pips * pip_value)
    return lot_size
    
    # Place Buy Order
def place_buy_order(symbol, lot_size, take_profit, stop_loss):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to retrieve symbol information for {symbol}. Skipping order.")
        return None

    symbol_info_tick = mt5.symbol_info_tick(symbol)
    if symbol_info_tick is None:
        logging.error(f"Failed to retrieve symbol tick data for {symbol}. Skipping order.")
        return None

    # Use symbol_info_tick to calculate TP/SL
    sl = symbol_info_tick.ask - (stop_loss * symbol_info.point)
    tp = symbol_info_tick.ask + (take_profit * symbol_info.point)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY,
        "price": symbol_info_tick.ask,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "ADE FX Automatic Buy",
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC
    }

    result = mt5.order_send(request)

    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info(f"Buy order executed successfully for {symbol}. Lot Size: {lot_size}, Take Profit: {take_profit}, Stop Loss: {stop_loss}")
    else:
        logging.error(f"Buy order failed for {symbol}, Error code: {result.retcode if result else 'No result received'}")

    return result

# Place Sell Order
def place_sell_order(symbol, lot_size, take_profit, stop_loss):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to retrieve symbol information for {symbol}. Skipping order.")
        return None

    symbol_info_tick = mt5.symbol_info_tick(symbol)
    if symbol_info_tick is None:
        logging.error(f"Failed to retrieve symbol tick data for {symbol}. Skipping order.")
        return None

    # Use symbol_info_tick to calculate TP/SL
    sl = symbol_info_tick.ask - (stop_loss * symbol_info.point)
    tp = symbol_info_tick.ask + (take_profit * symbol_info.point)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_SELL,
        "price": symbol_info_tick.bid,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "ADE FX Automatic Sell",
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC
    }

    result = mt5.order_send(request)

    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info(f"Sell order executed successfully for {symbol}. Lot Size: {lot_size}, Take Profit: {take_profit}, Stop Loss: {stop_loss}")
    else:
        logging.error(f"Sell order failed for {symbol}, Error code: {result.retcode if result else 'No result received'}")

    return result
    
    # Function to get market data for the specified timeframe
def get_data(symbol, timeframe, bars=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:  # Check for missing data
        logging.warning(f"Failed to retrieve data for {symbol}.")
        return pd.DataFrame()  # Return an empty DataFrame
    
    # Convert to DataFrame with correct columns
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert timestamp to readable date
    return df
    
# Function to detect market trend and identify ranging markets
def detect_trend(df, atr_threshold=0.001, trend_strength=3):
    # Calculate the current price and the 5-period Simple Moving Average (SMA)
    current_price = df['close'].iloc[-1]
    sma = df['SMA'].iloc[-1]

    # Check for uptrend and downtrend conditions
    price_above_sma = (df['close'][-trend_strength:] > df['SMA'][-trend_strength:]).all()
    price_below_sma = (df['close'][-trend_strength:] < df['SMA'][-trend_strength:]).all()

    if price_above_sma:
        return "Uptrend"
    elif price_below_sma:
        return "Downtrend"

    # Identify a ranging market using ATR and price range
    atr = calculate_atr(df)
    price_range = df['high'].max() - df['low'].min()

    if price_range < atr_threshold:
        return "Ranging"  # Market is ranging

    return "Ranging"  # Default to "Ranging" if no clear trend

# Function to calculate CCI (Commodity Channel Index)
def calculate_cci(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    moving_avg = tp.rolling(window=period).mean()
    mean_dev = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (tp - moving_avg) / (0.015 * mean_dev)
    return cci

# Function to add multiple moving averages to the dataframe
def add_moving_averages(df):
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_60'] = df['close'].rolling(window=60).mean()
    df['SMA_100'] = df['close'].rolling(window=100).mean()
    df['SMA_120'] = df['close'].rolling(window=120).mean()
    return df

# Function to generate a refined signal based on multiple moving averages
def generate_refined_signal(df):
    sma_threshold = 1  # Set a fixed value instead of random
    df['Refined Signal'] = 0  # Default to 0 (sell)

    buy_condition = (
        (df['close'] > df['SMA_5']) &
        (df['close'] > df['SMA_20'] * sma_threshold) &
        (df['close'] > df['SMA_30']) &
        (df['close'] > df['SMA_50']) &
        (df['close'] > df['SMA_100']) &
        (df['close'] > df['SMA_120']) &
        (df['RSI'] < 70) &
        (df['CCI'] > 100)
    )
    
    sell_condition = (
        (df['close'] < df['SMA_5']) &
        (df['close'] < df['SMA_20'] * sma_threshold) &
        (df['close'] < df['SMA_30']) &
        (df['close'] < df['SMA_50']) &
        (df['close'] < df['SMA_100']) &
        (df['close'] < df['SMA_120']) &
        (df['RSI'] > 30) &
        (df['CCI'] < -100)
    )

    df['Refined Signal'] = np.where(buy_condition, 1, df['Refined Signal'])
    df['Refined Signal'] = np.where(sell_condition, 0, df['Refined Signal'])

    return df

# Function to monitor open trades
def monitor_open_trades():
    open_positions = mt5.positions_get()
    if open_positions is None:
        print("No open trades found")
        return
    
    print("\n--- Open Trades ---")
    for position in open_positions:
        print(f"Symbol: {position.symbol}, Lot Size: {position.volume}, Profit: {position.profit}, "
              f"Price Open: {position.price_open}, Current Price: {position.price_current}")
    print("-------------------\n")

# Function to monitor closed trades
def monitor_closed_trades():
    from_time = datetime.datetime.now() - datetime.timedelta(days=1)
    to_time = datetime.datetime.now()

    closed_orders = mt5.history_deals_get(from_time, to_time)
    if closed_orders is None:
        print("No closed trades found in the last 24 hours")
        return

    print("\n--- Closed Trades ---")
    for order in closed_orders:
        # Convert timestamp to local time (WAT - Africa/Lagos)
        closed_time = datetime.datetime.fromtimestamp(order.time, pytz.timezone('Africa/Lagos'))
        
        # Print the details including the time closed
        print(f"Symbol: {order.symbol}, Volume: {order.volume}, Profit: {order.profit}, "
              f"Time Closed: {closed_time}")
    print("---------------------\n")

# Example market data with multiple symbols
symbols = ['XAUUSD', 'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']

# Filter available symbols
available_symbols = [sym for sym in mt5.symbols_get()]
symbols = [sym for sym in symbols if sym in available_symbols]

# Check if there are valid symbols
if not symbols:
    print("No available symbols for trading.")
    exit()

# Initialize the logging system
setup_logging()

# Your main trading function
def trading_bot():
    try:
        start_time = datetime.datetime.now()
        max_runtime = datetime.timedelta(hours=10)
        
        while datetime.datetime.now() - start_time < max_runtime:
            # Bot logic here
            current_session = get_trading_session()
            print(f"Current session: {current_session}")
            
            if current_session is None:
                print("Outside trading hours. Retrying in 1 minute...")
                time.sleep(60)
                continue

            # Define session-specific pairs
            session_pairs = []
            if current_session == 'Tokyo':
                session_pairs = ['USDJPY', 'EURJPY', 'AUDUSD', 'NZDUSD']
            elif current_session == 'Frankfurt':
                session_pairs = ['EURUSD', 'EURGBP', 'USDCHF', 'AUDUSD', 'GBPUSD']
            elif current_session == 'London':
                session_pairs = ['GBPUSD', 'EURCHF', 'USDCHF', 'GBPJPY']
            elif current_session == 'New York':
                session_pairs = ['XAUUSD', 'USDCAD', 'EURCAD', 'EURUSD']

            for symbol in session_pairs:
                try:
                    # Fetch market data for the symbol
                    df = get_data(symbol, timeframe)

                    if df.empty:
                        logging.warning(f"No data for {symbol}. Skipping...")
                        continue

                    if len(df) < 50:
                        logging.warning(f"Insufficient data for {symbol}. Skipping...")
                        continue

                    # Add moving averages to the dataframe
                    df = add_moving_averages(df)

                    # Calculate the RSI and fill any NaN values
                    df['RSI'] = calculate_rsi(df['close']).fillna(0)

                    # (Optional) Calculate other indicators (e.g., CCI)
                    df['CCI'] = calculate_cci(df)

                    # Calculate ATR (Average True Range)
                    atr = calculate_atr(df)  # ATR is typically used to determine volatility

                    # Check if ATR is calculated properly (no NaN values)
                    if pd.isna(atr):
                        logging.warning(f"ATR calculation failed for {symbol}. Skipping...")
                        continue  # Skip to the next symbol if ATR is invalid

                    # Calculate Take Profit and Stop Loss based on ATR
                    df['Take Profit'] = df['close'] + (atr * 2)  # Example: ATR-based Take Profit
                    df['Stop Loss'] = df['close'] - (atr * 1.5)  # Example: ATR-based Stop Loss

                    # Generate trading signals based on other indicators like moving averages, RSI, etc.
                    df = generate_refined_signal(df)

                    balance = get_balance()
                    stop_loss_pips = atr * 10  # Example: Use ATR to determine stop loss in pips
                    lot_size = calculate_lot_size(balance, stop_loss_pips)

                    # Execute trades based on the refined signal
                    if df['Refined Signal'].iloc[-1] == 1:
                        result = place_buy_order(symbol, lot_size, df['Take Profit'].iloc[-1], df['Stop Loss'].iloc[-1])
                        log_trade_execution("Buy", symbol, lot_size, df['Take Profit'].iloc[-1], df['Stop Loss'].iloc[-1], result)
                    elif df['Refined Signal'].iloc[-1] == 0:
                        result = place_sell_order(symbol, lot_size, df['Take Profit'].iloc[-1], df['Stop Loss'].iloc[-1])
                        log_trade_execution("Sell", symbol, lot_size, df['Take Profit'].iloc[-1], df['Stop Loss'].iloc[-1], result)

                except Exception as e:
                    handle_error(e)
                    continue

            time.sleep(900)  # Wait for 15 minutes before the next iteration

    except KeyboardInterrupt:
        print("Trading bot interrupted by user.")
    finally:
        generate_report()
        print("ADE FX is shutting down...")
        mt5.shutdown()
print("ADE FX Shutting down complete.")
