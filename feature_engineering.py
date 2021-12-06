import tuned_constants
import pandas as pd
import numpy as np
import talib

def macd_indicators(raw_df):
    exp1 = raw_df.Close.ewm(span=tuned_constants.MACD_1, adjust=False).mean()
    exp2 = raw_df.Close.ewm(span=tuned_constants.MACD_2, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=tuned_constants.MACD_SIGNAL, adjust=False).mean()
    raw_df['MACD'] = macd
    raw_df['MACD_signal'] = signal
    raw_df['MACD_hist'] = macd -  signal
    return raw_df

# Reference: https://blog.quantinsti.com/build-technical-indicators-in-python/
# Commodity Channel Index 
def CCI(raw_df): 
    tp = (raw_df['High'] + raw_df['Low'] + raw_df['Close']) / 3 
    sma = tp.rolling(tuned_constants.CCI).mean()
    mad = tp.rolling(tuned_constants.CCI).apply(lambda x: pd.Series(x).mad())
    raw_df['CCI'] = (tp - sma) / (0.015 * mad) 
    return raw_df

def MTM(raw_df):
    # MTM= (Current Closing Price – Prior Closing Price) x Prior Quantity x Multiplier
    diff = raw_df.Close.diff(tuned_constants.MTM)
    prev_vol = raw_df.Close.shift(tuned_constants.MTM)
    #With prior volume
    raw_df['MTM'] = diff*prev_vol
    #with current_vol
#     raw_df['MTM'] = diff*raw_df['Volume']
    return raw_df

def ROC(raw_df):
    prev_close = raw_df.Close.shift(tuned_constants.ROC)
    raw_df['ROC']  = (raw_df['Close'] - prev_close) / prev_close
    return raw_df

def RSI(raw_df, periods = 14):
#   Reference:   https://www.roelpeters.be/many-ways-to-calculate-the-rsi-in-python-pandas/
    """
    Returns a pd.Series with the relative strength index.
    """
    periods = tuned_constants.RSI
    close_delta = raw_df['Close'].diff()
    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    raw_df['RSI'] = rsi
    return raw_df

def stochastics(raw_df):
    # Resource: #     https://stackoverflow.com/questions/30261541/slow-stochastic-implementation-in-python-pandas
    """
    Fast stochastic calculation
    %K = (Current Close - Lowest Low)/
    (Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K

    Slow stochastic calculation
    %K = %D of fast stochastic
    %D = 3-day SMA of %K

    When %K crosses above %D, buy signal 
    When the %K crosses below %D, sell signal
    """

    k = tuned_constants.SLOW_K
    d = tuned_constants.SLOW_D
    # Set minimum low and maximum high of the k stoch
    low_min  = raw_df['Low'].rolling( window = k ).min()
    high_max = raw_df['High'].rolling( window = k ).max()

    # Fast Stochastic
    k_fast = 100 * (raw_df['Close'] - low_min)/(high_max - low_min)
    d_fast = k_fast.rolling(window = d).mean()

    # Slow Stochastic
    raw_df['SLOW_K'] = d_fast
    raw_df['SLOW_D'] = raw_df['SLOW_K'].rolling(window = d).mean()
    return raw_df

def adosc(raw_df):
    raw_df['ADOSC'] = talib.ADOSC(raw_df['High'],raw_df['Low'],raw_df['Close'],raw_df['Volume'],
                            slowperiod = tuned_constants.ADSOC_SLOW,
                           fastperiod = tuned_constants.ADSOC_SLOW)
    return raw_df

def AR(raw_df):
    high,low = raw_df['High'], raw_df['Low']
    aroondown, aroonup = talib.AROON(high, low, timeperiod=14)
    raw_df['AR_D'], raw_df['AR_U'] = aroondown, aroonup
    return raw_df

def VR(raw_df):
    log_ret = np.log(raw_df['Close'] / raw_df['Close'].shift(1))
    raw_df['VR'] = log_ret.rolling(window=tuned_constants.VR).std() * np.sqrt(tuned_constants.VR)
    return raw_df

def bias(raw_df):
    # BIAS = [ (Closing price of the day — N-day average price) / N-day average price ] * 100%
    SMA = raw_df.Close.rolling(tuned_constants.BIAS).mean()
    raw_df['bias'] = ((raw_df.Close - SMA)*100)/SMA
    return raw_df

def add_indicators(raw_df):
    raw_df['price_change'] = raw_df.Close.diff(tuned_constants.PRICE_CHANGE)
    raw_df['price_change_pct'] = raw_df.Close.pct_change(tuned_constants.PRICE_CHANGE)
    raw_df['SMA'] = raw_df.Close.rolling(tuned_constants.SMA).mean()
    # MACD 'MACD','MACD_signal','MACD_hist'
    raw_df = macd_indicators(raw_df)
    raw_df = CCI(raw_df)
    raw_df = MTM(raw_df)
    raw_df = ROC(raw_df)
    raw_df = RSI(raw_df)
    #SLOW_K and SLOW_D
    raw_df = stochastics(raw_df)
    #AR_D, AR_U
    raw_df = adosc(raw_df)
    raw_df = VR(raw_df)
    raw_df = bias(raw_df)
    return raw_df
