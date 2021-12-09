import tuned_constants
import pandas as pd
import numpy as np
import talib
from feature_engineering import *
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def shrink_data(raw_df,horizon = -1):
    raw_df = raw_df[pd.to_datetime(raw_df.index)>pd.to_datetime(tuned_constants.START_DATE)]
    X_cols = ['Close','Open', 'High', 'Low', 'Adj Close', 'Volume', 'price_change',
       'price_change_pct', 'SMA', 'MACD', 'MACD_signal', 'MACD_hist', 'CCI',
       'MTM', 'ROC', 'RSI', 'SLOW_K', 'SLOW_D', 'ADOSC', 'VR', 'bias']
    #horizon_prediction
    horizon = -1
    X = raw_df[X_cols]
    Y = create_Y_up_down(raw_df,horizon=horizon)
    #Change Y after looking at the paper
    X = X.iloc[abs(horizon):horizon]
    Y = Y.iloc[abs(horizon):horizon]
    return raw_df, X, Y

def feature_selection(X,Y):
    # Feature selection. 
    model = RandomForestClassifier()
    fs_model = RFE(model)
    fs_model.fit(X,Y)
    X = X.loc[:,fs_model.support_]
    return X

def dim_red(X):
    pca = PCA(n_components=9)
    pca.fit(X)
    ratios = pca.explained_variance_ratio_
    sum = 0
    for i in range(ratios.shape[0]):
        len_ = i
        if sum > 0.96:
            break
        sum += ratios[i]
        
    pca = PCA(n_components= len_)
    pca.fit(X)
    X_pca = pca.transform(X)
    return X_pca

def scaling(X,raw_df):
    # normalize the dataset
    y_scalar = MinMaxScaler(feature_range=(0,1))
    y = y_scalar.fit_transform(raw_df[['Close']]) 
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    return X

# def create_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back-1):
#         a = dataset[i:(i+look_back), 0]
#         dataX.append(a)
#         # dataY.append( np.sign( dataset[i + look_back, 0] - dataset[i] )  )
#     return np.array(dataX), np.array(dataY)

def create_dataset(dataset,Y, look_back=1):
    dataX, dataY = [], []
    for i in range(look_back,len(dataset)-1):
      a = dataset[i-look_back:i, :]
      dataX.append(a)
      dataY.append(Y[i])
      # dataY.append( np.sign( dataset[i + look_back, 0] - dataset[i] )  )
    return np.array(dataX), np.array(dataY)

def train_test_split(X,Y,look_back = 1):
    # split into train and test sets
    train_size = int(len(X) * 0.67)
    test_size = len(X) - train_size
    train, test = X[0:train_size,:], X[train_size:,:]
    train_y, test_y = Y.iloc[0:train_size],Y.iloc[train_size:]

    train_x,train_y = create_dataset(train,train_y)
    test_x, test_y = create_dataset(test,test_y)
    # reshape input to be [samples, time steps, features]
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
    
    return train_x,test_x,train_y,test_y

