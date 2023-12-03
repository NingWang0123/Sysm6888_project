import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import mean_squared_error

def filter_out_cate(df,column):
    if len(df[column].unique())>10:
        print('column needs to be removed')
    else:
        print('column does not need to be removed')
        
        
def visualize_data(df):
    sns.set(style="whitegrid")
    scatter_columns = ['District', 'Stories', 'Year_Built', 'Units', 'Bdrms', 'Fbath', 'Hbath', 'Lotsize']
    for col in scatter_columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[col], y=df['Sale_price'])
        plt.title(f'{col} vs Sale_price')
        plt.xlabel(col)
        plt.ylabel('Sale_price')
        plt.show()

    df['COVID_Status'] = df.apply(lambda x: 'COVID Period' if x['covid_period_yes'] == 1 else 'Non-COVID Period', axis=1)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='COVID_Status', y='Sale_price', data=df)
    plt.title('Boxplot of Sale_price by COVID Status')
    plt.xlabel('COVID Status')
    plt.ylabel('Sale Price')
    plt.show()
    
def create_model(layers, activation, solver, alpha, learning_rate, input_dim):
    model = Sequential()
    for i, layer_size in enumerate(layers):
        if i == 0:
            model.add(Dense(layer_size, activation=activation, input_dim=input_dim))
        else:
            model.add(Dense(layer_size, activation=activation))
    model.compile(optimizer=solver, loss='mean_squared_error')
    return model

def create_keras_model(layers, activation, solver, alpha, learning_rate, input_dim):
    model = Sequential()
    for i, layer_size in enumerate(layers):
        if i == 0:
            model.add(Dense(layer_size, activation=activation, input_dim=input_dim))
        else:
            model.add(Dense(layer_size, activation=activation))
    model.add(Dense(1, activation='linear')) 
    model.compile(optimizer=solver, loss='mean_squared_error')
    return model




def evaluate_keras_model(model, X, y, cv_splits):
    rmse_scores = []
    for train_idx, test_idx in cv_splits.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, X_test, y_train, y_test = map(
            lambda array: np.array(array).astype('float32'),
            [X_train, X_test, y_train, y_test]
        )

        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            raise ValueError("NaN or Inf in X_train")
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            raise ValueError("NaN or Inf in y_train")
        if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
            raise ValueError("NaN or Inf in X_test")
        if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)):
            raise ValueError("NaN or Inf in y_test")

        model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=0)

        predictions = model.predict(X_test).flatten()

        mse = np.mean((predictions - y_test) ** 2)
        rmse_scores.append(np.sqrt(mse))

    return np.mean(rmse_scores)


def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def fit_keras_model(model,X_train,y_train):
    X_train, y_train = map(
            lambda array: np.array(array).astype('float32'),
            [X_train, y_train]
        )
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        raise ValueError("NaN or Inf in X_train")
    if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
        raise ValueError("NaN or Inf in y_train")
    model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=0) 
    return model

def keras_rmse(X_test,Y_test,model):
    X_test, Y_test = map(
            lambda array: np.array(array).astype('float32'),
            [X_test, Y_test]
        )
    if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
        raise ValueError("NaN or Inf in X_train")
    if np.any(np.isnan(Y_test)) or np.any(np.isinf(Y_test)):
        raise ValueError("NaN or Inf in y_train")
    predictions = model.predict(X_test).flatten()

    mse = np.mean((predictions - y_test) ** 2)
    return np.sqrt(mse)

