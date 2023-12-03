import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam

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
    
def create_model(hidden_layer_sizes, activation, solver, alpha, learning_rate, input_dim):
    model = Sequential()
    model.add(Dense(hidden_layer_sizes[0], input_dim=input_dim, activation=activation))
    for units in hidden_layer_sizes[1:]:
        model.add(Dense(units, activation=activation))
    model.add(Dense(1))  # Output layer for regression

    if solver == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    elif solver == 'adam':
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
    return model

def evaluate_keras_model(model, X, y, cv_splits):
    rmse_scores = []
    for train_idx, test_idx in cv_splits.split(X):
        X_train_k, X_test_k = X[train_idx], X[test_idx]
        y_train_k, y_test_k = y[train_idx], y[test_idx]
        model.fit(X_train_k, y_train_k, epochs=1000, batch_size=32, verbose=0)
        predictions = model.predict(X_test_k)
        mse = np.mean((predictions.flatten() - y_test_k) ** 2)
        rmse_scores.append(np.sqrt(mse))
    return np.mean(rmse_scores)


