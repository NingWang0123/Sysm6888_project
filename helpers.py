import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    
