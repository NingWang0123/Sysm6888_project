def filter_out_cate(df,column):
    if len(filtered_df[column].unique())>10:
        print('column needs to be removed')
    else:
        print('column does not need to be removed')