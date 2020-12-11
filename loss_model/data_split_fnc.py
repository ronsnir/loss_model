# %%
def train_val_test_split_date(df, target, sorty_by, train_share, val_share):
    
    #sort the df by the sort_by argument
    sorted_df = df.sort_values(by = sorty_by, ascending = True).reset_index(drop=True)
    
    #calculate the share of train, val, and test
    train_share_n = round(train_share/100, 2)
    val_share_n = round(val_share/100, 2)
    # test_share_test = round(1 - train_share_n - val_share_n, 2)
    

    #calculate the numbers of rows for the train/val df
    train_val_index = round( (train_share_n + val_share_n) * len(sorted_df) )
    
    #split the data to train/val & test
    train_val_df = sorted_df.iloc[:train_val_index]
    test_df = sorted_df.iloc[train_val_index:]

    #delete initial df
    del sorted_df
    del train_val_index

    #split the train/val df randomly
    ##create the new shares
    train_share_n_split = train_share_n / (train_share_n + val_share_n)
    # val_share_n_split = val_share_n / (train_share_n + val_share_n)
    ##create X and y for the split
    X = train_val_df.drop([target], axis=1)
    y = train_val_df[target]
    ##split
    from sklearn.model_selection import train_test_split
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=train_share_n_split, random_state=101)

    #create X and y for the test
    X_test = test_df.drop([target], axis=1)
    y_test = test_df[target]

    #delete df's
    del train_val_df
    del test_df
    del X
    del y

    #return
    return X_train, X_validation, X_test, y_train, y_validation, y_test