from loss_model import load_the_data, remove_full_na, na_share_threshold, path, line_step, date_col_list, days_diff, days_diff_list, neg_to_zero_col_list, neg_to_zero_dict, neg_to_none_col_list, neg_to_none_dict, years_diff_list, years_diff, today_col, exeptions_list, num_to_none, neg_to_none_string_dict, neg_to_none_col_string_list, na_to_unknown_list, na_to_other_list, to_drop_list_beginning, cor_selection, to_drop_list_irrelevant, shopping_col, train_val_test_split_date, target_col, classifier_column, tree_model_train, tree_model_type, tree_model_name, constant_params_catboost, cat_var_list, filtered_column_list, num_to_int_col_list
import pandas as pd
import numpy as np

def main():
    #load the data
    df = load_the_data(path, line_step, is_test=True)
    # df = load_the_data(path)

    #drop columns
    df = df.drop(to_drop_list_beginning, axis=1)
    df = df.drop(to_drop_list_irrelevant, axis=1)

    #convert to datetime
    for i in date_col_list:
        df[i] = pd.to_datetime(df[i], utc=True)
    
    #days diff calculation
    for i in days_diff_list:
        df[i + '_days'] = days_diff(df['decision_time'], df[i])
        df = df.drop(columns=i)
    
    #years diff calculation
    df = years_diff(df=df,today=today_col,years_diff_list=years_diff_list)

    #replace negative values (numeric) with None
    df = num_to_none(df=df, replacements=neg_to_none_dict)
    # replace negative values (strings) with None
    df[neg_to_none_col_string_list] = df[neg_to_none_col_string_list].replace(neg_to_none_string_dict)
    
    # #replacing negative values with zero
    # df[neg_to_zero_col_list] = df[neg_to_zero_col_list].replace(neg_to_zero_dict)
    # #replacing negative values with None
    # df[neg_to_none_col_list] = df[neg_to_none_col_list].replace(neg_to_none_dict)

    ## Fill NA's with 'UNKNOWN'
    df[na_to_unknown_list] = df[na_to_unknown_list].fillna(value='UNKNOWN')
    ## Fill NA's with 'Other'
    df[na_to_other_list] = df[na_to_other_list].fillna(value='Other')

    # Remove NA's
    df = remove_full_na(df = df, na_share_threshold = na_share_threshold, exeptions = exeptions_list)

    # create the target
    df['unpaid_at_60_rate'] = df['unpaid_at_60'] / df['captured_amount']
    df = df.drop(columns='unpaid_at_60')

    # Create the target column
    df = classifier_column(
        df=df
        , original_column_name='unpaid_at_60_rate'
        , condition_value=0.1
        , new_column_name='is_default'
        , is_bigger_condition=True
        , drop_the_original=True)

    ## Calculate and drop high correlated variables
    #
    columns_to_drop_list_corr = cor_selection(df=df,target_col=target_col, exeptions_list=exeptions_list)

    #
    df = df.drop(columns_to_drop_list_corr, axis=1)

    # Create is_shopping column
    df = shopping_col(df=df, merchant_id_col='merchant_id')

    # Change numeric categorical columns to integer
    for col in num_to_int_col_list:
        if np.isin(col, df.columns) == True:
            df[col] = df[col].astype(str)

    # Get the categotical columns list that is still in the data (after all filtering)
    filtered_col_list = filtered_column_list(df=df, col_list=cat_var_list)
    # Fill NA with 'other' for categorical variables
    for col in filtered_col_list:
        df[col] = df[col].fillna(value='Other')

    # Split the data
    X_train, X_validation, X_test, y_train, y_validation, y_test = train_val_test_split_date(
        df = df, target = target_col, sorty_by = 'decision_time', train_share = 70, val_share = 15)

    # Train the model
    model = tree_model_train(
        model_type=tree_model_type
        , model_name=tree_model_name
        , constant_params=constant_params_catboost
        , categorical_variables=filtered_col_list
        , X_train=X_train
        , y_train=y_train
        , X_validation=X_validation
        , y_validation=y_validation
        , months=3
        , year='2020')




    
    # # Create test df
    # df_test = df[['decision_time', 'is_shopping', 'unpaid_at_60_rate', 'is_default']]

    # #
    # print(df_test.head())
    # print('\n')
    # print(len(df))
    # print('\n')
    # print(len(df.columns))

if __name__ == "__main__":
    main()