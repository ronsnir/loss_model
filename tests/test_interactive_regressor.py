# %%
from loss_model import load_the_data, remove_full_na, na_share_threshold, path, line_step, date_col_list, days_diff, days_diff_list, neg_to_zero_col_list, neg_to_zero_dict, neg_to_none_col_list, neg_to_none_dict, years_diff_list, years_diff, today_col, exeptions_list, num_to_none, neg_to_none_string_dict, neg_to_none_col_string_list, na_to_unknown_list, na_to_other_list, to_drop_list_beginning, cor_selection, to_drop_list_irrelevant, shopping_col, train_val_test_split_date, classifier_column, tree_model_train, tree_model_type, tree_model_name, constant_params_catboost, cat_var_list, filtered_column_list, num_to_int_col_list, model_report, target_col_class, target_col_reg, to_drop_list_end
import pandas as pd
import numpy as np
# %%load the data
df = load_the_data(path, line_step, is_test=True)
# df = load_the_data(path)
# %%
# df_copy = df
# # %%
# df = df_copy

# # %%Drop rows with NA's in unpaid_at_60
# df_copy.dropna(subset=['unpaid_at_60'])

# %% define target column by model type
if tree_model_type == 'regression':
    target_col = target_col_reg
elif tree_model_type == 'classification':
    target_col = target_col_class

# %%drop columns
df = df.drop(to_drop_list_beginning, axis=1)
df = df.drop(to_drop_list_irrelevant, axis=1)

# %%convert to datetime
for i in date_col_list:
    df[i] = pd.to_datetime(df[i], utc=True)

# %%days diff calculation
for i in days_diff_list:
    df[i + '_days'] = days_diff(df['decision_time'], df[i])
    df = df.drop(columns=i)

# %%years diff calculation
df = years_diff(df=df,today=today_col,years_diff_list=years_diff_list)

# %%replace negative values (numeric) with None
df = num_to_none(df=df, replacements=neg_to_none_dict)
# %% replace negative values (strings) with None
df[neg_to_none_col_string_list] = df[neg_to_none_col_string_list].replace(neg_to_none_string_dict)

# #replacing negative values with zero
# df[neg_to_zero_col_list] = df[neg_to_zero_col_list].replace(neg_to_zero_dict)
# #replacing negative values with None
# df[neg_to_none_col_list] = df[neg_to_none_col_list].replace(neg_to_none_dict)

# %%# Fill NA's with 'UNKNOWN'
df[na_to_unknown_list] = df[na_to_unknown_list].fillna(value='UNKNOWN')
# %%# Fill NA's with 'Other'
df[na_to_other_list] = df[na_to_other_list].fillna(value='Other')

# %% Remove NA's
df = remove_full_na(df = df, na_share_threshold = na_share_threshold, exeptions = exeptions_list)

# %% create the target
df['unpaid_at_60_rate'] = df['unpaid_at_60'] / df['captured_amount']
df = df.drop(columns='unpaid_at_60')

# %% Create the target column
if tree_model_type == 'classification':
    df = classifier_column(
        df=df
        , original_column_name='unpaid_at_60_rate'
        , condition_value=0.1
        , new_column_name='is_default'
        , is_bigger_condition=True
        , drop_the_original=True)
elif tree_model_type == 'regression':
    df = df

# %%# Calculate and drop high correlated variables
#
columns_to_drop_list_corr = cor_selection(df=df,target_col=target_col, exeptions_list=exeptions_list)
if np.isin(target_col,columns_to_drop_list_corr) == True:
    columns_to_drop_list_corr.remove(target_col)
# %%
df = df.drop(columns_to_drop_list_corr, axis=1)

# %% Create is_shopping column
df = shopping_col(df=df, merchant_id_col='merchant_id')

# %% Change numeric categorical columns to string
for col in num_to_int_col_list:
    if np.isin(col, df.columns) == True:
        df[col] = df[col].astype(str)

# %% Get the categotical columns list that is still in the data (after all filtering)
filtered_col_list = filtered_column_list(df=df, col_list=cat_var_list)
# %% Fill NA with 'other' for categorical variables
for col in filtered_col_list:
    df[col] = df[col].fillna(value='Other')

# # %% Drop columns (end)
# df = df.drop(to_drop_list_end, axis=1)

# %% Get the final columns list
final_col_list = df.columns.to_list()

# %% Split the data
X_train, X_validation, X_test, y_train, y_validation, y_test = train_val_test_split_date(
    df = df, target = target_col, sorty_by = 'decision_time', train_share = 70, val_share = 15)

# %% Train the model
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

# %%
threshold=0.025
probs = model_report(model=model, model_name='catboost', tree_model_type='regression', threshold=threshold, X_train=X_train, X_test=X_test, y_train=y_train, y_validation=y_validation, y_test=y_test)

# %%
# %%
probs_test = model.predict(X_test)









########## PREDICTIONS ##########
# %%
import joblib
# %% load
model = joblib.load('/home/ron.snir/git/loss_model/tests/model_3_months_2020__catboost')

# %%
df_pred = load_the_data('/home/ron.snir/git/loss_model/data/pi4_immature000.csv')
# %%
df_pred_copy = df_pred
# %%drop columns
df_pred = df_pred.drop(to_drop_list_beginning, axis=1)
df_pred = df_pred.drop(to_drop_list_irrelevant, axis=1)

# %%convert to datetime
for i in date_col_list:
    df_pred[i] = pd.to_datetime(df_pred[i], utc=True)

# %%
df_pred = df_pred[(df_pred['decision_time'] >= '2020-10-01') & (df_pred['decision_time'] <= '2020-11-01')]


# %%days diff calculation
for i in days_diff_list:
    df_pred[i + '_days'] = days_diff(df_pred['decision_time'], df_pred[i])
    df_pred = df_pred.drop(columns=i)

# %%years diff calculation
df_pred = years_diff(df=df_pred,today=today_col,years_diff_list=years_diff_list)

# %%replace negative values (numeric) with None
df_pred = num_to_none(df=df_pred, replacements=neg_to_none_dict)
# %% replace negative values (strings) with None
df_pred[neg_to_none_col_string_list] = df_pred[neg_to_none_col_string_list].replace(neg_to_none_string_dict)

# %%# Fill NA's with 'UNKNOWN'
df_pred[na_to_unknown_list] = df_pred[na_to_unknown_list].fillna(value='UNKNOWN')
# %%# Fill NA's with 'Other'
df_pred[na_to_other_list] = df_pred[na_to_other_list].fillna(value='Other')

# %% create the target
df_pred['unpaid_at_60_rate'] = df_pred['unpaid_at_60'] / df_pred['captured_amount']
df_pred = df_pred.drop(columns='unpaid_at_60')

# %% Create the target column
if tree_model_type == 'classification':
    df_pred = classifier_column(
        df=df_pred
        , original_column_name='unpaid_at_60_rate'
        , condition_value=0.1
        , new_column_name='is_default'
        , is_bigger_condition=True
        , drop_the_original=True)
elif tree_model_type == 'regression':
    df_pred = df_pred

# %% Create is_shopping column
df_pred = shopping_col(df=df_pred, merchant_id_col='merchant_id')

# %% Change numeric categorical columns to string
for col in num_to_int_col_list:
    if np.isin(col, df_pred.columns) == True:
        df_pred[col] = df_pred[col].astype(str)

# %% Get the categotical columns list that is still in the data (after all filtering)
filtered_col_list = filtered_column_list(df=df_pred, col_list=cat_var_list)
# %% Fill NA with 'other' for categorical variables
for col in filtered_col_list:
    df_pred[col] = df_pred[col].fillna(value='Other')

# %%
df_pred = df_pred[final_col_list]
# %%
df_pred_features = df_pred.drop('unpaid_at_60_rate', axis=1)
df_pred_target = df_pred['unpaid_at_60_rate']

# %%
probs = model.predict(df_pred_features)

# %%
probs.mean()

# %%
df_pred_features_index_list = df_pred_features.index.to_list()
# %%
probs_d = {'predicted_unpaid_rate_at_60':probs}
# %%
probs_df = pd.DataFrame(data=probs_d, index=df_pred_features_index_list)

# %%
df_pred_features_pred = pd.concat([df_pred_features, probs_df], axis=1)

# %%
predicted_unpaid_browser_new = df_pred_features_pred[(df_pred_features_pred['is_shopping']=='1') & (df_pred_features_pred['rvl_cid_has_paid']==False)]['predicted_unpaid_rate_at_60'].mean()
predicted_unpaid_browser_returning = df_pred_features_pred[(df_pred_features_pred['is_shopping']=='1') & (df_pred_features_pred['rvl_cid_has_paid']==True)]['predicted_unpaid_rate_at_60'].mean()
predicted_unpaid_merchant_new = df_pred_features_pred[(df_pred_features_pred['is_shopping']=='0') & (df_pred_features_pred['rvl_cid_has_paid']==False)]['predicted_unpaid_rate_at_60'].mean()
predicted_unpaid_merchant_returning = df_pred_features_pred[(df_pred_features_pred['is_shopping']=='0') & (df_pred_features_pred['rvl_cid_has_paid']==True)]['predicted_unpaid_rate_at_60'].mean()

# %%
print(f'The predicted unpaid rate for browser-new is: {round(predicted_unpaid_browser_new*100, 4)}%')
print(f'The predicted unpaid rate for browser-returning is: {round(predicted_unpaid_browser_returning*100, 4)}%')
print(f'The predicted unpaid rate for merchant-new is: {round(predicted_unpaid_merchant_new*100, 4)}%')
print(f'The predicted unpaid rate for merchant-returning is: {round(predicted_unpaid_merchant_returning*100, 4)}%')



###### TEST PRINTS ######
# %% Create test df
df_test = df[['decision_time', 'is_shopping', 'unpaid_at_60_rate', 'is_default']]

# %%
print(df_test.head())
print('\n')
print(len(df))
print('\n')
print(len(df.columns))