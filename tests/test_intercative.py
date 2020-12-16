# %%
import loss_model
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.utils import resample
from sklearn.calibration import calibration_curve
# %%
from statsmodels.stats.weightstats import DescrStatsW
# %%
import datetime
# %%
import maya as maya
# %%
import sys
# %%
for path in sys.path:
    print (path)

# %% to see all columns instead of truncated version
pd.options.display.max_columns = None

# %% load the data
df_sh = pd.read_csv('/home/ron.snir/git/loss_model/data/pi4_full000.csv', skiprows = lambda i: i % 10 != 0)
# %% create a copy
df_copy = df_sh
# # %%
# df = df_copy
# %%
df.info()
# %%
df.head()
# %%
na_list = loss_model.na_list_fnc(df_sh)
# # %%
# df_test = df[['consumer_date_of_birth', 'tu_dob']]
# %%
loss_model.custom_summary(df=df_sh,col='tu_freeze')
# # %%
# for i in list(df_test):
#     custom_summary(df=df_test,col=i)
# %% negative values to None RUN #####################
df_ttt2 = loss_model.num_to_none(df=df_sh, replacements=loss_model.neg_to_none_dict)

# %% drop irrelevant features RUN #####################
df_ttt2 = df_ttt2.drop(loss_model.to_drop_list_beginning, axis=1)

# %% RUN ##################### -------------CHECK
na_share_threshold = 95
df_ttt = loss_model.remove_full_na(df = df_ttt2, na_share_threshold = na_share_threshold, exeptions = loss_model.exeptions_list)
# %% check NA's
loss_model.na_list_fnc(df_ttt)
# %% RUN #####################
na_list = loss_model.na_list_fnc(df_ttt)
# %%
loss_model.custom_summary(df=X_train,col='usemailageemailiprisk_ip_postalcode')
# %% RUN #####################
df_ttt['payment_method_card_level'] = df_ttt['payment_method_card_level'].fillna(value='UNKNOWN')
# %%
loss_model.custom_summary(df=df_ttt,col='usemailageemailiprisk_source_industry')
# %% RUN #####################
df_ttt['usemailageemailiprisk_source_industry'] = df_ttt['usemailageemailiprisk_source_industry'].fillna(value='Other')
# %%
loss_model.custom_summary(df=df_ttt,col='usemailageemailiprisk_lastflaggedon')

# %% RUN #####################
for i in loss_model.date_col_list:
    df_ttt[i] = pd.to_datetime(df_ttt[i], utc=True)


# %% days diff calculation  RUN #####################
for i in loss_model.days_diff_list:
    df_ttt[i + '_days'] = loss_model.days_diff(df_ttt['decision_time'], df_ttt[i])

# %%
loss_model.custom_summary(df=df_ttt,col='psp_credit_card_issuing_bank')

# %% RUN #####################
df_ttt['psp_credit_card_issuing_bank'] = df_ttt['psp_credit_card_issuing_bank'].fillna(value='UNKNOWN')

# # %%
# loss_model.custom_summary(df=df_ttt,col='rvl_cid_num_paid_sibc_24m')

# # %% RUN #####################
# df_ttt['rvl_cid_num_paid_sibc_24m'] = df_ttt['rvl_cid_num_paid_sibc_24m'].replace(loss_model.neg_to_zero_dict)

# # %%
# loss_model.custom_summary(df=df_ttt,col='rvl_cid_num_paid_sibc_12m')
# # %% RUN #####################
# df_ttt['rvl_cid_num_paid_sibc_12m'] = df_ttt['rvl_cid_num_paid_sibc_12m'].replace(loss_model.neg_to_zero_dict)

# # %%
# loss_model.custom_summary(df=df_ttt,col='rvl_cid_days_since_last_payment_inv')

# # %% RUN #####################
# df_ttt['rvl_cid_days_since_last_payment_inv'] = df_ttt['rvl_cid_days_since_last_payment_inv'].replace(loss_model.neg_to_none_dict)

# # %%
# loss_model.custom_summary(df=df_ttt,col='rvl_cid_days_since_last_payment_credit')

# # %% RUN #####################
# df_ttt['rvl_cid_days_since_last_payment_credit'] = df_ttt['rvl_cid_days_since_last_payment_credit'].replace(loss_model.neg_to_none_dict)

# # %%
# loss_model.custom_summary(df=df_ttt,col='rvl_cid_num_invoice_0_24m')

# # %% RUN #####################
# df_ttt['rvl_cid_num_invoice_0_24m'] = df_ttt['rvl_cid_num_invoice_0_24m'].replace(loss_model.neg_to_zero_dict)

# # %%
# loss_model.custom_summary(df=df_ttt,col='rvl_cid_days_since_last_payment_acct')

# # %% RUN #####################
# df_ttt['rvl_cid_days_since_last_payment_acct'] = df_ttt['rvl_cid_days_since_last_payment_acct'].replace(loss_model.neg_to_none_dict)

# # %%
# loss_model.custom_summary(df=df_ttt,col='rvl_cid_num_unpaid_orders')

# # %% RUN #####################
# df_ttt['rvl_cid_num_unpaid_orders'] = df_ttt['rvl_cid_num_unpaid_orders'].replace(loss_model.neg_to_zero_dict)

# # %%
# loss_model.custom_summary(df=df_ttt,col='crid_realtime_num_unpaid_sibc')

# # %% RUN #####################
# df_ttt['crid_realtime_num_unpaid_sibc'] = df_ttt['crid_realtime_num_unpaid_sibc'].replace(loss_model.neg_to_zero_dict)

# # %%
# loss_model.custom_summary(df=df_ttt,col='rvl_cid_num_unpaid_sibc')

# # %% RUN #####################
# df_ttt['rvl_cid_num_unpaid_sibc'] = df_ttt['rvl_cid_num_unpaid_sibc'].replace(loss_model.neg_to_zero_dict)

# # %%
# loss_model.custom_summary(df=df_ttt,col='rvl_cid_num_unpaid_sibc_orders')

# # %% RUN #####################
# df_ttt['rvl_cid_num_unpaid_sibc_orders'] = df_ttt['rvl_cid_num_unpaid_sibc_orders'].replace(loss_model.neg_to_zero_dict)

# %%
loss_model.custom_summary(df=df_ttt,col='consumer_date_of_birth')

# %% RUN #####################
df_ttt['consumer_date_of_birth'] = pd.to_datetime(df_ttt['consumer_date_of_birth'], utc=True)

# %% RUN #####################
df_ttt = loss_model.years_diff(df=df_ttt,today=loss_model.today_col,years_diff_list=loss_model.years_diff_list)



# # %%
# loss_model.custom_summary(df=df_ttt,col='rvl_eid_estore_id_avg_ts_0_12m')

# # %% RUN #####################
# df_ttt['rvl_eid_estore_id_avg_ts_0_12m'] = df_ttt['rvl_eid_estore_id_avg_ts_0_12m'].replace(loss_model.neg_to_none_dict)

# # %%
# loss_model.custom_summary(df=df_ttt,col='rvl_eid_estore_id_fraud_rate_30d')

# # %% RUN #####################
# df_ttt['rvl_eid_estore_id_fraud_rate_30d'] = df_ttt['rvl_eid_estore_id_fraud_rate_30d'].replace(loss_model.neg_to_zero_dict)

# # %%
# loss_model.custom_summary(df=df_ttt,col='rvl_eid_estore_id_fraud_cnt_30d')

# # %% RUN #####################
# df_ttt['rvl_eid_estore_id_fraud_cnt_30d'] = df_ttt['rvl_eid_estore_id_fraud_cnt_30d'].replace(loss_model.neg_to_zero_dict)

# %%
loss_model.custom_summary(df=df_ttt,col='rvl_cid_payment_method_2nd_last_purchase_0_3m')
loss_model.custom_summary(df=df_ttt,col='rvl_cid_payment_method_last_purchase_0_3m')
# %% get object columns
df_ttt.select_dtypes(include=['object']).columns.tolist()

# # %%
# df_ttt[neg_to_none_col_string_list] = df_ttt[neg_to_none_col_string_list].replace(neg_to_none_string_dict)

# %%
loss_model.custom_summary(df=df_ttt,col='payment_method_credit_card_client_reference')

# %% get object columns
df_ttt.select_dtypes(include=['object']).columns.tolist()

# %% create the target
df_ttt['unpaid_at_60_rate'] = df_ttt['unpaid_at_60'] / df_ttt['captured_amount']










# %% ##### CORRELATION
# %% RUN #####################
columns_to_drop_list = loss_model.cor_selection(df=df_ttt,target_col='unpaid_at_60_rate')

# %% drop high correlated features RUN #####################
df_ttt = df_ttt.drop(columns_to_drop_list, axis=1)

# %%
df_ttt.select_dtypes(include=[np.number])

# %%
df_ttt['is_shopping'] = (df_ttt['merchant_id']=='N101065').astype(int)

# %%
numeric_list = df.select_dtypes(include=[np.number]).columns[(df.select_dtypes(include=[np.number]) < 0).any()].tolist()

# %%
df = classifier_column(
    df=df
    , original_column_name='unpaid_at_60_rate'
    , condition_value=0.1
    , new_column_name='is_default'
    , is_bigger_condition=True
    , drop_the_original=False)



# %% split the data
X_train, X_validation, X_test, y_train, y_validation, y_test = train_val_test_split_date(
    df = df, target = target_col, sorty_by = 'decision_time', train_share = 70, val_share = 15)

# %%
list_1_test = np.array(['decision_time', 'is_shopping', 'unpaid_at_60_rate'])
# %%
list_1_test = df.columns

# %%
list_2_test = np.array(['is_shopping', 'unpaid_at_60_rate'])

# %%
bool_list = np.array(~np.isin(list_1_test, list_2_test), dtype=bool)

# %%
list_1_test[bool_list]

# %%
df[list_1_test[bool_list]]

# %%
df_exp = 

# %%
exeptions_list = np.array(exeptions_list)
columns_list = df.columns
bool_filter_list = np.array(np.isin(columns_list, exeptions_list), dtype=bool)
bool_keep_list = np.array(~np.isin(columns_list, exeptions_list), dtype=bool)
df_without_exp = df[bool_filter_list]

# %%
def tree_model_train(model_type:str, model_name:str, constant_params, categorical_variables, X_train, y_train, X_validation, y_validation, months:int, year:str, dum:str = ''):
# %%
import joblib
import numpy as np

# %%# Convert the categorical_variables to np.array
categorical_variables = np.array(cat_var_list)
# %%# Get the total columns array
columns_list = X_train.columns
# %%# Get boolean array for keeping the variables
bool_filter_list = np.array(np.isin(columns_list, categorical_variables), dtype=bool)
# %%# Get the column list after filtering
filtered_col_list = list(columns_list[bool_filter_list])
# %%
from catboost import CatBoostClassifier
# %%
model = CatBoostClassifier(**constant_params_catboost)
# %%
#fit the model
model.fit(X_train, y_train,cat_features=filtered_col_list,eval_set=(X_validation, y_validation),plot=True)
# %%
#save the model
joblib.dump(model, 'model_'+str(months)+'_months_'+year+'_'+dum+'_'+model_name)

# %%
# %%
#fill NA with 'other' for categorical variables
for col in na_col_list:
    df[col] = df[col].fillna(value='other')

# %%
## Correlation selection function
def cor_selection(df,target_col, exeptions_list):
    # Take out exeption columns from the df before removing the hige corr columns
    
    ## Convert the expetions_list to np.array
    exeptions_list = np.array(exeptions_list)
    ## Get the total columns array
    columns_list = df.columns
    ## Get boolean array for either to keep the variables or to filter them out
    bool_filter_list = np.array(np.isin(columns_list, exeptions_list), dtype=bool)
    ## Get the column list after filtering
    filtered_col_list = columns_list[bool_filter_list]
    ## Create the df without the exeptions
    df_without_exp = df[filtered_col_list]

    # Create correlation matrix
    corr_matrix = df_without_exp.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Create empty list of columns to drop
    columns_to_drop_list = []
    # 
    for col in upper.columns.to_list():
        # For each column, get the columns with high correlation. Then create a df with all the columns, together with the target column
        corr_columns_list = upper[upper[col] > 0.95][col].index.to_list()
        df_for_f_regression = pd.concat([df[col], df[corr_columns_list], df[target_col]], axis=1).dropna()
        if len(corr_columns_list) > 0:
            # Get the target and features after filtering NA's
            target = df_for_f_regression.iloc[: , len(df_for_f_regression.columns)-1:]
            features = df_for_f_regression.iloc[: , 0:len(df_for_f_regression.columns)-1]
            # Get the f-values of the test
            f_regression_result,_ = f_regression(features, target)
            # Get the feature location of the best score
            the_best_feature = f_regression_result.argmax()
            # Get the feature name of the best score
            the_best_feature_name = features.columns[the_best_feature]
            # Get a list of columns to drop
            columns_to_drop = features.loc[:, ~features.columns.isin([the_best_feature_name])].columns.to_list()
            
            # Add the columns to drop to the aggregated list of columns to drop
            columns_to_drop_list.extend(columns_to_drop)
            # Remove duplictes and sort it
            columns_to_drop_list = sorted(set(columns_to_drop_list))
            
        # else:
        #     columns_to_drop = []

    # Return
    return columns_to_drop_list

# %%
## Convert the expetions_list to np.array
exeptions_list = np.array(exeptions_list)

# %%
## Get the total columns array
columns_list = df.columns
# %%# Get boolean array for either to keep the variables or to filter them out
bool_filter_list = np.array(~np.isin(columns_list, exeptions_list), dtype=bool)
# %%# Get the column list after filtering
filtered_col_list = columns_list[bool_filter_list]
# %%# Create the df without the exeptions
df_without_exp = df[filtered_col_list]

# %%
# Function to remove variables with high share of NA's
def remove_full_na(df, na_share_threshold, exeptions):
    # run na_list_fnc and get the share of NA's per column in a dataframe
    na_df = loss_model.na_list_fnc(df=df)
    if na_df['variable'].isin(exeptions).any():
        na_df = na_df[~na_df['variable'].isin(exeptions)]
    else:
        na_df = na_df
    # get the full NA columns
    full_na = na_df[na_df['na_share']>=na_share_threshold]['variable']
    # drop the full NA columns from the original df
    df = df.drop(columns=full_na)
    return df

# %%
# run na_list_fnc and get the share of NA's per column in a dataframe
na_df = na_list_fnc(df=df)

# %%
df_copy['unpaid_at_60_rate'] = df_copy['unpaid_at_60'] / df_copy['captured_amount']
# %%
probs_df = pd.DataFrame({'prob': probs}, columns=['prob'], index=y_test.index.to_list())
# %%
probs_df.join(df_copy['unpaid_at_60_rate'], how='left')



# %%
test_list = ['a', 'b', 'c']

# %%
