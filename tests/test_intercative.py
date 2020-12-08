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
loss_model.custom_summary(df=df_ttt,col='payment_method_card_level')
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