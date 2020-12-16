import numpy as np

# The path to load the data
path = '/home/ron.snir/git/loss_model/data/pi4_full000.csv'
path_browser_new = '/home/ron.snir/git/loss_model/data/pi4_browser_new000.csv'
path_browser_returning = '/home/ron.snir/git/loss_model/data/pi4_browser_returning000.csv'

# Columns to convert to datetime
date_columns_list = [
    'usemailageemailiprisk_lastflaggedon'
    , 'decision_time'
]

# The model type and name
# tree_model_type = 'classification'
tree_model_type = 'regression'
tree_model_name = 'catboost'

# Catbood parameters
constant_params_catboost = {'iterations': 1000,
                    'random_seed': 101,
                    'learning_rate': 0.1,
                    'eval_metric': 'RMSE',
                    'early_stopping_rounds': 20}

# Ther target column
target_col_class = 'is_default'
target_col_reg = 'unpaid_at_60_rate'

# # Segments filters
# browser_new_filter = (df_pred_features_pred['is_shopping']=='1') & (df_pred_features_pred['rvl_cid_has_paid']==False)
# browser_returning_filter = (df_pred_features_pred['is_shopping']=='1') & (df_pred_features_pred['rvl_cid_has_paid']==True)
# merchant_new = (df_pred_features_pred['is_shopping']=='0') & (df_pred_features_pred['rvl_cid_has_paid']==False)
# merchant_returning_filter = (df_pred_features_pred['is_shopping']=='0') & (df_pred_features_pred['rvl_cid_has_paid']==True)

# The share of NA's that above will be removed
na_share_threshold = 95

# The columns that we don't want to drop, even if they have high share of NA's
exeptions_list = [
    'nr_errands'
    , 'unpaid_at_60_rate'
    # , 'is_default'
]

# Columns which NA's will be converted to 'UNKNOWN'
na_to_unknown_list = [
    'payment_method_card_level'
    , 'psp_credit_card_issuing_bank'
]

# Columns which NA's will be converted to 'Other'
na_to_other_list = [
    'usemailageemailiprisk_source_industry'
]

# Number of lines to skip before taking the line to the test df
line_step = 10

# List of columns to convert to datetime
date_col_list = [
    'usemailageemailiprisk_lastflaggedon'
    , 'decision_time'
    , 'consumer_date_of_birth'
    , 'usemailageemailiprisk_domain_age'
    , 'usemailageemailiprisk_first_verification_date'
    , 'usemailageemailiprisk_last_verification_date'
]

# List of columns to calculate their days diff from another date column
days_diff_list = [
    'usemailageemailiprisk_lastflaggedon'
    , 'usemailageemailiprisk_domain_age'
    , 'usemailageemailiprisk_first_verification_date'
    , 'usemailageemailiprisk_last_verification_date'
]

# List of columns to calculate their years diff from another date column
years_diff_list = [
    'consumer_date_of_birth'
]

# List of columns to drop (unsuse)
to_drop_list_beginning = [
    'usemailageemailiprisk_email_age'                   # date - unique 18059
    , 'usemailageemailiprisk_location'                  # unique 11768
    , 'usemailageemailiprisk_phoneowner'                # name - unique 76280
    , 'rvl_cid_account_paid_minus_pay_to_terms_0_3m'    # only -9999
    , 'rvl_cid_account_paid_div_by_pay_to_terms_0_12m'  # only -9999
    , 'rvl_cid_account_paid_minus_min_to_pay_0_3m'      # only -9999
    , 'rvl_cid_ts_2nd_last_rej_0_14d'                   # mostly -9999
    , 'rvl_cid_ts_1st_last_rej_0_14d'                   # mostly -9999
    , 'rvl_cid_has_paid_elv_24m_int'                    # only 0
    , 'rvl_cid_has_paid_elv_12m_int'                    # only 0
    , 'rvl_cid_account_paid_div_by_min_to_pay_0_3m'     # only -9999
    , 'rvl_cid_account_paid_div_by_pay_to_terms_0_6m'   # only -9999
    , 'rvl_cid_account_paid_div_by_pay_to_terms_0_3m'   # only -9999
    , 'rvl_cid_account_paid_minus_pay_to_terms_0_6m'    # only -9999
    , 'rvl_cid_has_paid_elv_int'                        # only 0
    , 'rvl_cid_account_paid_div_by_min_to_pay_0_6m'     # only -9999
    , 'rvl_cid_account_paid_minus_min_to_pay_0_6m'      # only -9999
    , 'rvl_cid_account_paid_minus_pay_to_terms_0_12m'   # only -9999
    , 'rvl_cid_worst_status_active_sibc'                # only 0
    , 'rvl_cid_has_converted_from_sibc'                 # mostly False
    , 'rvl_cid_account_paid_div_by_min_to_pay_0_12m'    # only -9999
    , 'rvl_cid_account_paid_minus_min_to_pay_0_12m'     # only -9999
    , 'rvl_cid_days_since_last_payment_inv'             # mostly NA
    , 'usemailageemailiprisk_lang'                      # only en-US
    , 'usemailageemailiprisk_domain_exists'             # only Yes
    , 'usemailageemailiprisk_query_type'                # only EmailIPRisk
    , 'velocityvariables_is_found'                      # only True
    , 'tu_identified_on_ofac_list'                      # only clear
    , 'tu_is_valid_lookup'                              # only True
    , 'rvl_cid_recovery_debt_36m'                       # only undefined
    , 'payment_method_card_identifier'                  # unique 122028
    , 'payment_method_credit_card_connector'            # only vantiv
    , 'payment_method_credit_card_client_reference'     # unique 134517
]

# List of irrelevant columns
to_drop_list_irrelevant = [
    'usemailageemailiprisk_ip_longitude'
    , 'usemailageemailiprisk_ip_latitude'
    , 'usemailageemailiprisk_iptimezone'
    , 'order_id'
    , 'capture_id'
    , 'order_date'
    , 'capture_date'
    , 'usemailageemailiprisk_ip_map'
    , 'usemailageemailiprisk_created'
    , 'usemailageemailiprisk_ip_anonymousdetected'
    , 'usemailageemailiprisk_ip_net_speed_cell'
    , 'usemailageemailiprisk_userdefinedrecordid'
    , 'rvl_eid_estore_group'                        # there is category variable - more granular
    , 'brms_email_domain_shipping'                  # there is similar variable - usemailageemailiprisk_domainname
    , 'payment_method_card_issuing_country'         # either USA or US
    , 'payment_method_card_card_number'
    , 'psp_credit_card_brand'                       # there is similar variable - psp_credit_card_brand
    , 'md_gdpr_access_control_date'
    , 'md_update_timestamp'
    , 'brms_supplied_year_email'                    # starnge values (ex. 1994.0, 201914.0)
    # , 'usemailageemailiprisk_ip_postalcode'
    , 'transparent_2020_score'
    , 'transparent_2019_score'
    , 'abuse_2019_v2_score'
    , 'abuse_2020_v3_score'
    , 'brms_internal_score'
]

# List of columns to drop in the end
to_drop_list_end = [
    'merchant_id'
]

# List of columns to drop (unsuse)
to_drop_list_na = [
    'abuse_2019_v1_score'                                           # Only NA's
    , 'rvl_cid_count_open_authorizations'                           # Only NA's
    , 'rvl_cid_max_num_failed_payment_attempts'                     # Only NA's
    , 'rvl_cid_settled_amount_14d'                                  # Only NA's
    , 'rvl_cid_new_exposure_3h'                                     # Only NA's
    , 'idology_summary_result'                                      # Only NA's
    , 'rvl_cid_new_exposure_7d'                                     # Only NA's
    , 'rvl_cid_new_exposure_1d'                                     # Only NA's
    , 'usemailageemailiprisk_phoneownertype'                        # Only NA's
    , 'rvl_cid_new_exposure_14d'                                    # Only NA's
    , 'rvl_cid_settled_amount_3h'                                   # Only NA's
    , 'rvl_cid_max_num_failed_payment_attempts_all_authorizations'  # Only NA's
    , 'rvl_cid_settled_amount_1d'                                   # Only NA's
    , 'rvl_cid_settled_amount_7d'                                   # Only NA's
    , 'acquiring_channel'                                           # Mostly NA's
    , 'velocityvariables_apm_card_amount_on_email_merchant_id_1m'   # Mostly NA's
    , 'velocityvariables_cards_on_shipping_email_merchant_id_1d'    # Mostly NA's
    , 'velocityvariables_apm_amount_on_card_merchant_id_7d'         # Mostly NA's
    , 'velocityvariables_cards_on_shipping_email_merchant_id_7d'    # Mostly NA's
    , 'velocityvariables_cards_on_shipping_email_merchant_id_1m'    # Mostly NA's
    , 'velocityvariables_apm_card_amount_on_email_merchant_id_1d'   # Mostly NA's
    , 'velocityvariables_apm_card_amount_on_email_merchant_id_1h'   # Mostly NA's
    , 'velocityvariables_apm_amount_on_card_merchant_id_1h'         # Mostly NA's
    , 'velocityvariables_apm_amount_on_card_merchant_id_1d'         # Mostly NA's
    , 'velocityvariables_apm_card_amount_on_email_merchant_id_7d'   # Mostly NA's
    , 'velocityvariables_cards_on_shipping_email_merchant_id_1h'    # Mostly NA's
    , 'velocityvariables_apm_amount_on_card_merchant_id_1m'         # Mostly NA's
    , 'tu_rt20s'                                                    # Mostly NA's
    , 'nr_card_changes'                                             # Mostly NA's
    , 'nr_card_changes_per_order'                                   # Mostly NA's
    , 'usemailageemailiprisk_fraud_type'                            # Mostly NA's
]

# Replace negative with 0
neg_to_zero_col_list = [
    'rvl_cid_num_paid_sibc_24m'
    , 'rvl_cid_num_paid_sibc_12m'
    , 'rvl_cid_num_invoice_0_24m'
    , 'rvl_cid_num_unpaid_orders'
    , 'crid_realtime_num_unpaid_sibc'
    , 'rvl_cid_num_unpaid_sibc'
    , 'rvl_cid_num_unpaid_sibc_orders'
    , 'rvl_eid_estore_id_fraud_rate_30d'
    , 'rvl_eid_estore_id_fraud_cnt_30d'
]

# Dictionary to replace negative values with 0
neg_to_zero_dict = {
    -9999:0
    , -9998:0
    , -9997:0
    , -1:0
    , -2:0
    , -3:0
    , -4:0
    , -5:0
    , -6:0
    , -7:0
    , -8:0
    , -9:0
}

# Replace negative with None
neg_to_none_col_list = [
    'rvl_cid_days_since_last_payment_inv'
    , 'rvl_cid_days_since_last_payment_credit'
    , 'rvl_cid_days_since_last_payment_acct'
    , 'rvl_eid_estore_id_avg_ts_0_12m'
]

# Dictionary to replace negative values (numeric) with None
neg_to_none_dict = {
    -9999:np.nan
    , -9998:np.nan
    , -9997:np.nan
    , -1:np.nan
    , -2:np.nan
    , -3:np.nan
    , -4:np.nan
    , -5:np.nan
    , -6:np.nan
    , -7:np.nan
    , -8:np.nan
    , -9:np.nan
}

# Replace negative with None
neg_to_none_col_string_list = [
    'rvl_cid_payment_method_2nd_last_purchase_0_3m'
    , 'rvl_cid_payment_method_last_purchase_0_3m'
]

# Dictionary to replace negative values (strings) with None
neg_to_none_string_dict = {
    '-9999':np.nan
    , '-9998':np.nan
    , '-9997':np.nan
    , '-1':np.nan
    , '-2':np.nan
    , '-3':np.nan
    , '-4':np.nan
    , '-5':np.nan
    , '-6':np.nan
    , '-7':np.nan
    , '-8':np.nan
    , '-9':np.nan
}

# Columns to check
to_check_col_list = [
    'crid_realtime_incoming_sibc_debt'
    , 'crid_realtime_incoming_invoice_debt'
    , 'rvl_cid_incoming_sibc_debt'
]

# variables for age calculation
today_col = 'decision_time'

# Categorical variables
cat_var_list = [
    'psp_credit_card_brand'
    , 'psp_credit_card_issuing_bank'
    , 'acquiring_source'
    , 'usemailageemailiprisk_ipdomain'
    , 'usemailageemailiprisk_ip_continent_code'
    , 'usemailageemailiprisk_ipasnum'
    , 'usemailageemailiprisk_custphone_inbillingloc'
    , 'usemailageemailiprisk_ip_reputation'
    , 'usemailageemailiprisk_ip_org'
    , 'usemailageemailiprisk_ip_country_code'
    , 'usemailageemailiprisk_phone_status'
    , 'usemailageemailiprisk_phoneownermatch'
    , 'usemailageemailiprisk_shipcitypostalmatch'
    , 'usemailageemailiprisk_status'
    , 'usemailageemailiprisk_fraud_risk'
    , 'usemailageemailiprisk__e_a_advice'
    , 'usemailageemailiprisk_domaincompany'
    # , 'usemailageemailiprisk_ip_postalcode'
    , 'usemailageemailiprisk_phonecarriername'
    , 'usemailageemailiprisk_ip_risklevel'
    , 'usemailageemailiprisk__e_a_risk_band'
    , 'usemailageemailiprisk_country'
    , 'usemailageemailiprisk_citypostalmatch'
    , 'usemailageemailiprisk_ipcountrymatch'
    , 'usemailageemailiprisk_namematch'
    , 'usemailageemailiprisk_domainrelevantinfo'
    , 'usemailageemailiprisk_domainrisklevel'
    , 'usemailageemailiprisk_domainname'
    , 'usemailageemailiprisk_domaincountrymatch'
    , 'usemailageemailiprisk_ip_region'
    , 'usemailageemailiprisk_shipforward'
    , 'usemailageemailiprisk_ip_city'
    , 'usemailageemailiprisk__e_a_reason'
    , 'usemailageemailiprisk_domaincategory'
    , 'usemailageemailiprisk_ip_riskreason'
    , 'usemailageemailiprisk_ip_user_type'
    , 'usemailageemailiprisk_domaincountryname'
    , 'usemailageemailiprisk_ip_isp'
    , 'usemailageemailiprisk_domaincorporate'
    , 'usemailageemailiprisk_source_industry'
    , 'usemailageemailiprisk_email_exists'
    , 'usemailageemailiprisk_ip_corporate_proxy'
    , 'usemailageemailiprisk_retrievedfromcache'
    , 'usemailageemailiprisk_phonecarriertype'
    , 'rvl_eid_estore_category'
    # , 'rvl_eid_estore_group'
    , 'brms_device'
    , 'tu_identified_as_fraud_alert'
    , 'tu_militarty_active_duty'
    , 'tu_retrievedfromcache'
    , 'tu_not_enough_info'
    , 'tu_freeze'
    , 'rvl_cid_has_paid_credit'
    , 'rvl_cid_has_paid_24m'
    , 'rvl_cid_has_paid_inv'
    , 'rvl_cid_has_paid_12m'
    , 'rvl_cid_has_paid_credit_24m'
    , 'rvl_cid_has_paid_credit_12m'
    , 'rvl_cid_has_paid_inv_12m'
    , 'rvl_cid_has_paid_inv_24m'
    , 'rvl_cid_payment_method_2nd_last_purchase_0_3m'
    , 'rvl_cid_has_paid_acct_24m'
    , 'rvl_cid_has_paid_acct_12m'
    , 'rvl_cid_has_paid_acct'
    , 'payment_method_card_type'
    , 'payment_method_card_level'
    , 'payment_method_card_issuing_bank'
    , 'is_paused'
    , 'rvl_cid_payment_method_last_purchase_0_3m'
    , 'payment_method_card_brand'

    , 'usemailageemailiprisk_ip_callingcode'
    , 'usemailageemailiprisk__e_a_advice_i_d'
    , 'usemailageemailiprisk_ip_postalconf'
    , 'usemailageemailiprisk_ip_cityconf'
    , 'usemailageemailiprisk__e_a_status_i_d'
    , 'usemailageemailiprisk_ip_riskreasonid'
    , 'usemailageemailiprisk_ip_countryconf'
    , 'usemailageemailiprisk_domainrisklevel_i_d'
    , 'usemailageemailiprisk_domainrelevantinfo_i_d'
    , 'usemailageemailiprisk__e_a_reason_i_d'
    , 'usemailageemailiprisk__e_a_risk_band_i_d'
    , 'usemailageemailiprisk_ip_regionconf'
    , 'usemailageemailiprisk_ip_metro_code'
    , 'brms_day_of_week'
    , 'tu_most_negative_influence_on_vantagescore_factor1'
    , 'tu_most_negative_influence_on_vantagescore_factor2'
    , 'tu_most_negative_influence_on_vantagescore_factor3'
    , 'tu_most_negative_influence_on_vantagescore_factor4'
    , 'tu_vtg4_most_negative_influence_factor1'
    , 'tu_vtg4_most_negative_influence_factor2'
    , 'tu_vtg4_most_negative_influence_factor3'
    , 'tu_vtg4_most_negative_influence_factor4'
    , 'rvl_cid_account_worst_pstatus_0_12m'
    , 'rvl_cid_account_worst_pstatus_3_6m' #check
    , 'rvl_cid_has_paid_inv_int' #check
    , 'rvl_cid_pstatus_2nd_last_archived_0_3m' #check
    , 'rvl_cid_pstatus_max_archived_0_12_months' #check
    , 'rvl_cid_has_paid_credit_int' #check
    , 'rvl_cid_pstatus_max_archived_0_24_months' #check
    , 'rvl_cid_pstatus_3rd_last_archived_0_24m' #check
    , 'rvl_cid_pstatus_last_archived_0_12m'
    , 'rvl_cid_has_paid_credit_12m_int'
    , 'rvl_cid_pstatus_last_archived_0_24m'
    , 'rvl_cid_has_paid_12m_int'
    , 'rvl_cid_pstatus_2nd_last_archived_0_12m'
    , 'rvl_cid_pstatus_max_archived_0_6_months'
    , 'rvl_cid_pstatus_3rd_last_archived_0_12m'
    , 'rvl_cid_has_rejection_14d_int'
    , 'rvl_cid_account_incoming_pstatus_3m'
    , 'rvl_cid_worst_pstatus_active_inv'
    , 'rvl_cid_account_worst_pstatus_0_3m'
    , 'rvl_cid_has_paid_inv_12m_int'
    , 'rvl_cid_oldest_pstatus_active_inv'
    , 'rvl_cid_account_worst_pstatus_3_12m'
    , 'rvl_cid_pstatus_3rd_last_archived_0_6m'
    , 'rvl_cid_pstatus_last_archived_0_3m'
    , 'rvl_cid_has_paid_inv_24m_int'
    , 'rvl_cid_pstatus_last_archived_0_6m'
    , 'rvl_cid_pstatus_3rd_last_archived_0_3m'
    , 'rvl_cid_email_has_paid_int'
    , 'rvl_cid_zip_has_paid_int'
    , 'payment_method_card_exp_month'
    , 'is_shopping'
]

# Categorical columns with numbers that should be converted to integer
num_to_int_col_list = [
    'usemailageemailiprisk_ip_callingcode'
    , 'usemailageemailiprisk_ip_metro_code'
    , 'tu_most_negative_influence_on_vantagescore_factor1'
    , 'tu_most_negative_influence_on_vantagescore_factor2'
    , 'tu_most_negative_influence_on_vantagescore_factor3'
    , 'tu_most_negative_influence_on_vantagescore_factor4'
    , 'tu_vtg4_most_negative_influence_factor1'
    , 'tu_vtg4_most_negative_influence_factor2'
    , 'tu_vtg4_most_negative_influence_factor3'
    , 'tu_vtg4_most_negative_influence_factor4'
]

# Categorical columns with numbers that should be converted to integer
num_to_int_col_list = [
    'usemailageemailiprisk_ip_callingcode'
    , 'usemailageemailiprisk__e_a_advice_i_d'
    , 'usemailageemailiprisk_ip_postalconf'
    , 'usemailageemailiprisk_ip_cityconf'
    , 'usemailageemailiprisk__e_a_status_i_d'
    , 'usemailageemailiprisk_ip_riskreasonid'
    , 'usemailageemailiprisk_ip_countryconf'
    , 'usemailageemailiprisk_domainrisklevel_i_d'
    , 'usemailageemailiprisk_domainrelevantinfo_i_d'
    , 'usemailageemailiprisk__e_a_reason_i_d'
    , 'usemailageemailiprisk__e_a_risk_band_i_d'
    , 'usemailageemailiprisk_ip_regionconf'
    , 'usemailageemailiprisk_ip_metro_code'
    , 'brms_day_of_week'
    , 'tu_most_negative_influence_on_vantagescore_factor1'
    , 'tu_most_negative_influence_on_vantagescore_factor2'
    , 'tu_most_negative_influence_on_vantagescore_factor3'
    , 'tu_most_negative_influence_on_vantagescore_factor4'
    , 'tu_vtg4_most_negative_influence_factor1'
    , 'tu_vtg4_most_negative_influence_factor2'
    , 'tu_vtg4_most_negative_influence_factor3'
    , 'tu_vtg4_most_negative_influence_factor4'
    , 'rvl_cid_account_worst_pstatus_0_12m'
    , 'rvl_cid_account_worst_pstatus_3_6m' #check
    , 'rvl_cid_has_paid_inv_int' #check
    , 'rvl_cid_pstatus_2nd_last_archived_0_3m' #check
    , 'rvl_cid_pstatus_max_archived_0_12_months' #check
    , 'rvl_cid_has_paid_credit_int' #check
    , 'rvl_cid_pstatus_max_archived_0_24_months' #check
    , 'rvl_cid_pstatus_3rd_last_archived_0_24m' #check
    , 'rvl_cid_pstatus_last_archived_0_12m'
    , 'rvl_cid_has_paid_credit_12m_int'
    , 'rvl_cid_pstatus_last_archived_0_24m'
    , 'rvl_cid_has_paid_12m_int'
    , 'rvl_cid_pstatus_2nd_last_archived_0_12m'
    , 'rvl_cid_pstatus_max_archived_0_6_months'
    , 'rvl_cid_pstatus_3rd_last_archived_0_12m'
    , 'rvl_cid_has_rejection_14d_int'
    , 'rvl_cid_account_incoming_pstatus_3m'
    , 'rvl_cid_worst_pstatus_active_inv'
    , 'rvl_cid_account_worst_pstatus_0_3m'
    , 'rvl_cid_has_paid_inv_12m_int'
    , 'rvl_cid_oldest_pstatus_active_inv'
    , 'rvl_cid_account_worst_pstatus_3_12m'
    , 'rvl_cid_pstatus_3rd_last_archived_0_6m'
    , 'rvl_cid_pstatus_last_archived_0_3m'
    , 'rvl_cid_has_paid_inv_24m_int'
    , 'rvl_cid_pstatus_last_archived_0_6m'
    , 'rvl_cid_pstatus_3rd_last_archived_0_3m'
    , 'rvl_cid_email_has_paid_int'
    , 'rvl_cid_zip_has_paid_int'
    , 'payment_method_card_exp_month'
    , 'is_shopping'
    # , 'usemailageemailiprisk_ip_postalcode'
]


#
final_col_list = ['captured_amount',
 'nr_of_captures',
 'total_tax_amount',
 'psp_credit_card_issuing_bank',
 'acquiring_source',
 'is_paused',
 'nr_errands',
 'decision_time',
 'usemailageemailiprisk_ipdomain',
 'usemailageemailiprisk_ip_continent_code',
 'usemailageemailiprisk_ipasnum',
 'usemailageemailiprisk_custphone_inbillingloc',
 'usemailageemailiprisk_ip_reputation',
 'usemailageemailiprisk_ip_org',
 'usemailageemailiprisk_count',
 'usemailageemailiprisk_ip_callingcode',
 'usemailageemailiprisk_ip_country_code',
 'usemailageemailiprisk__e_a_advice_i_d',
 'usemailageemailiprisk_phone_status',
 'usemailageemailiprisk_phoneownermatch',
 'usemailageemailiprisk_totalhits',
 'usemailageemailiprisk_ip_postalconf',
 'usemailageemailiprisk_shipcitypostalmatch',
 'usemailageemailiprisk_status',
 'usemailageemailiprisk_ip_risklevelid',
 'usemailageemailiprisk_fraud_risk',
 'usemailageemailiprisk_response_count',
 'usemailageemailiprisk_ip_cityconf',
 'usemailageemailiprisk__e_a_advice',
 'usemailageemailiprisk_domaincompany',
 'usemailageemailiprisk__e_a_status_i_d',
#  'usemailageemailiprisk_ip_postalcode',
 'usemailageemailiprisk_phonecarriername',
 'usemailageemailiprisk_ip_risklevel',
 'usemailageemailiprisk__e_a_risk_band',
 'usemailageemailiprisk_country',
 'usemailageemailiprisk_uniquehits',
 'usemailageemailiprisk_ip_riskreasonid',
 'usemailageemailiprisk_citypostalmatch',
 'usemailageemailiprisk_ip_countryconf',
 'usemailageemailiprisk_ipcountrymatch',
 'usemailageemailiprisk_namematch',
 'usemailageemailiprisk_domainrelevantinfo',
 'usemailageemailiprisk_domainrisklevel',
 'usemailageemailiprisk_domainname',
 'usemailageemailiprisk_domaincountrymatch',
 'usemailageemailiprisk_ip_region',
 'usemailageemailiprisk_shipforward',
 'usemailageemailiprisk_ip_city',
 'usemailageemailiprisk__e_a_reason',
 'usemailageemailiprisk_domainrisklevel_i_d',
 'usemailageemailiprisk_domaincategory',
 'usemailageemailiprisk_ip_riskreason',
 'usemailageemailiprisk_ip_user_type',
 'usemailageemailiprisk_lookup_timestamp_ms_since_epoch',
 'usemailageemailiprisk_domainrelevantinfo_i_d',
 'usemailageemailiprisk__e_a_score',
 'usemailageemailiprisk_domaincountryname',
 'usemailageemailiprisk__e_a_reason_i_d',
 'usemailageemailiprisk_ip_isp',
 'usemailageemailiprisk_ipdistancemil',
 'usemailageemailiprisk__e_a_risk_band_i_d',
 'usemailageemailiprisk_ip_regionconf',
 'usemailageemailiprisk_domaincorporate',
 'usemailageemailiprisk_ip_metro_code',
 'usemailageemailiprisk_source_industry',
 'usemailageemailiprisk_email_exists',
 'usemailageemailiprisk_ip_corporate_proxy',
 'usemailageemailiprisk_retrievedfromcache',
 'usemailageemailiprisk_phonecarriertype',
 'usemailageemailiprisk_smfriends',
 'rvl_eid_estore_category',
 'rvl_eid_estore_id_avg_ts_0_12m',
 'rvl_eid_estore_id_fraud_rate_30d',
 'rvl_eid_estore_id_fraud_cnt_30d',
 'rvl_eid_estore_group_bad_rate_12m',
 'rvl_eid_estore_category_bad_rate_12m',
 'rvl_eid_estore_id_avg_age_0_12m',
 'rvl_eid_estore_id_tot_tx_0_12m',
 'velocityvariables_apm_cids_on_shipping_email_1m',
 'velocityvariables_approved_apm_order_ids_on_cid_14d',
 'velocityvariables_apm_card_cids_on_ip_zip_1d',
 'velocityvariables_shipping_full_names_on_account_id_14d',
 'velocityvariables_shipping_street_zip_on_order_id_14d',
 'velocityvariables_acp_on_cid_shipping_zip_1d',
 'velocityvariables_apm_pad_cids_on_ip_zip_1h',
 'velocityvariables_apm_on_billing_email_2d',
 'velocityvariables_card_countries_on_shipping_zip_1d',
 'velocityvariables_cids_on_shipping_email_2d',
 'velocityvariables_apm_order_ids_on_shipping_street_zip_2d',
 'velocityvariables_cards_on_shipping_email_1m',
 'velocityvariables_cids_on_shipping_email_1h',
 'velocityvariables_apm_card_cids_on_ip_shipping_zip_2d',
 'velocityvariables_apm_card_cids_on_ip_shipping_zip_1h',
 'velocityvariables_billing_emails_on_billing_phone_2d',
 'velocityvariables_apm_order_ids_on_billing_email_cid_2d',
 'velocityvariables_billing_emails_on_billing_phone_1h',
 'velocityvariables_card_numbers_on_cid_1h',
 'velocityvariables_apm_billing_emails_on_billing_street_zip_ip_2d',
 'velocityvariables_apm_order_ids_on_billing_street_zip_1h',
 'velocityvariables_shipping_phones_on_account_id_14d',
 'velocityvariables_card_countries_on_shipping_zip_3d',
 'velocityvariables_gpm_on_order_id_1d',
 'velocityvariables_apm_order_ids_on_billing_street_zip_2d',
 'velocityvariables_apm_cids_on_card_7d',
 'velocityvariables_apm_pad_cids_on_ip_zip_1d',
 'velocityvariables_apm_on_billing_street_zip_ip_2d',
 'velocityvariables_acp_on_shipping_email_zip_1h',
 'velocityvariables_apm_cids_on_card_1d',
 'velocityvariables_apm_cids_on_card_1m',
 'velocityvariables_card_countries_on_shipping_zip_14d',
 'velocityvariables_apm_on_billing_phone_1h',
 'velocityvariables_apm_billing_emails_on_billing_street_zip_ip_1h',
 'velocityvariables_apm_emails_on_order_id_7d',
 'velocityvariables_is_error',
 'velocityvariables_account_ids_on_cid_14d',
 'velocityvariables_acp_on_ip_shipping_zip_1h',
 'velocityvariables_billing_emails_on_card_1h',
 'velocityvariables_apm_on_shipping_street_zip_ip_2d',
 'velocityvariables_apm_card_cids_on_ip_shipping_zip_1d',
 'velocityvariables_card_numbers_on_cid_1d',
 'velocityvariables_acp_on_ip_shipping_zip_7d',
 'velocityvariables_acp_on_shipping_email_zip_7d',
 'velocityvariables_billing_emails_on_billing_phone_1m',
 'velocityvariables_billing_phones_on_account_id_14d',
 'velocityvariables_billing_emails_on_card_2d',
 'velocityvariables_cids_on_shipping_street_zip_ip_2d',
 'velocityvariables_cids_on_ip_1h',
 'velocityvariables_apm_cids_on_card_number_exp_year_exp_month_7d',
 'velocityvariables_account_ids_on_shipping_zip_14d',
 'velocityvariables_cids_on_shipping_street_zip_ip_1h',
 'velocityvariables_billing_full_names_on_account_id_14d',
 'velocityvariables_apm_card_cids_on_ip_zip_2d',
 'velocityvariables_is_disabled',
 'velocityvariables_apm_pad_cids_on_ip_1d',
 'velocityvariables_apm_cids_on_card_identifier_7d',
 'velocityvariables_shipping_full_name_on_order_id_1h',
 'velocityvariables_apm_card_cids_on_ip_zip_1h',
 'velocityvariables_acp_on_cid_shipping_zip_1h',
 'velocityvariables_order_ids_on_card_1h',
 'velocityvariables_order_ids_on_card_2d',
 'velocityvariables_account_ids_on_billing_zip_14d',
 'velocityvariables_acp_on_ip_shipping_zip_1d',
 'velocityvariables_apm_billing_emails_on_billing_street_zip_ip_1m',
 'velocityvariables_apm_emails_on_cid_2d',
 'velocityvariables_apm_cids_on_order_id_7d',
 'velocityvariables_acp_on_shipping_email_zip_1d',
 'velocityvariables_acp_on_cid_shipping_zip_7d',
 'velocityvariables_billing_emails_on_card_7d',
 'velocityvariables_billing_street_zip_on_order_id_1h',
 'velocityvariables_apm_order_ids_on_ip_1h',
 'velocityvariables_order_ids_on_card_7d',
 'velocityvariables_apm_order_ids_on_shipping_email_cid_1h',
 'velocityvariables_apm_order_id_7d',
 'velocityvariables_apm_card_order_id_7d',
 'velocityvariables_apm_cids_on_card_number_exp_year_exp_month_1m',
 'velocityvariables_billing_full_name_on_order_id_1h',
 'velocityvariables_approved_apm_order_ids_on_cid_1m',
 'velocityvariables_apm_cids_on_card_number_exp_year_exp_month_1d',
 'velocityvariables_cards_on_shipping_email_test4_7d',
 'velocityvariables_approved_apm_order_ids_on_cid_1d',
 'velocityvariables_apm_cids_on_card_identifier_1m',
 'velocityvariables_apm_cids_on_card_identifier_1d',
 'velocityvariables_cids_on_shipping_street_zip_ip_1m',
 'velocityvariables_apm_cids_on_shipping_email_7d',
 'velocityvariables_apm_country_zip_street_2d',
 'velocityvariables_apm_country_zip_street_1h',
 'velocityvariables_apm_pad_cids_on_ip_zip_2d',
 'brms_number_of_articles',
 'brms_total_gift_card_count',
 'brms_is_returning_sibc',
 'brms_day_of_week',
 'brms_sum_paid',
 'brms_seconds_since_midnight',
 'brms_cheapest_article_price',
 'brms_is_po_box',
 'brms_number_of_distinct_articles',
 'brms_max_same_article_price',
 'brms_max_same_article_count',
 'brms_contains_electronic_gift_card',
 'brms_contains_expshipping',
 'brms_contains_gift_card',
 'brms_device',
 'brms_contains_nextdayship',
 'brms_total_gift_card_value',
 'brms_contains_hrshoes',
 'brms_shipping_equals_billing',
 'brms_is_firefox',
 'brms_internal_score',
 'brms_resulting_balance',
 'brms_resulting_balance_sibc_email',
 'abuse_2019_v2_score',
 'abuse_2020_v3_score',
 'transparent_2019_score',
 'transparent_2020_score',
 'tu_most_negative_influence_on_vantagescore_factor1',
 'tu_most_negative_influence_on_vantagescore_factor2',
 'tu_most_negative_influence_on_vantagescore_factor3',
 'tu_most_negative_influence_on_vantagescore_factor4',
 'tu_vtg4_most_negative_influence_factor1',
 'tu_vtg4_most_negative_influence_factor2',
 'tu_vtg4_most_negative_influence_factor3',
 'tu_vtg4_most_negative_influence_factor4',
 'tu_vtg4',
 'tu_installment_trade_count',
 'tu_mortgage_trade_count',
 'tu_unspecified_trade_count',
 'tu_total_trade_count',
 'tu_open_trade_count',
 'tu_negative_trade_count',
 'tu_historical_negative_trade_count',
 'tu_collection_count',
 'tu_total_balance_of_collections',
 'tu_historical_negative_occurrences_count',
 'tu_public_record_count',
 'tu_total_inquiry_count',
 'tu_paymnt10',
 'tu_paymnt11',
 'tu_rev12',
 'tu_rev122',
 'tu_rev205',
 'tu_rev321',
 'tu_revs124',
 'tu_g001s',
 'tu_g106s',
 'tu_g209s',
 'tu_g230s',
 'tu_g416s',
 'tu_g990s',
 'tu_re01s',
 'tu_re20s',
 'tu_re24s',
 'tu_re102s',
 'tu_trv01',
 'tu_trv02',
 'tu_cv05',
 'tu_cv14',
 'tu_s004s',
 'tu_se20s',
 'tu_s114s',
 'tu_bkc11',
 'tu_bkc122',
 'tu_bkc322',
 'tu_ret201',
 'tu_in20s',
 'tu_aggs911',
 'tu_br101s',
 'tu_at01a',
 'tu_dob',
 'tu_no_currently_satisfactory_revolving_trades2',
 'tu_open_revolving_trades_verified_past12_months',
 'tu_open_bank_revolving_trades_verified_past12_months',
 'tu_open_trades_verified_past12_months',
 'tu_open_trades_verified_past12_months_excl_mortage',
 'tu_no_scheduled_monthly_payment_open_mortgage_trades',
 'tu_monthly_payment_on_open_or_closed_trades',
 'tu_no_bankruptcies_last24months',
 'tu_delinquent_trades_60days',
 'tu_identified_as_fraud_alert',
 'tu_militarty_active_duty',
 'tu_retrievedfromcache',
 'tu_not_enough_info',
 'tu_freeze',
 'rvl_cid_num_invoice_0_3m',
 'rvl_cid_num_arch_written_off_6_12m',
 'rvl_cid_amount_2nd_last_purchase_0_3m',
 'rvl_cid_days_since_last_purchase_0_3m',
 'rvl_cid_account_days_in_term_0_12m',
 'rvl_cid_sum_fees_account_0_3m',
 'rvl_cid_account_pstatus',
 'rvl_cid_pstatus_2nd_last_archived_0_6m',
 'rvl_cid_num_paid_inv_3_6m',
 'rvl_cid_account_amount_added_0_12m',
 'rvl_cid_account_days_in_ok_6_12m',
 'rvl_cid_sum_paid_account_0_3m',
 'rvl_cid_num_klarna_direct_0_3m',
 'rvl_cid_num_arch_enforcement_0_24m',
 'rvl_cid_account_days_in_term_12_24m',
 'rvl_cid_has_paid_credit',
 'rvl_cid_num_arch_rem_6_12m',
 'rvl_cid_num_klarna_direct_0_6m',
 'rvl_cid_pstatus_max_archived_0_3_months',
 'rvl_cid_account_days_in_ok_12_24m',
 'rvl_cid_num_arch_written_off_3_12m',
 'rvl_cid_max_paid_account_6_12m',
 'rvl_cid_amount_last_purchase_0_3m',
 'rvl_cid_active_account_age_0_12m',
 'rvl_cid_num_arch_rem_12_24m',
 'rvl_cid_stds_inv_ticket_size_0_3m',
 'rvl_cid_account_worst_pstatus_0_12m',
 'rvl_cid_num_arch_written_off_0_6m',
 'rvl_cid_account_days_in_dc_6_12m',
 'rvl_cid_num_arch_rem_0_6m',
 'rvl_cid_num_arch_dc_0_24m',
 'rvl_cid_num_invoice_0_24m',
 'rvl_cid_account_worst_pstatus_3_6m',
 'rvl_cid_max_num_failed_sibc_payment_attempts',
 'rvl_cid_max_paid_account_12_24m',
 'rvl_cid_days_since_2nd_last_purchase_0_3m',
 'rvl_cid_account_days_in_rem_0_12m',
 'rvl_cid_hours_since_2nd_last_rej_0_14d',
 'rvl_cid_has_paid_24m',
 'rvl_cid_num_paid_orders',
 'rvl_cid_avg_inv_ticket_size_0_12m',
 'rvl_cid_sum_paid_account_12_24m',
 'rvl_cid_pstatus_2nd_last_archived_0_3m',
 'rvl_cid_days_since_last_payment_credit',
 'rvl_cid_account_days_in_rem_0_24m',
 'rvl_cid_num_arch_dc_12_24m',
 'rvl_cid_num_arch_rem_0_3m',
 'rvl_cid_num_arch_written_off_0_3m',
 'rvl_cid_account_amount_added_0_3m',
 'rvl_cid_num_arch_enforcement_0_12m',
 'rvl_cid_avg_inv_ticket_size_0_6m',
 'rvl_cid_has_paid_inv',
 'rvl_cid_avg_payment_span_0_12m',
 'rvl_cid_pstatus_max_archived_0_12_months',
 'rvl_cid_max_paid_account_0_3m',
 'rvl_cid_num_paid_inv_12_24m',
 'rvl_cid_account_days_in_term_3_6m',
 'rvl_cid_num_pix_0_6m',
 'rvl_cid_num_flexible_0_6m',
 'rvl_cid_account_days_in_ok_0_3m',
 'rvl_cid_num_direct_banking_0_6m',
 'rvl_cid_num_arch_enforcement_0_6m',
 'rvl_cid_account_days_delinquent_0_3m',
 'rvl_cid_num_arch_12_24m',
 'rvl_cid_num_arch_ok_or_rem_3_6m',
 'rvl_cid_days_since_last_payment',
 'rvl_cid_incoming_debt_incl_3y_sold',
 'rvl_cid_num_arch_written_off_0_12m',
 'rvl_cid_avg_inv_ticket_size_0_3m',
 'rvl_cid_has_paid_12m',
 'rvl_cid_num_arch_rem_0_24m',
 'rvl_cid_sum_fees_account_6_12m',
 'rvl_cid_sum_paid_account_3_6m',
 'rvl_cid_account_days_delinquent_0_12m',
 'rvl_cid_has_paid_credit_24m',
 'rvl_cid_has_paid_credit_12m',
 'rvl_cid_sum_fees_account_3_6m',
 'rvl_cid_stds_inv_ticket_size_0_24m',
 'rvl_cid_sum_fees_account_12_24m',
 'rvl_cid_pstatus_max_archived_0_24_months',
 'rvl_cid_num_arch_worse_than_rem_0_3m',
 'rvl_cid_num_part_pay_0_6m',
 'rvl_cid_hours_since_1st_last_rej_0_14d',
 'rvl_cid_num_flexible_0_3m',
 'rvl_cid_pstatus_last_archived_0_12m',
 'rvl_cid_num_arch_enforcement_0_3m',
 'rvl_cid_has_paid_inv_12m',
 'rvl_cid_account_age',
 'rvl_cid_num_pix_0_3m',
 'rvl_cid_num_arch_ok_or_rem_6_12m',
 'rvl_cid_num_direct_banking_0_3m',
 'rvl_cid_num_dist_emails_2d_12m',
 'rvl_cid_sum_paid_account_6_12m',
 'rvl_cid_num_dist_emails_2d_6m',
 'rvl_cid_pstatus_last_archived_0_24m',
 'rvl_cid_incoming_invoice_debt_incl_3y_sold',
 'rvl_cid_has_paid_inv_24m',
 'rvl_cid_account_incoming_pstatus_6m',
 'rvl_cid_account_amount_added_6_12m',
 'rvl_cid_account_days_in_ok_0_24m',
 'rvl_cid_account_days_in_term_6_12m',
 'rvl_cid_max_paid_inv_0_12m',
 'rvl_cid_account_days_in_rem_0_3m',
 'rvl_cid_account_recovery_int',
 'rvl_cid_pstatus_2nd_last_archived_0_12m',
 'rvl_cid_num_part_pay_0_3m',
 'rvl_cid_num_dist_emails_2d_2w',
 'rvl_cid_pstatus_max_archived_0_6_months',
 'rvl_cid_account_days_in_ok_0_12m',
 'rvl_cid_num_arch_ok_or_rem_3_12m',
 'rvl_cid_payment_method_2nd_last_purchase_0_3m',
 'rvl_cid_sum_paid_account_0_36m',
 'rvl_cid_cid_age',
 'rvl_cid_account_worst_pstatus_6_12m',
 'rvl_cid_has_paid_acct_24m',
 'rvl_cid_num_dist_emails_2d_3m',
 'rvl_cid_account_amount_added_3_6m',
 'rvl_cid_num_converted_invoices_0_6m',
 'rvl_cid_sum_paid_inv_0_3m',
 'rvl_cid_num_arch_written_off_3_6m',
 'rvl_cid_max_paid_inv_0_6m',
 'rvl_cid_num_elv_0_6m',
 'rvl_cid_num_credit_card_0_6m',
 'rvl_cid_has_history',
 'rvl_cid_account_days_in_rem_12_24m',
 'rvl_cid_pstatus_3rd_last_archived_0_12m',
 'rvl_cid_has_paid_credit_24m_int',
 'rvl_cid_payment_method_last_purchase_0_3m',
 'rvl_cid_max_paid_account_3_6m',
 'rvl_cid_num_paid_inv_0_3m',
 'rvl_cid_sum_paid_inv_0_6m',
 'rvl_cid_has_rejection_14d_int',
 'rvl_cid_account_incoming_pstatus_3m',
 'rvl_cid_incoming_debt',
 'rvl_cid_sum_fees_account_0_24m',
 'rvl_cid_num_arch_enforcement_6_12m',
 'rvl_cid_num_converted_invoices_0_3m',
 'rvl_cid_max_paid_inv_0_3m',
 'rvl_cid_account_days_in_ok_3_6m',
 'rvl_cid_has_2nd_rejection_14d',
 'rvl_cid_worst_pstatus_active_inv',
 'rvl_cid_account_worst_pstatus_0_3m',
 'rvl_cid_has_paid_inv_12m_int',
 'rvl_cid_num_elv_0_3m',
 'rvl_cid_oldest_pstatus_active_inv',
 'rvl_cid_has_paid_acct_12m',
 'rvl_cid_account_debt',
 'rvl_cid_num_credit_card_0_3m',
 'rvl_cid_sum_paid_account_0_12m',
 'rvl_cid_account_days_in_rem_6_12m',
 'rvl_cid_num_arch_enforcement_12_24m',
 'rvl_cid_account_days_in_term_0_24m',
 'rvl_cid_num_arch_worse_than_rem_12_24m',
 'rvl_cid_avg_payment_span_0_6m',
 'rvl_cid_account_amount_added_0_24m',
 'rvl_cid_account_worst_pstatus_3_12m',
 'rvl_cid_num_arch_enforcement_3_6m',
 'rvl_cid_pstatus_3rd_last_archived_0_6m',
 'rvl_cid_account_amount_added_12_24m',
 'rvl_cid_account_days_in_rem_3_6m',
 'rvl_cid_incoming_account_debt',
 'rvl_cid_sum_paid_account_0_6m',
 'rvl_cid_num_arch_worse_than_rem_6_12m',
 'rvl_cid_account_days_in_term_0_3m',
 'rvl_cid_has_paid_acct',
 'rvl_cid_sum_fees_account_0_12m',
 'rvl_cid_sum_fees_account_0_6m',
 'rvl_cid_num_unpaid_sibc_orders',
 'rvl_cid_num_invoice_0_6m',
 'rvl_cid_sum_paid_inv_12_24m',
 'rvl_cid_invoice_debt',
 'rvl_cid_num_arch_enforcement_3_12m',
 'rvl_cid_account_days_in_dc_12_24m',
 'rvl_cid_num_arch_dc_0_6m',
 'rvl_cid_has_paid',
 'rvl_cid_has_paid_inv_24m_int',
 'rvl_cid_avg_payment_span_0_3m',
 'rvl_cid_pstatus_last_archived_0_6m',
 'rvl_cid_stds_inv_ticket_size_0_6m',
 'rvl_cid_pstatus_3rd_last_archived_0_3m',
 'rvl_cid_email_has_paid_24m_int',
 'rvl_composite_amount_vs_estore_avg_12m',
 'rvl_domain_bad_rate_12m',
 'rvl_zip3_bad_moving_count_12m',
 'rvl_amount_vs_inv_ticket_avg_24m',
 'rvl_incoming_household_debt',
 'rvl_num_dist_billing_phones_on_email_2d_2w',
 'rvl_cid_zip_has_paid_int',
 'rvl_country_bad_count_coarse_12m',
 'rvl_num_dist_billing_addresses_on_email_2d_1w',
 'crid_realtime_incoming_invoice_debt',
 'crid_realtime_incoming_sibc_debt',
 'crid_realtime_num_unpaid_sibc',
 'payment_method_downpayment',
 'payment_method_number_of_installments',
 'payment_method_card_type',
 'payment_method_card_level',
 'payment_method_card_issuing_bank',
 'payment_method_card_exp_year',
 'payment_method_card_exp_month',
 'payment_method_card_brand',
 'payment_method_credit_card_3ds_supported',
 'usemailageemailiprisk_lastflaggedon_days',
 'usemailageemailiprisk_domain_age_days',
 'usemailageemailiprisk_last_verification_date_days',
 'unpaid_at_60_rate',
 'is_shopping']

 # final col_list no score
final_col_list_noscore = ['captured_amount',
 'nr_of_captures',
 'total_tax_amount',
 'psp_credit_card_issuing_bank',
 'acquiring_source',
 'is_paused',
 'nr_errands',
 'decision_time',
 'usemailageemailiprisk_ipdomain',
 'usemailageemailiprisk_ip_continent_code',
 'usemailageemailiprisk_ipasnum',
 'usemailageemailiprisk_custphone_inbillingloc',
 'usemailageemailiprisk_ip_reputation',
 'usemailageemailiprisk_ip_org',
 'usemailageemailiprisk_count',
 'usemailageemailiprisk_ip_callingcode',
 'usemailageemailiprisk_ip_country_code',
 'usemailageemailiprisk__e_a_advice_i_d',
 'usemailageemailiprisk_phone_status',
 'usemailageemailiprisk_phoneownermatch',
 'usemailageemailiprisk_totalhits',
 'usemailageemailiprisk_ip_postalconf',
 'usemailageemailiprisk_shipcitypostalmatch',
 'usemailageemailiprisk_status',
 'usemailageemailiprisk_ip_risklevelid',
 'usemailageemailiprisk_fraud_risk',
 'usemailageemailiprisk_response_count',
 'usemailageemailiprisk_ip_cityconf',
 'usemailageemailiprisk__e_a_advice',
 'usemailageemailiprisk_domaincompany',
 'usemailageemailiprisk__e_a_status_i_d',
 'usemailageemailiprisk_phonecarriername',
 'usemailageemailiprisk_ip_risklevel',
 'usemailageemailiprisk__e_a_risk_band',
 'usemailageemailiprisk_country',
 'usemailageemailiprisk_uniquehits',
 'usemailageemailiprisk_ip_riskreasonid',
 'usemailageemailiprisk_citypostalmatch',
 'usemailageemailiprisk_ip_countryconf',
 'usemailageemailiprisk_ipcountrymatch',
 'usemailageemailiprisk_namematch',
 'usemailageemailiprisk_domainrelevantinfo',
 'usemailageemailiprisk_domainrisklevel',
 'usemailageemailiprisk_domainname',
 'usemailageemailiprisk_domaincountrymatch',
 'usemailageemailiprisk_ip_region',
 'usemailageemailiprisk_shipforward',
 'usemailageemailiprisk_ip_city',
 'usemailageemailiprisk__e_a_reason',
 'usemailageemailiprisk_domainrisklevel_i_d',
 'usemailageemailiprisk_domaincategory',
 'usemailageemailiprisk_ip_riskreason',
 'usemailageemailiprisk_ip_user_type',
 'usemailageemailiprisk_lookup_timestamp_ms_since_epoch',
 'usemailageemailiprisk_domainrelevantinfo_i_d',
 'usemailageemailiprisk__e_a_score',
 'usemailageemailiprisk_domaincountryname',
 'usemailageemailiprisk__e_a_reason_i_d',
 'usemailageemailiprisk_ip_isp',
 'usemailageemailiprisk_ipdistancemil',
 'usemailageemailiprisk__e_a_risk_band_i_d',
 'usemailageemailiprisk_ip_regionconf',
 'usemailageemailiprisk_domaincorporate',
 'usemailageemailiprisk_ip_metro_code',
 'usemailageemailiprisk_source_industry',
 'usemailageemailiprisk_email_exists',
 'usemailageemailiprisk_ip_corporate_proxy',
 'usemailageemailiprisk_retrievedfromcache',
 'usemailageemailiprisk_phonecarriertype',
 'usemailageemailiprisk_smfriends',
 'rvl_eid_estore_category',
 'rvl_eid_estore_id_avg_ts_0_12m',
 'rvl_eid_estore_id_fraud_rate_30d',
 'rvl_eid_estore_id_fraud_cnt_30d',
 'rvl_eid_estore_group_bad_rate_12m',
 'rvl_eid_estore_category_bad_rate_12m',
 'rvl_eid_estore_id_avg_age_0_12m',
 'rvl_eid_estore_id_tot_tx_0_12m',
 'velocityvariables_apm_cids_on_shipping_email_1m',
 'velocityvariables_approved_apm_order_ids_on_cid_14d',
 'velocityvariables_apm_card_cids_on_ip_zip_1d',
 'velocityvariables_shipping_full_names_on_account_id_14d',
 'velocityvariables_shipping_street_zip_on_order_id_14d',
 'velocityvariables_acp_on_cid_shipping_zip_1d',
 'velocityvariables_apm_pad_cids_on_ip_zip_1h',
 'velocityvariables_apm_on_billing_email_2d',
 'velocityvariables_card_countries_on_shipping_zip_1d',
 'velocityvariables_cids_on_shipping_email_2d',
 'velocityvariables_apm_order_ids_on_shipping_street_zip_2d',
 'velocityvariables_cards_on_shipping_email_1m',
 'velocityvariables_cids_on_shipping_email_1h',
 'velocityvariables_apm_card_cids_on_ip_shipping_zip_2d',
 'velocityvariables_apm_card_cids_on_ip_shipping_zip_1h',
 'velocityvariables_billing_emails_on_billing_phone_2d',
 'velocityvariables_apm_order_ids_on_billing_email_cid_2d',
 'velocityvariables_billing_emails_on_billing_phone_1h',
 'velocityvariables_card_numbers_on_cid_1h',
 'velocityvariables_apm_billing_emails_on_billing_street_zip_ip_2d',
 'velocityvariables_apm_order_ids_on_billing_street_zip_1h',
 'velocityvariables_shipping_phones_on_account_id_14d',
 'velocityvariables_card_countries_on_shipping_zip_3d',
 'velocityvariables_gpm_on_order_id_1d',
 'velocityvariables_apm_order_ids_on_billing_street_zip_2d',
 'velocityvariables_apm_cids_on_card_7d',
 'velocityvariables_apm_pad_cids_on_ip_zip_1d',
 'velocityvariables_apm_on_billing_street_zip_ip_2d',
 'velocityvariables_acp_on_shipping_email_zip_1h',
 'velocityvariables_apm_cids_on_card_1d',
 'velocityvariables_apm_cids_on_card_1m',
 'velocityvariables_card_countries_on_shipping_zip_14d',
 'velocityvariables_apm_on_billing_phone_1h',
 'velocityvariables_apm_billing_emails_on_billing_street_zip_ip_1h',
 'velocityvariables_apm_emails_on_order_id_7d',
 'velocityvariables_is_error',
 'velocityvariables_account_ids_on_cid_14d',
 'velocityvariables_acp_on_ip_shipping_zip_1h',
 'velocityvariables_billing_emails_on_card_1h',
 'velocityvariables_apm_on_shipping_street_zip_ip_2d',
 'velocityvariables_apm_card_cids_on_ip_shipping_zip_1d',
 'velocityvariables_card_numbers_on_cid_1d',
 'velocityvariables_acp_on_ip_shipping_zip_7d',
 'velocityvariables_acp_on_shipping_email_zip_7d',
 'velocityvariables_billing_emails_on_billing_phone_1m',
 'velocityvariables_billing_phones_on_account_id_14d',
 'velocityvariables_billing_emails_on_card_2d',
 'velocityvariables_cids_on_shipping_street_zip_ip_2d',
 'velocityvariables_cids_on_ip_1h',
 'velocityvariables_apm_cids_on_card_number_exp_year_exp_month_7d',
 'velocityvariables_account_ids_on_shipping_zip_14d',
 'velocityvariables_cids_on_shipping_street_zip_ip_1h',
 'velocityvariables_billing_full_names_on_account_id_14d',
 'velocityvariables_apm_card_cids_on_ip_zip_2d',
 'velocityvariables_is_disabled',
 'velocityvariables_apm_pad_cids_on_ip_1d',
 'velocityvariables_apm_cids_on_card_identifier_7d',
 'velocityvariables_shipping_full_name_on_order_id_1h',
 'velocityvariables_apm_card_cids_on_ip_zip_1h',
 'velocityvariables_acp_on_cid_shipping_zip_1h',
 'velocityvariables_order_ids_on_card_1h',
 'velocityvariables_order_ids_on_card_2d',
 'velocityvariables_account_ids_on_billing_zip_14d',
 'velocityvariables_acp_on_ip_shipping_zip_1d',
 'velocityvariables_apm_billing_emails_on_billing_street_zip_ip_1m',
 'velocityvariables_apm_emails_on_cid_2d',
 'velocityvariables_apm_cids_on_order_id_7d',
 'velocityvariables_acp_on_shipping_email_zip_1d',
 'velocityvariables_acp_on_cid_shipping_zip_7d',
 'velocityvariables_billing_emails_on_card_7d',
 'velocityvariables_billing_street_zip_on_order_id_1h',
 'velocityvariables_apm_order_ids_on_ip_1h',
 'velocityvariables_order_ids_on_card_7d',
 'velocityvariables_apm_order_ids_on_shipping_email_cid_1h',
 'velocityvariables_apm_order_id_7d',
 'velocityvariables_apm_card_order_id_7d',
 'velocityvariables_apm_cids_on_card_number_exp_year_exp_month_1m',
 'velocityvariables_billing_full_name_on_order_id_1h',
 'velocityvariables_approved_apm_order_ids_on_cid_1m',
 'velocityvariables_apm_cids_on_card_number_exp_year_exp_month_1d',
 'velocityvariables_cards_on_shipping_email_test4_7d',
 'velocityvariables_approved_apm_order_ids_on_cid_1d',
 'velocityvariables_apm_cids_on_card_identifier_1m',
 'velocityvariables_apm_cids_on_card_identifier_1d',
 'velocityvariables_cids_on_shipping_street_zip_ip_1m',
 'velocityvariables_apm_cids_on_shipping_email_7d',
 'velocityvariables_apm_country_zip_street_2d',
 'velocityvariables_apm_country_zip_street_1h',
 'velocityvariables_apm_pad_cids_on_ip_zip_2d',
 'brms_number_of_articles',
 'brms_total_gift_card_count',
 'brms_is_returning_sibc',
 'brms_day_of_week',
 'brms_sum_paid',
 'brms_seconds_since_midnight',
 'brms_cheapest_article_price',
 'brms_is_po_box',
 'brms_number_of_distinct_articles',
 'brms_max_same_article_price',
 'brms_max_same_article_count',
 'brms_contains_electronic_gift_card',
 'brms_contains_expshipping',
 'brms_contains_gift_card',
 'brms_device',
 'brms_contains_nextdayship',
 'brms_total_gift_card_value',
 'brms_contains_hrshoes',
 'brms_shipping_equals_billing',
 'brms_is_firefox',
 'brms_resulting_balance',
 'brms_resulting_balance_sibc_email',
 'tu_most_negative_influence_on_vantagescore_factor1',
 'tu_most_negative_influence_on_vantagescore_factor2',
 'tu_most_negative_influence_on_vantagescore_factor3',
 'tu_most_negative_influence_on_vantagescore_factor4',
 'tu_vtg4_most_negative_influence_factor1',
 'tu_vtg4_most_negative_influence_factor2',
 'tu_vtg4_most_negative_influence_factor3',
 'tu_vtg4_most_negative_influence_factor4',
 'tu_vtg4',
 'tu_installment_trade_count',
 'tu_mortgage_trade_count',
 'tu_unspecified_trade_count',
 'tu_total_trade_count',
 'tu_open_trade_count',
 'tu_negative_trade_count',
 'tu_historical_negative_trade_count',
 'tu_collection_count',
 'tu_total_balance_of_collections',
 'tu_historical_negative_occurrences_count',
 'tu_public_record_count',
 'tu_total_inquiry_count',
 'tu_paymnt10',
 'tu_paymnt11',
 'tu_rev12',
 'tu_rev122',
 'tu_rev205',
 'tu_rev321',
 'tu_revs124',
 'tu_g001s',
 'tu_g106s',
 'tu_g209s',
 'tu_g230s',
 'tu_g416s',
 'tu_g990s',
 'tu_re01s',
 'tu_re20s',
 'tu_re24s',
 'tu_re102s',
 'tu_trv01',
 'tu_trv02',
 'tu_cv05',
 'tu_cv14',
 'tu_s004s',
 'tu_se20s',
 'tu_s114s',
 'tu_bkc11',
 'tu_bkc122',
 'tu_bkc322',
 'tu_ret201',
 'tu_in20s',
 'tu_aggs911',
 'tu_br101s',
 'tu_at01a',
 'tu_dob',
 'tu_no_currently_satisfactory_revolving_trades2',
 'tu_open_revolving_trades_verified_past12_months',
 'tu_open_bank_revolving_trades_verified_past12_months',
 'tu_open_trades_verified_past12_months',
 'tu_open_trades_verified_past12_months_excl_mortage',
 'tu_no_scheduled_monthly_payment_open_mortgage_trades',
 'tu_monthly_payment_on_open_or_closed_trades',
 'tu_no_bankruptcies_last24months',
 'tu_delinquent_trades_60days',
 'tu_identified_as_fraud_alert',
 'tu_militarty_active_duty',
 'tu_retrievedfromcache',
 'tu_not_enough_info',
 'tu_freeze',
 'rvl_cid_num_invoice_0_3m',
 'rvl_cid_num_arch_written_off_6_12m',
 'rvl_cid_amount_2nd_last_purchase_0_3m',
 'rvl_cid_days_since_last_purchase_0_3m',
 'rvl_cid_account_days_in_term_0_12m',
 'rvl_cid_sum_fees_account_0_3m',
 'rvl_cid_account_pstatus',
 'rvl_cid_pstatus_2nd_last_archived_0_6m',
 'rvl_cid_num_paid_inv_3_6m',
 'rvl_cid_account_amount_added_0_12m',
 'rvl_cid_account_days_in_ok_6_12m',
 'rvl_cid_sum_paid_account_0_3m',
 'rvl_cid_num_klarna_direct_0_3m',
 'rvl_cid_num_arch_enforcement_0_24m',
 'rvl_cid_account_days_in_term_12_24m',
 'rvl_cid_has_paid_credit',
 'rvl_cid_num_arch_rem_6_12m',
 'rvl_cid_num_klarna_direct_0_6m',
 'rvl_cid_pstatus_max_archived_0_3_months',
 'rvl_cid_account_days_in_ok_12_24m',
 'rvl_cid_num_arch_written_off_3_12m',
 'rvl_cid_max_paid_account_6_12m',
 'rvl_cid_amount_last_purchase_0_3m',
 'rvl_cid_active_account_age_0_12m',
 'rvl_cid_num_arch_rem_12_24m',
 'rvl_cid_stds_inv_ticket_size_0_3m',
 'rvl_cid_account_worst_pstatus_0_12m',
 'rvl_cid_num_arch_written_off_0_6m',
 'rvl_cid_account_days_in_dc_6_12m',
 'rvl_cid_num_arch_rem_0_6m',
 'rvl_cid_num_arch_dc_0_24m',
 'rvl_cid_num_invoice_0_24m',
 'rvl_cid_account_worst_pstatus_3_6m',
 'rvl_cid_max_num_failed_sibc_payment_attempts',
 'rvl_cid_max_paid_account_12_24m',
 'rvl_cid_days_since_2nd_last_purchase_0_3m',
 'rvl_cid_account_days_in_rem_0_12m',
 'rvl_cid_hours_since_2nd_last_rej_0_14d',
 'rvl_cid_has_paid_24m',
 'rvl_cid_num_paid_orders',
 'rvl_cid_avg_inv_ticket_size_0_12m',
 'rvl_cid_sum_paid_account_12_24m',
 'rvl_cid_pstatus_2nd_last_archived_0_3m',
 'rvl_cid_days_since_last_payment_credit',
 'rvl_cid_account_days_in_rem_0_24m',
 'rvl_cid_num_arch_dc_12_24m',
 'rvl_cid_num_arch_rem_0_3m',
 'rvl_cid_num_arch_written_off_0_3m',
 'rvl_cid_account_amount_added_0_3m',
 'rvl_cid_num_arch_enforcement_0_12m',
 'rvl_cid_avg_inv_ticket_size_0_6m',
 'rvl_cid_has_paid_inv',
 'rvl_cid_avg_payment_span_0_12m',
 'rvl_cid_pstatus_max_archived_0_12_months',
 'rvl_cid_max_paid_account_0_3m',
 'rvl_cid_num_paid_inv_12_24m',
 'rvl_cid_account_days_in_term_3_6m',
 'rvl_cid_num_pix_0_6m',
 'rvl_cid_num_flexible_0_6m',
 'rvl_cid_account_days_in_ok_0_3m',
 'rvl_cid_num_direct_banking_0_6m',
 'rvl_cid_num_arch_enforcement_0_6m',
 'rvl_cid_account_days_delinquent_0_3m',
 'rvl_cid_num_arch_12_24m',
 'rvl_cid_num_arch_ok_or_rem_3_6m',
 'rvl_cid_days_since_last_payment',
 'rvl_cid_incoming_debt_incl_3y_sold',
 'rvl_cid_num_arch_written_off_0_12m',
 'rvl_cid_avg_inv_ticket_size_0_3m',
 'rvl_cid_has_paid_12m',
 'rvl_cid_num_arch_rem_0_24m',
 'rvl_cid_sum_fees_account_6_12m',
 'rvl_cid_sum_paid_account_3_6m',
 'rvl_cid_account_days_delinquent_0_12m',
 'rvl_cid_has_paid_credit_24m',
 'rvl_cid_has_paid_credit_12m',
 'rvl_cid_sum_fees_account_3_6m',
 'rvl_cid_stds_inv_ticket_size_0_24m',
 'rvl_cid_sum_fees_account_12_24m',
 'rvl_cid_pstatus_max_archived_0_24_months',
 'rvl_cid_num_arch_worse_than_rem_0_3m',
 'rvl_cid_num_part_pay_0_6m',
 'rvl_cid_hours_since_1st_last_rej_0_14d',
 'rvl_cid_num_flexible_0_3m',
 'rvl_cid_pstatus_last_archived_0_12m',
 'rvl_cid_num_arch_enforcement_0_3m',
 'rvl_cid_has_paid_inv_12m',
 'rvl_cid_account_age',
 'rvl_cid_num_pix_0_3m',
 'rvl_cid_num_arch_ok_or_rem_6_12m',
 'rvl_cid_num_direct_banking_0_3m',
 'rvl_cid_num_dist_emails_2d_12m',
 'rvl_cid_sum_paid_account_6_12m',
 'rvl_cid_num_dist_emails_2d_6m',
 'rvl_cid_pstatus_last_archived_0_24m',
 'rvl_cid_incoming_invoice_debt_incl_3y_sold',
 'rvl_cid_has_paid_inv_24m',
 'rvl_cid_account_incoming_pstatus_6m',
 'rvl_cid_account_amount_added_6_12m',
 'rvl_cid_account_days_in_ok_0_24m',
 'rvl_cid_account_days_in_term_6_12m',
 'rvl_cid_max_paid_inv_0_12m',
 'rvl_cid_account_days_in_rem_0_3m',
 'rvl_cid_account_recovery_int',
 'rvl_cid_pstatus_2nd_last_archived_0_12m',
 'rvl_cid_num_part_pay_0_3m',
 'rvl_cid_num_dist_emails_2d_2w',
 'rvl_cid_pstatus_max_archived_0_6_months',
 'rvl_cid_account_days_in_ok_0_12m',
 'rvl_cid_num_arch_ok_or_rem_3_12m',
 'rvl_cid_payment_method_2nd_last_purchase_0_3m',
 'rvl_cid_sum_paid_account_0_36m',
 'rvl_cid_cid_age',
 'rvl_cid_account_worst_pstatus_6_12m',
 'rvl_cid_has_paid_acct_24m',
 'rvl_cid_num_dist_emails_2d_3m',
 'rvl_cid_account_amount_added_3_6m',
 'rvl_cid_num_converted_invoices_0_6m',
 'rvl_cid_sum_paid_inv_0_3m',
 'rvl_cid_num_arch_written_off_3_6m',
 'rvl_cid_max_paid_inv_0_6m',
 'rvl_cid_num_elv_0_6m',
 'rvl_cid_num_credit_card_0_6m',
 'rvl_cid_has_history',
 'rvl_cid_account_days_in_rem_12_24m',
 'rvl_cid_pstatus_3rd_last_archived_0_12m',
 'rvl_cid_has_paid_credit_24m_int',
 'rvl_cid_payment_method_last_purchase_0_3m',
 'rvl_cid_max_paid_account_3_6m',
 'rvl_cid_num_paid_inv_0_3m',
 'rvl_cid_sum_paid_inv_0_6m',
 'rvl_cid_has_rejection_14d_int',
 'rvl_cid_account_incoming_pstatus_3m',
 'rvl_cid_incoming_debt',
 'rvl_cid_sum_fees_account_0_24m',
 'rvl_cid_num_arch_enforcement_6_12m',
 'rvl_cid_num_converted_invoices_0_3m',
 'rvl_cid_max_paid_inv_0_3m',
 'rvl_cid_account_days_in_ok_3_6m',
 'rvl_cid_has_2nd_rejection_14d',
 'rvl_cid_worst_pstatus_active_inv',
 'rvl_cid_account_worst_pstatus_0_3m',
 'rvl_cid_has_paid_inv_12m_int',
 'rvl_cid_num_elv_0_3m',
 'rvl_cid_oldest_pstatus_active_inv',
 'rvl_cid_has_paid_acct_12m',
 'rvl_cid_account_debt',
 'rvl_cid_num_credit_card_0_3m',
 'rvl_cid_sum_paid_account_0_12m',
 'rvl_cid_account_days_in_rem_6_12m',
 'rvl_cid_num_arch_enforcement_12_24m',
 'rvl_cid_account_days_in_term_0_24m',
 'rvl_cid_num_arch_worse_than_rem_12_24m',
 'rvl_cid_avg_payment_span_0_6m',
 'rvl_cid_account_amount_added_0_24m',
 'rvl_cid_account_worst_pstatus_3_12m',
 'rvl_cid_num_arch_enforcement_3_6m',
 'rvl_cid_pstatus_3rd_last_archived_0_6m',
 'rvl_cid_account_amount_added_12_24m',
 'rvl_cid_account_days_in_rem_3_6m',
 'rvl_cid_incoming_account_debt',
 'rvl_cid_sum_paid_account_0_6m',
 'rvl_cid_num_arch_worse_than_rem_6_12m',
 'rvl_cid_account_days_in_term_0_3m',
 'rvl_cid_has_paid_acct',
 'rvl_cid_sum_fees_account_0_12m',
 'rvl_cid_sum_fees_account_0_6m',
 'rvl_cid_num_unpaid_sibc_orders',
 'rvl_cid_num_invoice_0_6m',
 'rvl_cid_sum_paid_inv_12_24m',
 'rvl_cid_invoice_debt',
 'rvl_cid_num_arch_enforcement_3_12m',
 'rvl_cid_account_days_in_dc_12_24m',
 'rvl_cid_num_arch_dc_0_6m',
 'rvl_cid_has_paid',
 'rvl_cid_has_paid_inv_24m_int',
 'rvl_cid_avg_payment_span_0_3m',
 'rvl_cid_pstatus_last_archived_0_6m',
 'rvl_cid_stds_inv_ticket_size_0_6m',
 'rvl_cid_pstatus_3rd_last_archived_0_3m',
 'rvl_cid_email_has_paid_24m_int',
 'rvl_composite_amount_vs_estore_avg_12m',
 'rvl_domain_bad_rate_12m',
 'rvl_zip3_bad_moving_count_12m',
 'rvl_amount_vs_inv_ticket_avg_24m',
 'rvl_incoming_household_debt',
 'rvl_num_dist_billing_phones_on_email_2d_2w',
 'rvl_cid_zip_has_paid_int',
 'rvl_country_bad_count_coarse_12m',
 'rvl_num_dist_billing_addresses_on_email_2d_1w',
 'crid_realtime_incoming_invoice_debt',
 'crid_realtime_incoming_sibc_debt',
 'crid_realtime_num_unpaid_sibc',
 'payment_method_downpayment',
 'payment_method_number_of_installments',
 'payment_method_card_type',
 'payment_method_card_level',
 'payment_method_card_issuing_bank',
 'payment_method_card_exp_year',
 'payment_method_card_exp_month',
 'payment_method_card_brand',
 'payment_method_credit_card_3ds_supported',
 'usemailageemailiprisk_lastflaggedon_days',
 'usemailageemailiprisk_domain_age_days',
 'usemailageemailiprisk_last_verification_date_days',
 'unpaid_at_60_rate',
 'is_shopping']

 #browser-new final col list
final_col_list_browser_new = ['captured_amount',
 'nr_of_captures',
#  'merchant_id',
 'total_tax_amount',
 'months_to_card_exp',
 'psp_credit_card_issuing_bank',
 'acquiring_source',
 'is_paused',
 'nr_errands',
 'decision_time',
 'usemailageemailiprisk_ipdomain',
 'usemailageemailiprisk_ip_continent_code',
 'usemailageemailiprisk_ipasnum',
 'usemailageemailiprisk_custphone_inbillingloc',
 'usemailageemailiprisk_ip_reputation',
 'usemailageemailiprisk_ip_org',
 'usemailageemailiprisk_count',
 'usemailageemailiprisk_ip_callingcode',
 'usemailageemailiprisk_ip_country_code',
 'usemailageemailiprisk__e_a_advice_i_d',
 'usemailageemailiprisk_phone_status',
 'usemailageemailiprisk_phoneownermatch',
 'usemailageemailiprisk_totalhits',
 'usemailageemailiprisk_ip_postalconf',
 'usemailageemailiprisk_shipcitypostalmatch',
 'usemailageemailiprisk_status',
 'usemailageemailiprisk_ip_risklevelid',
 'usemailageemailiprisk_fraud_risk',
 'usemailageemailiprisk_response_count',
 'usemailageemailiprisk_ip_cityconf',
 'usemailageemailiprisk__e_a_advice',
 'usemailageemailiprisk_domaincompany',
 'usemailageemailiprisk__e_a_status_i_d',
 'usemailageemailiprisk_phonecarriername',
 'usemailageemailiprisk_ip_risklevel',
 'usemailageemailiprisk__e_a_risk_band',
 'usemailageemailiprisk_country',
 'usemailageemailiprisk_uniquehits',
 'usemailageemailiprisk_ip_riskreasonid',
 'usemailageemailiprisk_citypostalmatch',
 'usemailageemailiprisk_ip_countryconf',
 'usemailageemailiprisk_ipcountrymatch',
 'usemailageemailiprisk_namematch',
 'usemailageemailiprisk_domainrelevantinfo',
 'usemailageemailiprisk_domainrisklevel',
 'usemailageemailiprisk_domainname',
 'usemailageemailiprisk_domaincountrymatch',
 'usemailageemailiprisk_ip_region',
 'usemailageemailiprisk_domain_creation_days',
 'usemailageemailiprisk_shipforward',
 'usemailageemailiprisk_ip_city',
 'usemailageemailiprisk__e_a_reason',
 'usemailageemailiprisk_domainrisklevel_i_d',
 'usemailageemailiprisk_domaincategory',
 'usemailageemailiprisk_ip_riskreason',
 'usemailageemailiprisk_ip_user_type',
 'usemailageemailiprisk_lookup_timestamp_ms_since_epoch',
 'usemailageemailiprisk_domainrelevantinfo_i_d',
 'usemailageemailiprisk__e_a_score',
 'usemailageemailiprisk_domaincountryname',
 'usemailageemailiprisk__e_a_reason_i_d',
 'usemailageemailiprisk_ip_isp',
 'usemailageemailiprisk_ipdistancemil',
 'usemailageemailiprisk__e_a_risk_band_i_d',
 'usemailageemailiprisk_ip_regionconf',
 'usemailageemailiprisk_domaincorporate',
 'usemailageemailiprisk_ip_metro_code',
 'usemailageemailiprisk_source_industry',
 'usemailageemailiprisk_email_exists',
 'usemailageemailiprisk_ip_corporate_proxy',
 'usemailageemailiprisk_retrievedfromcache',
 'usemailageemailiprisk_phonecarriertype',
 'usemailageemailiprisk_smfriends',
 'rvl_eid_estore_category',
 'rvl_eid_estore_id_fraud_cnt_30d',
 'rvl_eid_estore_category_bad_rate_12m',
 'rvl_eid_estore_id_orders_cnt_7d',
 'rvl_eid_estore_id_tot_tx_0_12m',
 'velocityvariables_apm_cids_on_shipping_email_1m',
 'velocityvariables_approved_apm_order_ids_on_cid_14d',
 'velocityvariables_shipping_full_names_on_account_id_14d',
 'velocityvariables_shipping_street_zip_on_order_id_14d',
 'velocityvariables_acp_on_cid_shipping_zip_1d',
 'velocityvariables_apm_amount_on_card_merchant_id_1d',
 'velocityvariables_apm_pad_cids_on_ip_zip_1h',
 'velocityvariables_card_countries_on_shipping_zip_1d',
 'velocityvariables_cids_on_shipping_email_2d',
 'velocityvariables_apm_order_ids_on_shipping_street_zip_1h',
 'velocityvariables_cards_on_shipping_email_1m',
 'velocityvariables_cids_on_shipping_email_1h',
 'velocityvariables_billing_emails_on_billing_phone_2d',
 'velocityvariables_apm_billing_emails_on_ip_1m',
 'velocityvariables_billing_emails_on_billing_phone_1h',
 'velocityvariables_card_numbers_on_cid_1h',
 'velocityvariables_apm_pad_on_shipping_country_street_zip_1h',
 'velocityvariables_apm_billing_emails_on_billing_street_zip_ip_2d',
 'velocityvariables_card_numbers_on_cid_7d',
 'velocityvariables_shipping_phones_on_account_id_14d',
 'velocityvariables_card_countries_on_shipping_zip_3d',
 'velocityvariables_gpm_on_order_id_1d',
 'velocityvariables_apm_cids_on_card_7d',
 'velocityvariables_apm_pad_cids_on_ip_zip_1d',
 'velocityvariables_shipping_street_zip_on_order_id_1h',
 'velocityvariables_acp_on_shipping_email_zip_1h',
 'velocityvariables_cids_on_order_id_7d',
 'velocityvariables_apm_cids_on_card_1d',
 'velocityvariables_apm_cids_on_card_1m',
 'velocityvariables_card_countries_on_shipping_zip_14d',
 'velocityvariables_apm_on_billing_phone_2d',
 'velocityvariables_apm_on_billing_phone_1h',
 'velocityvariables_apm_billing_emails_on_billing_street_zip_ip_1h',
 'velocityvariables_apm_emails_on_order_id_7d',
 'velocityvariables_is_error',
 'velocityvariables_account_ids_on_cid_14d',
 'velocityvariables_acp_on_ip_shipping_zip_1h',
 'velocityvariables_billing_emails_on_card_1h',
 'velocityvariables_apm_on_shipping_street_zip_ip_2d',
 'velocityvariables_cards_on_shipping_email_merchant_id_1h',
 'velocityvariables_card_numbers_on_cid_1d',
 'velocityvariables_acp_on_ip_shipping_zip_7d',
 'velocityvariables_acp_on_shipping_email_zip_7d',
 'velocityvariables_billing_emails_on_billing_phone_1m',
 'velocityvariables_billing_phones_on_account_id_14d',
 'velocityvariables_apm_on_shipping_street_zip_ip_1h',
 'velocityvariables_billing_emails_on_card_2d',
 'velocityvariables_cids_on_shipping_street_zip_ip_2d',
 'velocityvariables_cids_on_ip_1h',
 'velocityvariables_apm_cids_on_card_number_exp_year_exp_month_7d',
 'velocityvariables_shipping_full_name_on_order_id_14d',
 'velocityvariables_cids_on_shipping_street_zip_ip_1h',
 'velocityvariables_billing_full_names_on_account_id_14d',
 'velocityvariables_is_disabled',
 'velocityvariables_apm_cids_on_card_identifier_7d',
 'velocityvariables_shipping_full_name_on_order_id_1h',
 'velocityvariables_apm_card_cids_on_ip_zip_1h',
 'velocityvariables_acp_on_cid_shipping_zip_1h',
 'velocityvariables_order_ids_on_card_1h',
 'velocityvariables_order_ids_on_card_2d',
 'velocityvariables_billing_street_zip_on_order_id_14d',
 'velocityvariables_account_ids_on_billing_zip_14d',
 'velocityvariables_acp_on_ip_shipping_zip_1d',
 'velocityvariables_apm_order_ids_on_ip_2d',
 'velocityvariables_apm_billing_emails_on_billing_street_zip_ip_1m',
 'velocityvariables_apm_emails_on_cid_2d',
 'velocityvariables_apm_cids_on_order_id_7d',
 'velocityvariables_acp_on_shipping_email_zip_1d',
 'velocityvariables_acp_on_cid_shipping_zip_7d',
 'velocityvariables_billing_emails_on_card_7d',
 'velocityvariables_billing_street_zip_on_order_id_1h',
 'velocityvariables_apm_order_ids_on_ip_1h',
 'velocityvariables_order_ids_on_card_7d',
 'velocityvariables_cards_on_shipping_email_merchant_id_1m',
 'velocityvariables_apm_order_ids_on_shipping_email_cid_2d',
 'velocityvariables_apm_order_ids_on_shipping_email_cid_1h',
 'velocityvariables_apm_order_id_7d',
 'velocityvariables_apm_card_order_id_7d',
 'velocityvariables_apm_cids_on_card_number_exp_year_exp_month_1m',
 'velocityvariables_billing_full_name_on_order_id_1h',
 'velocityvariables_approved_apm_order_ids_on_cid_1m',
 'velocityvariables_cards_on_shipping_email_merchant_id_1d',
 'velocityvariables_apm_cids_on_card_number_exp_year_exp_month_1d',
 'velocityvariables_cards_on_shipping_email_test4_7d',
 'velocityvariables_approved_apm_order_ids_on_cid_1d',
 'velocityvariables_apm_cids_on_card_identifier_1m',
 'velocityvariables_apm_cids_on_card_identifier_1d',
 'velocityvariables_cids_on_shipping_street_zip_ip_1m',
 'velocityvariables_apm_cids_on_shipping_email_7d',
 'velocityvariables_billing_full_name_on_order_id_14d',
 'velocityvariables_apm_country_zip_street_2d',
 'velocityvariables_apm_billing_emails_on_ip_1h',
 'velocityvariables_apm_pad_cids_on_ip_zip_2d',
 'brms_number_of_articles',
 'brms_total_gift_card_count',
 'brms_is_returning_sibc',
 'brms_day_of_week',
 'brms_sum_paid',
 'brms_seconds_since_midnight',
 'brms_is_po_box',
 'brms_max_same_article_count',
 'brms_contains_electronic_gift_card',
 'brms_contains_gift_card',
 'brms_device',
 'brms_contains_nextdayship',
 'brms_total_gift_card_value',
 'brms_contains_hrshoes',
 'brms_shipping_equals_billing',
 'brms_is_firefox',
 'brms_resulting_balance_sibc_email',
 'brms_resulting_balance_sibc',
 'tu_most_negative_influence_on_vantagescore_factor1',
 'tu_most_negative_influence_on_vantagescore_factor2',
 'tu_most_negative_influence_on_vantagescore_factor3',
 'tu_most_negative_influence_on_vantagescore_factor4',
 'tu_vtg4_most_negative_influence_factor1',
 'tu_vtg4_most_negative_influence_factor2',
 'tu_vtg4_most_negative_influence_factor3',
 'tu_vtg4_most_negative_influence_factor4',
 'tu_vtg4',
 'tu_installment_trade_count',
 'tu_revolving_trade_count',
 'tu_mortgage_trade_count',
 'tu_unspecified_trade_count',
 'tu_total_trade_count',
 'tu_open_trade_count',
 'tu_negative_trade_count',
 'tu_historical_negative_trade_count',
 'tu_collection_count',
 'tu_total_balance_of_collections',
 'tu_historical_negative_occurrences_count',
 'tu_public_record_count',
 'tu_total_inquiry_count',
 'tu_paymnt10',
 'tu_paymnt11',
 'tu_rev12',
 'tu_rev122',
 'tu_rev205',
 'tu_rev322',
 'tu_revs124',
 'tu_g001s',
 'tu_g106s',
 'tu_g209s',
 'tu_g230s',
 'tu_g416s',
 'tu_g990s',
 'tu_re20s',
 'tu_re24s',
 'tu_re102s',
 'tu_trv01',
 'tu_trv02',
 'tu_cv05',
 'tu_cv14',
 'tu_s004s',
 'tu_se20s',
 'tu_s114s',
 'tu_rt20s',
 'tu_bkc11',
 'tu_bkc122',
 'tu_bkc322',
 'tu_ret201',
 'tu_in20s',
 'tu_aggs911',
 'tu_br101s',
 'tu_at01a',
 'tu_dob',
 'tu_no_currently_satisfactory_revolving_trades1',
 'tu_open_revolving_trades_verified_past12_months',
 'tu_open_trades_verified_past12_months',
 'tu_open_trades_verified_past12_months_excl_mortage',
 'tu_open_credit_card_trades_verified_past12_months',
 'tu_no_scheduled_monthly_payment_open_mortgage_trades',
 'tu_monthly_payment_on_open_or_closed_trades',
 'tu_no_bankruptcies_last24months',
 'tu_delinquent_trades_60days',
 'tu_identified_as_fraud_alert',
 'tu_militarty_active_duty',
 'tu_retrievedfromcache',
 'tu_not_enough_info',
 'tu_freeze',
 'rvl_cid_num_invoice_0_3m',
 'rvl_cid_days_since_last_purchase_0_3m',
 'rvl_cid_num_paid_inv_3_6m',
 'rvl_cid_num_klarna_direct_0_3m',
 'rvl_cid_has_paid_int',
 'rvl_cid_has_paid_credit',
 'rvl_cid_num_klarna_direct_0_6m',
 'rvl_cid_amount_last_purchase_0_3m',
 'rvl_cid_has_paid_acct_24m_int',
 'rvl_cid_max_num_failed_sibc_payment_attempts',
 'rvl_cid_has_account_int',
 'rvl_cid_has_paid_24m',
 'rvl_cid_num_paid_orders',
 'rvl_cid_has_paid_inv_int',
 'rvl_cid_sum_paid_inv_0_24m',
 'rvl_cid_account_amount_added_0_3m',
 'rvl_cid_has_paid_inv',
 'rvl_cid_has_paid_credit_int',
 'rvl_cid_num_paid_inv_12_24m',
 'rvl_cid_num_pix_0_6m',
 'rvl_cid_num_flexible_0_6m',
 'rvl_cid_num_direct_banking_0_6m',
 'rvl_cid_incoming_debt_incl_3y_sold',
 'rvl_cid_has_paid_12m',
 'rvl_cid_has_paid_credit_24m',
 'rvl_cid_has_paid_credit_12m',
 'rvl_cid_num_part_pay_0_6m',
 'rvl_cid_hours_since_1st_last_rej_0_14d',
 'rvl_cid_num_flexible_0_3m',
 'rvl_cid_has_paid_inv_12m',
 'rvl_cid_has_paid_acct_int',
 'rvl_cid_num_pix_0_3m',
 'rvl_cid_num_direct_banking_0_3m',
 'rvl_cid_num_dist_emails_2d_24m',
 'rvl_cid_has_paid_credit_12m_int',
 'rvl_cid_num_dist_emails_2d_6m',
 'rvl_cid_incoming_invoice_debt_incl_3y_sold',
 'rvl_cid_has_paid_inv_24m',
 'rvl_cid_account_amount_added_6_12m',
 'rvl_cid_has_paid_24m_int',
 'rvl_cid_account_recovery_int',
 'rvl_cid_has_paid_12m_int',
 'rvl_cid_num_part_pay_0_3m',
 'rvl_cid_num_dist_emails_2d_2w',
 'rvl_cid_payment_method_2nd_last_purchase_0_3m',
 'rvl_cid_num_unpaid_bills',
 'rvl_cid_cid_age',
 'rvl_cid_has_paid_acct_24m',
 'rvl_cid_num_dist_emails_2d_3m',
 'rvl_cid_account_amount_added_3_6m',
 'rvl_cid_num_converted_invoices_0_6m',
 'rvl_cid_num_elv_0_6m',
 'rvl_cid_num_paid_inv_6_12m',
 'rvl_cid_num_credit_card_0_6m',
 'rvl_cid_has_history',
 'rvl_cid_has_paid_credit_24m_int',
 'rvl_cid_settled_amount',
 'rvl_cid_payment_method_last_purchase_0_3m',
 'rvl_cid_has_rejection_14d_int',
 'rvl_cid_incoming_debt',
 'rvl_cid_num_converted_invoices_0_3m',
 'rvl_cid_has_2nd_rejection_14d',
 'rvl_cid_has_paid_inv_12m_int',
 'rvl_cid_num_elv_0_3m',
 'rvl_cid_oldest_pstatus_active_inv',
 'rvl_cid_has_paid_acct_12m',
 'rvl_cid_account_debt',
 'rvl_cid_num_credit_card_0_3m',
 'rvl_cid_account_amount_added_12_24m',
 'rvl_cid_incoming_account_debt',
 'rvl_cid_has_paid_acct',
 'rvl_cid_num_unpaid_orders',
 'rvl_cid_has_paid_acct_12m_int',
 'rvl_cid_num_invoice_0_6m',
 'rvl_cid_sum_paid_inv_12_24m',
 'rvl_cid_invoice_debt',
 'rvl_cid_has_paid',
 'rvl_cid_has_paid_inv_24m_int',
 'rvl_cid_email_has_paid_int',
 'rvl_domain_bad_rate_12m',
 'rvl_zip3_bad_moving_count_12m',
 'rvl_incoming_household_debt',
 'rvl_num_dist_billing_phones_on_email_2d_2w',
 'rvl_cid_zip_has_paid_int',
 'rvl_country_bad_count_coarse_12m',
 'rvl_num_dist_billing_addresses_on_email_2d_1w',
 'crid_realtime_incoming_invoice_debt',
 'crid_realtime_incoming_sibc_debt',
 'crid_realtime_num_unpaid_sibc',
 'payment_method_downpayment',
 'payment_method_number_of_installments',
 'payment_method_card_type',
 'payment_method_card_level',
 'payment_method_card_issuing_bank',
 'payment_method_card_exp_month',
 'payment_method_card_brand',
 'payment_method_credit_card_3ds_supported',
 'usemailageemailiprisk_lastflaggedon_days',
 'usemailageemailiprisk_first_verification_date_days',
 'usemailageemailiprisk_last_verification_date_days',
 'unpaid_at_60_rate']