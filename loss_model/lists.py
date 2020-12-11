import numpy as np

# The path to load the data
path = '/home/ron.snir/git/loss_model/data/pi4_full000.csv'

# Columns to convert to datetime
date_columns_list = [
    'usemailageemailiprisk_lastflaggedon'
    , 'decision_time'
]

# The model type and name
tree_model_type = 'classification'
tree_model_name = 'catboost'

# Catbood parameters
constant_params_catboost = {'iterations': 1000,
                    'random_seed': 101,
                    'learning_rate': 0.1,
                    'eval_metric': 'AUC',
                    'early_stopping_rounds': 20}

# Ther target column
target_col = 'is_default'

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
    , 'usemailageemailiprisk_ip_postalcode'
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
    , 'usemailageemailiprisk_ip_postalcode'
]