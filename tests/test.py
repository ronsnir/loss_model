from loss_model import load_the_data, remove_full_na, na_share_threshold, path, line_step, date_col_list, days_diff, days_diff_list, neg_to_zero_col_list, neg_to_zero_dict, neg_to_none_col_list, neg_to_none_dict, years_diff_list, years_diff, today_col, exeptions_list, num_to_none, neg_to_none_string_dict, neg_to_none_col_string_list, na_to_unknown_list, na_to_other_list, to_drop_list_beginning
import pandas as pd

def main():
    #load the data
    df = load_the_data(path, line_step, is_test=True)
    #get small data set
    # df = df.head(1000)

    #drop columns
    df = df.drop(to_drop_list_beginning, axis=1)

    #convert to datetime
    for i in date_col_list:
        df[i] = pd.to_datetime(df[i], utc=True)
    
    #days diff calculation
    for i in days_diff_list:
        df[i + '_days'] = days_diff(df['decision_time'], df[i])
    
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

    #remove na
    df = remove_full_na(df = df, na_share_threshold = na_share_threshold, exeptions = exeptions_list)

    ## Fill NA's with 'UNKNOWN'
    df[na_to_unknown_list] = df[na_to_unknown_list].fillna(value='UNKNOWN')
    ## Fill NA's with 'Other'
    df[na_to_other_list] = df[na_to_other_list].fillna(value='Other')

    #create test df
    df_test = df[['decision_time', 'consumer_date_of_birth_years', 'rvl_cid_payment_method_last_purchase_0_3m']]

    #
    print(df_test.head())
    print(len(df_test))

if __name__ == "__main__":
    main()