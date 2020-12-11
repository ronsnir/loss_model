# %%
# Standard library imports
import pandas as pd
import numpy as np
from datetime import datetime
import maya as maya
from dateutil.relativedelta import relativedelta

# Third party imports


# Local application imports
import loss_model


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


# Calculate the days between two dates
def days_diff(date1,date2):
    return (date1-date2).dt.days


# Create age column and add it to the df
def years_diff(df,today,years_diff_list):
    for y in years_diff_list:
        age_list = []
        for i in df[y].index.to_list():
            if pd.isnull(df[y].loc[i]):
                age_val = None
            else:
                age_val = relativedelta(df[today].loc[i],df[y].loc[i]).years
            
            age_list.append(age_val)

        age_df = pd.DataFrame(age_list, columns=[y + '_years'])
        df = pd.concat([df,age_df], axis=1)
        df = df.drop(columns=y)
    
    return df


# Replace negative values with None
def num_to_none(df, replacements):
    # Create list of numeric columns
    numeric_list = df.select_dtypes(include=[np.number]).columns[(df.select_dtypes(include=[np.number]) < 0).any()].tolist()
    # replace negative with none
    df[numeric_list] = df[numeric_list].replace(replacements)
    #Return
    return df


# Is shopping function
def shopping_col(df, merchant_id_col):
    df['is_shopping'] = (df[merchant_id_col]=='N101065').astype(int)
    df = df.drop(columns=merchant_id_col)
    return df


# Is shopping function
def classifier_column(df, original_column_name:str, condition_value, new_column_name:str, is_bigger_condition: bool=True, drop_the_original: bool=False):
    
    if is_bigger_condition==True: # If we want the value to be bigger then something
        # Create the new column
        df[new_column_name] = (df[original_column_name]>condition_value).astype(int)
        if drop_the_original==True: # If we want to drop the original column
            # Drop the original column
            df = df.drop(columns=original_column_name)
    else: # If we want the value to be bigger equal to something
        df[new_column_name] = (df[original_column_name]==condition_value).astype(int)
        if drop_the_original==True: # If we want to drop the original column
            # Drop the original column
            df = df.drop(columns=original_column_name)
    # Return
    return df



# Function that keep only columns that still exist (after all filtering)
def filtered_column_list(df, col_list):
    ## Convert the categorical_variables to np.array
    small_col_list = np.array(col_list)
    ## Get the total columns array
    big_columns_list = df.columns
    ## Get boolean array for keeping the variables
    bool_filter_list = np.array(np.isin(big_columns_list, small_col_list), dtype=bool)
    ## Get the column list after filtering
    filtered_col_list = list(big_columns_list[bool_filter_list])

    ## Return
    return filtered_col_list


