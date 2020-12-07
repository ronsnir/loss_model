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
    if exeptions in na_df['variable'].values:
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

def num_to_none(df, replacements):
    # Create list of numeric columns
    numeric_list = df.select_dtypes(include=[np.number]).columns[(df.select_dtypes(include=[np.number]) < 0).any()].tolist()
    # replace negative with none
    df[numeric_list] = df[numeric_list].replace(replacements)
    #Return
    return df