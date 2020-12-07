# Standard library imports
import pandas as pd
import numpy as np

# Third party imports


# Local application imports
import loss_model


# Load the data
df = loss_model.load_the_data(loss_model.path)

# Processing

## Drop columns
df = df.drop(loss_model.to_drop_list_beginning, axis=1)

## Convert to datetime
for i in loss_model.date_col_list:
    df[i] = pd.to_datetime(df[i], utc=True)

#Days diff calculation
for i in loss_model.days_diff_list:
    df[i + '_days'] = loss_model.days_diff(df['decision_time'], df[i])

## Years diff calculation
df = loss_model.years_diff(df=df,today=loss_model.today_col,years_diff_list=loss_model.years_diff_list)

## Replace negative values (numeric) with None
df = loss_model.num_to_none(df=df, replacements=loss_model.neg_to_none_dict)
## Replace negative values (strings) with None
df[loss_model.neg_to_none_col_string_list] = df[loss_model.neg_to_none_col_string_list].replace(loss_model.neg_to_none_string_dict)

## Remove NA's for high share of NA's
df = loss_model.remove_full_na(df = df, na_share_threshold = loss_model.na_share_threshold, exeptions = loss_model.exeptions_list)
## Fill NA's with 'UNKNOWN'
df[loss_model.na_to_unknown_list] = df[loss_model.na_to_unknown_list].fillna(value='UNKNOWN')
## Fill NA's with 'Other'
df[loss_model.na_to_other_list] = df[loss_model.na_to_other_list].fillna(value='Other')
