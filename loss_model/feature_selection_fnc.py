# %%
# Standard library imports
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression,f_classif

# Third party imports


# Local application imports
import loss_model


# The functions:

## Correlation selection function
def cor_selection(df,target_col, exeptions_list):
    # Take out exeption columns from the df before removing the hige corr columns
    
    ## Convert the expetions_list to np.array
    exeptions_list = np.array(exeptions_list)
    ## Get the total columns array
    columns_list = df.columns
    ## Get boolean array for either to keep the variables or to filter them out
    bool_filter_list = np.array(~np.isin(columns_list, exeptions_list), dtype=bool)
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
        df_for_f_classif = pd.concat([df[col], df[corr_columns_list], df[target_col]], axis=1).dropna()
        if len(corr_columns_list) > 0:
            # Get the target and features after filtering NA's
            target = df_for_f_classif.iloc[: , len(df_for_f_classif.columns)-1:]
            features = df_for_f_classif.iloc[: , 0:len(df_for_f_classif.columns)-1]
            # Get the f-values of the test
            f_classif_result,_ = f_classif(features, target)
            # Get the feature location of the best score
            the_best_feature = f_classif_result.argmax()
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









