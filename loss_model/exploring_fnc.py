# %%
import pandas as pd


def na_list_fnc(df):
    na_df = pd.DataFrame((100 * df.isnull().sum() / len(df)).sort_values(ascending = False), columns=['na_share']).reset_index().rename(columns={'index': 'variable'})
    na_df = na_df[na_df['na_share'] > 0]
    return na_df


def na_col_fnc(df, col):
    na_share = (100 * df[col].isnull().sum() / len(df))
    return na_share


def custom_summary(df,col):
    # get the NA's share for the specific column
    na_share = na_col_fnc(df=df, col=col)
    # get the describe() and value_counts()
    des = df[col].describe()
    vc = df[col].value_counts()
    # print the results
    print('======================')
    print(col)
    print('\n')
    print(f"The share of NA's is {round(na_share,2)}%")
    print('\n')
    print(des)
    print('\n')
    print(vc)
    print('\n'*3)


class na_list_cls():
    def __init__(self, df):
        self.na_list = pd.DataFrame((100 * df.isnull().sum() / len(df)).sort_values(ascending = False), columns=['na_share']).reset_index().rename(columns={'index': 'variable'})


    def na_list(self):
        return self.na_list
    
    
    def col_na_share(self, col):
        return self.na_list[col]