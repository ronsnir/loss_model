# %%
# Train a model function
def tree_model_train(model_type:str, model_name:str, constant_params, categorical_variables, X_train, y_train, X_validation, y_validation, months:int, year:str, dum:str = ''):
    import joblib
    import numpy as np

    ## Convert the categorical_variables to np.array
    categorical_variables = np.array(categorical_variables)
    ## Get the total columns array
    columns_list = X_train.columns
    ## Get boolean array for keeping the variables
    bool_filter_list = np.array(np.isin(columns_list, categorical_variables), dtype=bool)
    ## Get the column list after filtering
    filtered_col_list = columns_list[bool_filter_list]

    if model_type == 'classification': # If it's classification model
        if model_name == 'xgb': # xgboost
            from xgboost import XGBClassifier
            model = XGBClassifier(**constant_params)

            #fit the model
            model.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_validation, y_validation)],
                    eval_metric='auc',
                    verbose=True)

            #save the model
            joblib.dump(model, 'model_'+str(months)+'_months_'+year+'_'+dum+'_'+model_name)

            return model

        elif model_name == 'catboost': # catboost
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(**constant_params)

            #fit the model
            model.fit(X_train, y_train,cat_features=filtered_col_list,eval_set=(X_validation, y_validation),plot=True)

            #save the model
            joblib.dump(model, 'model_'+str(months)+'_months_'+year+'_'+dum+'_'+model_name)

            return model

    elif model_type == 'regression': # If it's regression model
        if model_name == 'xgb': # xgboost
            from xgboost import XGBRegressor
            model = XGBRegressor(**constant_params)

            #fit the model
            model.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_validation, y_validation)],
                    eval_metric='auc',
                    verbose=True)

            #save the model
            joblib.dump(model, 'model_'+str(months)+'_months_'+year+'_'+dum+'_'+model_name)

            return model

        elif model_name == 'catboost': # catboost
            from catboost import CatBoostRegressor

            model = CatBoostRegressor(**constant_params)

            #fit the model
            model.fit(X_train, y_train,cat_features=filtered_col_list,eval_set=(X_validation, y_validation),plot=True)

            #save the model
            joblib.dump(model, 'model_'+str(months)+'_months_'+year+'_'+dum+'_'+model_name)

            return model