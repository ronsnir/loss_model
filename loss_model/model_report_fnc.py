# %%
#build the report function
def model_report(model, model_name, threshold, X_train, X_test, y_train, y_validation, y_test):
    from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix,precision_score,recall_score,accuracy_score
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    import pandas as pd
    #check the target distribution for all df's
    print("The return rate of y_train is: ", round(y_train.mean(),4))
    print("The return rate of y_validation is: ", round(y_validation.mean(),4))
    print("The return rate of y_test is: ", round(y_test.mean(),4))

    #calculate probabilities of the test df
    probs = model.predict_proba(X_test)[:,1]

    #calculate the classes of the test df
    probs_class = (probs >= threshold).astype('int')

    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, probs)

    # calculate precision-recall AUC
    pr_auc_score = auc(recall, precision)

    # plot the precision-recall curves
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label=model_name)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

    ###ROC###
    # create range of 0 for ns (no skill) to compare the ROC to
    ns_probs = [0 for _ in range(len(y_test))]
    #get the roc value
    roc_value = roc_auc_score(y_test,probs)
    #create values for thr ROC AUC plot
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    ns_fpr, ns_tpr, thresholds = roc_curve(y_test, ns_probs)
    #plot formatting
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(11,7))
    #plot the roc curve for the model
    plt.plot(ns_fpr,ns_tpr, linestyle='--',label='No Skill')
    plt.plot(fpr,tpr,marker='.',label=model_name+' (area = %0.2f)'%roc_value)
    #axis lables
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #show the legend
    plt.legend()
    #show the plot
    plt.show()

    #model evaluation
    print("=== CatBoost ===")
    print('\n')
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, probs_class))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_test, probs_class))
    print('\n')
    print("=== ROC AUC Scores ===")
    print(roc_value)
    print('\n')
    print("=== PR AUC Score ===")
    print(pr_auc_score)

    #create dataframe for the probabilities distribution
    pred_df = pd.DataFrame({'true': y_test, 'prob': probs}, columns=['true', 'prob'])
    #plot probabilities distribution
    pred_df.groupby("true").prob.plot(kind='hist'
                                    # , density = True
                                    , bins = 500
                                    , alpha=0.4
                                    , figsize=(14, 8)
                                    , legend = True
                                    # , xlim = (0,0.2)
                                    # , xticks = np.arange(0,0.2125,0.025)
                                    )
    plt.xlabel('Predicted Probabilities')

    print('\n')

    # #calculate the shap values
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X_train)

    # #shap bar plot
    # shap.summary_plot(shap_values, X_train, plot_type="bar")

    # #shap plot
    # shap.summary_plot(shap_values, X_train)


    # # reliability diagram
    # fop, mpv = calibration_curve(y_test, probs, n_bins=10)
    # # plot perfectly calibrated
    # plt.plot([0, 1], [0, 1], linestyle='--')
    # # plot model reliability
    # plt.plot(mpv, fop, marker='.')
    # plt.show()

    return probs