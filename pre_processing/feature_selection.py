from sklearn.feature_selection import SelectKBest, f_classif


def feature_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame):
    """ 
    Feature selection method to reduce the number of input features to those that are believed to be most useful to a model in order to predict the target variable.
    ----------
    X_train : Input variables of the training set
    y_train: Target variables of the training set
    X_test : Input variables of the test set
    Returns : The training set and test set without the non-important features. 

    """

    # Select K best = Select features according to the k highest scores. f_classif = ANOVA F-value between label/feature
    test = SelectKBest(f_classif, k=8) 
    X_train_fs = test.fit_transform(X_train, y_train) 
    X_test_fs = test.transform(X_test)
    
    # Get columns and create daframe with those
    new_cols = test.get_support(indices=True)
    X_train_fs = X_train.iloc[:,new_cols]
    X_test_fs = X_test.iloc[:,new_cols]
    
    return X_train_fs, X_test_fs