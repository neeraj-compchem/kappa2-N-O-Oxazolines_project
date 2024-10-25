from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def forward_stepwise_model_selector(X_train, y_train, max_features=5):
    """
    Performs a forward stepwise linear regression model search, 
    returns the model performance (maximum no. of features is 5 by default)
    """
    
    features = list(range(X_train.shape[1]))  # feature indices
    column_names = X_train.columns  

    selected_features = []  # to store the selected feature indices
    remaining_features = features.copy()  # remaining features to be evaluated
    models_performance = []  # to store the performance metrics

    # for cross-validations
    loo = LeaveOneOut()
    kfold = KFold(n_splits=3)

    # variables
    best_model = None
    best_subset_names = None
    final_y_train_loo = None
    final_X_train_subset = None

    for step in range(min(max_features, len(features))):  # main loop for best feature selection
        best_performance = None
        best_feature = None
        best_subset = None

        for feature in remaining_features:  # sub loop for adding next features
            current_subset = selected_features + [feature]
            subset_names = [column_names[i] for i in current_subset]  

            X_train_subset = X_train[subset_names] 

            # model creation and training
            model = LinearRegression()
            model.fit(X_train_subset, y_train)

            y_train_pred = model.predict(X_train_subset)  # prediction

            # metrics calculation for train samples
            mae_train = mean_absolute_error(y_train, y_train_pred)
            r2_train = r2_score(y_train, y_train_pred)

            # cross-validations
            y_train_loo = cross_val_predict(model, X_train_subset, y_train, cv=loo)
            mae_loo = mean_absolute_error(y_train, y_train_loo)
            y_train_3fold = cross_val_predict(model, X_train_subset, y_train, cv=kfold)
            mae_3fold = mean_absolute_error(y_train, y_train_3fold)

            # to check if current model has the best fit so far
            if best_performance is None or mae_train < best_performance:
                best_performance = mae_train
                best_feature = feature
                best_model = model
                best_subset = current_subset
                best_subset_names = subset_names
                final_y_train_loo = y_train_loo
                final_X_train_subset = X_train_subset

        # to update the feature selection
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        models_performance.append((mae_train, r2_train, mae_loo, mae_3fold, best_model, best_subset, [column_names[i] for i in best_subset]))  # to store model performance at this stage

        print(f"Step {step + 1}:")
        print(f"Selected feature: {column_names[best_feature]}")
        print(f"Train MAE: {mae_train}, Train R2: {r2_train}")
        print(f"LOO MAE: {mae_loo}, 3-fold CV MAE: {mae_3fold}")
        print(f"Selected features: {', '.join([column_names[i] for i in best_subset])}")
        print(f"Coefficients: {best_model.coef_}")
        print(f"Intercept: {best_model.intercept_}" + '\n')

        # to terminate if max features reached
        if len(selected_features) >= max_features:
            break



    return models_performance

