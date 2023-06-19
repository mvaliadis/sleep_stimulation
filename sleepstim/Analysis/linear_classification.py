#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:58:43 2023

@author: administrator
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

## OLS regression [𝑤=(𝑋𝑇𝑋)−1𝑋𝑇𝑦]
def plot_regression_results(y_test,y_pred,weights):
    '''Produces three plots to analyze the results of linear regression:
        -True vs predicted
        -Raw residual histogram
        -Weight histogram
        
    Inputs:
        y_test: (n_observations,) numpy array with true values
        y_pred: (n_observations,) numpy array with predicted values
        weights: (n_weights) numpy array with regression weights'''
    
    print('MSE: ', mean_squared_error(y_test,y_pred))
    print('r^2: ', r2_score(y_test,y_pred))
    
    fig,ax = plt.subplots(1,3,figsize=(9,3))
    #predicted vs true
    ax[0].scatter(y_test,y_pred)
    ax[0].set_title('True vs. Predicted')
    ax[0].set_xlabel('True %s' % (target_clm))
    ax[0].set_ylabel('Predicted %s' % (target_clm))

    #residuals
    error = np.squeeze(np.array(y_test)) - np.squeeze(np.array(y_pred))
    ax[1].hist(np.array(error),bins=30)
    ax[1].set_title('Raw residuals')
    ax[1].set_xlabel('(true-predicted)')

    #weight histogram
    ax[2].hist(weights,bins=30)
    ax[2].set_title('weight histogram')

    plt.tight_layout()
    
def sklearn_regression(X_test, X_train, y_train):
    '''Computes OLS weights for linear regression without regularization using the sklearn library on the training set and 
       returns weights and testset predictions.
    
       Inputs:
         X_test: (n_observations, 81), numpy array with predictor values of the test set 
         X_train: (n_observations, 81), numpy array with predictor values of the training set
         y_train: (n_observations,) numpy array with true target values for the training set
         
       Outputs:
         weights: The weight vector for the regerssion model including the offset
         y_pred: The predictions on the TEST set
          
         
       Note:
         The sklearn library automatically takes care of adding a column for the offset.     
    
    '''
    
    # initialize model
    lm_model = linear_model.LinearRegression().fit(X_train,y_train)
    # extract weights/coefficients + intercept
    weights = list(lm_model.coef_[0])
    weights.append(lm_model.intercept_[0])
    # predict y
    y_pred = lm_model.predict(X_test)
    
    return weights, y_pred

def ridge_regression_sklearn(X_test, X_train, y_train,alpha):
    '''Computes OLS weights for regularized linear regression with regularization strength alpha using the sklearn
       library on the training set and returns weights and testset predictions.
    
       Inputs:
         X_test: (n_observations, 81), numpy array with predictor values of the test set 
         X_train: (n_observations, 81), numpy array with predictor values of the training set
         y_train: (n_observations,) numpy array with true target values for the training set
         alpha: scalar, regularization strength
         
       Outputs:
         weights: The weight vector for the regerssion model including the offset
         y_pred: The predictions on the TEST set
          
       Note:
         The sklearn library automatically takes care of adding a column for the offset.     
   
    
    '''
    
    # initialize model
    rr_model = linear_model.Ridge(alpha).fit(X_train,y_train)
    # extract weights/coefficients + intercept
    weights = list(rr_model.coef_[0])
    weights.append(rr_model.intercept_[0])
    # predict y
    y_pred = rr_model.predict(X_test)
            
    return weights, y_pred

def ridgeCV(X, y, n_folds, alphas):
    '''Runs a n_fold-crossvalidation over the ridge regression parameter alpha. 
       The function should train the linear regression model for each fold on all values of alpha.
    
      Inputs: 
        X: (n_obs, n_features) numpy array - predictor
        y: (n_obs,) numpy array - target
        n_folds: integer - number of CV folds
        alphas: (n_parameters,) - regularization strength parameters to CV over
        
      Outputs:
        cv_results_mse: (n_folds, len(alphas)) numpy array, MSE for each cross-validation fold 
        
      Note: 
        Fix the seed for reproducibility.
        
        '''    
    
    cv_results_mse = np.zeros((n_folds, len(alphas)))
    np.random.seed(seed=2)
    
    ###
    idx = list(range(X.shape[0])) # list of all observations
    l_idx = len(idx) # number of observations
    fold_size = int(l_idx / n_folds) # calculate the fold size
    for fold in range(n_folds): #for each fold
        #pick the current fold indices by choosing fold_size number of elements from idx list without replacement
        i_test = np.random.choice(idx, fold_size, replace=False)
        # update the idx list such that the ones that are used as the test fold are not included anymore. 
        # The remaining will be assigned to the test fold later.
        idx = [x for x in idx if x not in i_test] 
        # add any observation that is not in the current test fold to the train fold
        i_train = [i for i in range(l_idx) if i not in i_test]         
        X_test, y_test = X[i_test,:], y[i_test,:] #get test X and y
        X_train, y_train = X[i_train,:], y[i_train,:] #get train X and y
        
        #loop through all possible alphas
        for idx_alpha, alpha in enumerate(alphas): 
            # get predictions 
            _, y_pred = ridge_regression_sklearn(X_test, X_train, y_train, alpha) 
            mse = mean_squared_error(y_test,y_pred) # get mse
            cv_results_mse[fold, idx_alpha] = mse # add mse to the loss array
            
    return cv_results_mse    