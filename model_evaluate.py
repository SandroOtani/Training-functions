from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from math import sqrt

def model_evaluation(ytest, ypred, model_type, nd=2):
    '''
    Function to calculate model metrics    
    Input
        ytest: vector representing the label of each sample
        ypred: vector composed of values predicted by a model 
        model_type: defines the problem ('Regression' or 'Classification')
    Output
        for a classification problem:
            conf_matrix: confusion matrix
            acc: accuracy
            prec: precision
            sens: sensitivity
            spec: specificity
            NPV: Negative predictive value
            PPV: Positive predictive value
            f1_score: f1_score
        for a regression problem:
            RMSE: Root mean square error
            R2: Coefficient of determination
            MAE: Mean absolute error
    Sandro K. Otani 11/21
    '''
    if model_type == 'Classification':
        conf_matrix = confusion_matrix(ytest,ypred)
        TN = conf_matrix[0,0]
        FN = conf_matrix[1,0]
        TP = conf_matrix[1,1]
        FP = conf_matrix[0,1]
        acc = np.round((TP+TN)/(TP+TN+FP+FN),nd)
        prec = np.round(TP/(TP + FP),nd)
        sens = np.round(TP/(TP + FN),nd)
        spec = np.round(TN/(TN + FP),nd)
        NPV = np.round(TN/(TN + FN),nd)
        PPV = np.round(TP/(TP+FP),nd)
        f1_score = np.round((2*sens*prec)/(sens+prec),nd)
        display(pd.DataFrame([acc,prec,sens,spec,NPV,PPV,f1_score],index=['Accuracy','Precision','Sensitivity','Specificity','NPV','PPV','F1-score']).T)
        return conf_matrix, acc, prec, sens, spec, NPV, PPV, f1_score
    elif model_type == 'Regression':
        MSE = np.square(np.subtract(ytest,ypred)).mean()
        MAE = np.round(np.abs(np.subtract(ytest,ypred)).mean(),nd)
        RMSE= np.round(sqrt(MSE),nd)
        corr_matrix = np.corrcoef(ytest, ypred)
        R = corr_matrix[0,1]
        R2 = np.round(R**2,nd)
        display(pd.DataFrame([RMSE, R2, MAE],index=['Root mean square error','Coefficient of determination','Mean absolute error']))
        return RMSE, R2, MAE