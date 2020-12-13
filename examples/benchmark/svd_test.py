# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:36:56 2020

@author: suyu
"""

from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise import Reader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,roc_auc_score,mean_absolute_error,log_loss
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from gammli.DataReader import data_initialize

def svd(wc, data, meta_info_ori,SVD_type="SVD", task_type="Regression", random_state=0):
    
    
    if SVD_type == "SVD":
        base = SVD(n_factors=3)
    elif SVD_type == "SVDpp":
        base = SVDpp(n_factors=3, verbose=True)
    elif SVD_type == "PMF":
        base = SVD(n_factors=3, biased=False)
        
    cold_mae = []
    cold_rmse = []
    warm_mae = []
    warm_rmse = []
    cold_auc = []
    cold_logloss = []
    warm_auc = []
    warm_logloss = []
        
    for j in range(10):
        
        train, test = train_test_split(data, test_size=0.2, random_state=j)
        tr_x, tr_Xi, tr_y, tr_idx, te_x, te_Xi, te_y, val_x, val_Xi, val_y, val_idx, meta_info, model_info, sy, sy_t = data_initialize(train, test, meta_info_ori, task_type, 'warm', random_state=0, verbose=False)           
        
        #Xi = tr_x[:,-2:]
        #Xi_t = te_x[:,-2:]
        Xi = tr_Xi
        Xi_t = te_Xi
        
        tr_ratings_dict = {'itemID': Xi[:,1].tolist(),
                'userID': Xi[:,0].tolist(),
                'rating': tr_y.ravel().tolist()}

        tr_df = pd.DataFrame(tr_ratings_dict)
        reader = Reader(rating_scale=(tr_y.min(), tr_y.max()))
        
        tr_data = Dataset.load_from_df(tr_df[['userID', 'itemID', 'rating']], reader)
        tr_data = tr_data.build_full_trainset()
        base.fit(tr_data)

        pred = []
        
        for i in range(Xi_t.shape[0]):
            pred.append(base.predict(Xi_t[i,0],Xi_t[i,1]).est)
    
        pred2 = np.array(pred).reshape([-1, 1])
            
        if task_type == "Regression":
            pred2 = sy.inverse_transform(pred2.reshape(-1,1))
            te_y = sy_t.inverse_transform(te_y.reshape(-1,1))
            if wc == 'warm':
                if len([(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')])!=1:
                    warm_y = te_y[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                    warm_pred = pred2[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                else:
                    warm_y = te_y
                    warm_pred= pred2
                
                
                warm_mae.append(mean_absolute_error(warm_y,warm_pred))
                warm_rmse.append(mean_squared_error(warm_y,warm_pred)**0.5)
                
            if wc == 'cold':
                try:
                    [(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')] != [True]
                    print('no cold samples')
                    continue
                except:
                    cold_y = te_y[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                    cold_pred = pred2[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]

                cold_mae.append(mean_absolute_error(cold_y,cold_pred))
                cold_rmse.append(mean_squared_error(cold_y,cold_pred)**0.5)

            if wc == 'warm':
            
                i_result = np.array(['SVD',np.mean(warm_mae),np.mean(warm_rmse),np.std(warm_mae),np.std(warm_rmse)]).reshape(1,-1)
                result = pd.DataFrame(i_result,columns=['model','warm_mae','warm_rmse','std_warm_mae','std_warm_rmse'])

            if wc == 'cold':
            
                i_result = np.array(['SVD',np.mean(cold_mae),np.mean(cold_rmse),np.std(cold_mae),np.std(cold_rmse)]).reshape(1,-1)
                result = pd.DataFrame(i_result,columns=['model','cold_mae','cold_rmse','std_cold_mae','std_cold_rmse'])

    
        elif task_type == "Classification":

            
            if wc == 'warm':
                
                if len([(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')])!=1:
                    warm_y = te_y[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                    warm_pred = pred2[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                else:
                    warm_y = te_y
                    warm_pred= pred2
                warm_auc.append(roc_auc_score(warm_y,warm_pred))
                warm_logloss.append(log_loss(warm_y,warm_pred))
                
            if wc == 'cold':
                
                try:
                    [(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')] != [True]
                    print('no cold samples')
                    continue
                except:
                    cold_y = te_y[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                    cold_pred = pred2[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                    
                cold_auc.append(roc_auc_score(cold_y,cold_pred))
                cold_logloss.append(log_loss(cold_y,cold_pred))
 
            if wc == 'warm':
                i_result = np.array(['SVD',np.mean(warm_auc),np.mean(warm_logloss),np.std(warm_auc),np.std(warm_logloss)]).reshape(1,-1)
                result = pd.DataFrame(i_result,columns=['model','warm_auc','warm_logloss','std_warm_auc','std_warm_logloss'])

            if wc == 'cold':
                i_result = np.array(['SVD',np.mean(cold_auc),np.mean(cold_logloss),np.std(cold_auc),np.std(cold_logloss)]).reshape(1,-1)
                result = pd.DataFrame(i_result,columns=['model','cold_auc','cold_logloss','std_cold_auc','std_cold_logloss'])
            

        return result, tr_df        
