# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 10:08:35 2020

@author: suyu
"""

import numpy as np
import pandas as pd 
import tensorflow as tf
from sklearn.metrics import mean_squared_error,roc_auc_score,mean_absolute_error,log_loss
from sklearn.preprocessing import MinMaxScaler
import sys

from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
sys.path.append('../')
from gammli.DataReader import data_initialize
sys.path.append('benchmark/deepfm' )
from fmDataReader import FeatureDictionary, DataParser
from DeepFM import DeepFM


def deepfm_fm(wc, data, meta_info_ori, task_type="Regression", random_state=0, params=None):
    
    train, test = train_test_split(data, test_size=0.2, random_state=0)
    tr_x, tr_Xi, tr_y, tr_idx, te_x, te_Xi, te_y, val_x, val_Xi, val_y, val_idx, meta_info, model_info, sy, sy_t = data_initialize(train, test, meta_info_ori, task_type, 'warm', random_state=0, verbose=True) 
                    
    epochs = params['epochs']
    loss_type = params['loss_type']
    eval_metric = params['eval_metric']
    greater_is_better = params['greater_is_better']
    verbose = params['verbose']
    early_stopping = params['early_stopping']
    
    
    NUMERIC_COLS = []
    CATEGORICAL_COLS = []
    IGNORE_COLS = []
    
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "categorical":
            if len(meta_info[key]['values']) ==2:
                NUMERIC_COLS.append(key)
            else:
                CATEGORICAL_COLS.append(key)
        elif item['type'] == "target":
            IGNORE_COLS.append(key)
        else:
            NUMERIC_COLS.append(key)
        
    
    # params
    dfm_params = {
            "embedding_size": 3,
            "deep_layers": [32, 32],
            "use_deep" : True ,
            "use_fm" : True , 
            "deep_layers_activation": tf.nn.relu,
            "loss_type" : loss_type,
            "epoch": epochs ,
            "batch_size": 1000,
            "learning_rate": 0.001,
            "optimizer_type": "adam",
            "batch_norm": 0,
            "batch_norm_decay": 0.995,
            "l2_reg": 0.1,
            "greater_is_better" : greater_is_better,
        "verbose": verbose,
        "eval_metric": eval_metric,
        "random_seed": random_state
        }
    
    def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
        fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=NUMERIC_COLS,
                           ignore_cols=IGNORE_COLS)
        data_parser = DataParser(feat_dict=fd)
        Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
        Xi_test, Xv_test, ids_test,idv_test = data_parser.parse(df=dfTest)
        dfm_params["feature_size"] = fd.feat_dim
        #print(fd.feat_dict)
        dfm_params["field_size"] = len(Xi_train[0])
        print(dfm_params)

        y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
        y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
        _get = lambda x, l: [x[i] for i in l]
        #gini_results_cv = np.zeros(len(folds), dtype=float)
        #gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
        #gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
        y_train = list(map(float,y_train))
        for i, (train_idx, valid_idx) in enumerate(folds):
            Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
            Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

            dfm = DeepFM(**dfm_params)
            
            dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_,early_stopping=early_stopping)

            y_train_meta[valid_idx,0] = dfm.predict(Xi_valid_, Xv_valid_)
            y_test_meta[:,0] += dfm.predict(Xi_test, Xv_test)
        
            #gini_results_cv[i] = mean_absolute_error(y_valid_, y_train_meta[valid_idx])
            #gini_results_epoch_train[i] = dfm.train_result
            #gini_results_epoch_valid[i] = dfm.valid_result

        y_test_meta /= float(len(folds))

        return y_train_meta, y_test_meta
    
    if task_type == "Regression":
        cold_mae = []
        cold_rmse = []
        warm_mae = []
        warm_rmse = []

        def model_choose(deep):
        
            dfm_params['use_deep']=deep
    
            for times in range(10):
                
                train, test = train_test_split(data, test_size=0.2, random_state=times)
                tr_x, tr_Xi, tr_y, tr_idx, te_x, te_Xi, te_y, val_x, val_Xi, val_y, val_idx, meta_info, model_info, sy, sy_t = data_initialize(train, test, meta_info_ori, task_type, 'warm', random_state=0, verbose=False) 
                
                train_x = np.concatenate([tr_x,val_x],0)
                train_y = np.concatenate([tr_y,val_y],0)
                
                train_y = sy.inverse_transform(train_y)
                te_y = sy_t.inverse_transform(te_y)
                
                train_Xi = np.concatenate([tr_Xi,val_Xi],0)
                train_ = np.concatenate([train_x,train_Xi,train_y],1)
    
                test_ = np.concatenate([te_x,te_Xi,te_y],1)
    
                dfTrain = pd.DataFrame(train_,columns=train.columns)
                dfTest = pd.DataFrame(test_,columns=test.columns)
                
                dfTrain = train
                dfTest = test
                
                #dfTrain.user_id = dfTrain.user_id.astype(int).astype(str)
                #dfTrain.item_id = dfTrain.item_id.astype(int).astype(str)
                #dfTrain.target = dfTrain.target.astype(str)

                
                #posi = dfTrain.shape[1]-3
                
                #dfTrain.iloc[:,np.r_[:posi,-1]].astype(float)
                #dfTest.iloc[:,np.r_[:posi,-1]].astype(float)
                

    
                folds = list(KFold(n_splits=3, shuffle=True,
                             random_state=random_state).split(dfTrain.iloc[:,:-1].values, dfTrain.target.values))
                
                y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)
                
                #y_test_dfm = sy.inverse_transform(y_test_dfm.reshape(-1,1))
                #te_y = sy_t.inverse_transform(te_y.reshape(-1,1))
                if wc == 'warm':
                    if len([(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')])!=1:
                        warm_y = te_y[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                        warm_pred = y_test_dfm[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                    else:
                        warm_y = te_y
                        warm_pred= y_test_dfm
                    warm_mae.append(mean_absolute_error(warm_y,warm_pred))
                    warm_rmse.append(mean_squared_error(warm_y,warm_pred)**0.5)
    
                if wc == 'cold':
                    try:
                        [(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')] != [True]
                        print('no cold samples')
                        continue
                    except:
                        cold_y = te_y[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                        cold_pred = y_test_dfm[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]

                    cold_mae.append(mean_absolute_error(cold_y,cold_pred))
                    cold_rmse.append(mean_squared_error(cold_y,cold_pred)**0.5)

                
            if deep==True:
                test_model = 'deepfm'
            else:
                test_model = 'fm'
                
            if wc == 'warm':
                i_result = np.array([test_model,np.mean(warm_mae),np.mean(warm_rmse),np.std(warm_mae),np.std(warm_rmse)]).reshape(1,-1)
                result = pd.DataFrame(i_result,columns=['model','warm_mae','warm_rmse','std_warm_mae','std_warm_rmse'])

            if wc == 'cold':
                i_result = np.array([test_model,np.mean(cold_mae),np.mean(cold_rmse),np.std(cold_mae),np.std(cold_rmse)]).reshape(1,-1)
                result = pd.DataFrame(i_result,columns=['model','cold_mae','cold_rmse','std_cold_mae','std_cold_rmse'])
            
            return result
    
        result_1 = (model_choose(True))
        result_2 = (model_choose(False))

        return result_1, result_2
    
    if task_type == "Classification":
        cold_auc = []
        cold_logloss = []
        warm_auc = []
        warm_logloss = []

        def model_choose(deep):
        
            dfm_params['use_deep']=deep
    
            for times in range(10):
                
                train, test = train_test_split(data, test_size=0.2, random_state=times)
                tr_x, tr_Xi, tr_y, tr_idx, te_x, te_Xi, te_y, val_x, val_Xi, val_y, val_idx, meta_info, model_info, sy, sy_t = data_initialize(train, test, meta_info_ori, task_type, 'warm', random_state=0, verbose=False) 
                
                train_x = np.concatenate([tr_x,val_x],0)
                train_y = np.concatenate([tr_y,val_y],0)
                train_Xi = np.concatenate([tr_Xi,val_Xi],0)
                train_ = np.concatenate([train_x,train_Xi,train_y],1)
    
                test_ = np.concatenate([te_x,te_Xi,te_y],1)
    
                dfTrain = pd.DataFrame(train_,columns=train.columns)
                dfTest = pd.DataFrame(test_,columns=test.columns)
                
                dfTrain = dfTrain.astype(str)
                dfTest = dfTest.astype(str)
                
                dfTrain = train
                dfTest = test
                
                #posi = dfTrain.shape[1]-3
                
                #dfTrain.iloc[:,np.r_[:posi,-1]].astype(float)
                #dfTest.iloc[:,np.r_[:posi,-1]].astype(float)


    
                folds = list(KFold(n_splits=3, shuffle=True,
                             random_state=random_state).split(train_x, train_y))
                

                y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)
            
                if wc == 'warm':            

                    if len([(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')])!=1:
                        warm_y = te_y[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                        warm_pred = y_test_dfm[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                    else:
                        warm_y = te_y
                        warm_pred= y_test_dfm
                    warm_auc.append(roc_auc_score(warm_y,warm_pred))
                    warm_logloss.append(log_loss(warm_y,warm_pred))
                    
                if wc == 'cold':
                    
                    try:
                        [(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')] != [True]
                        print('no cold samples')
                        continue
                    except:
                        cold_y = te_y[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                        cold_pred = y_test_dfm[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]  
                    cold_auc.append(roc_auc_score(cold_y,cold_pred))
                    cold_logloss.append(log_loss(cold_y,cold_pred))

            if deep==True:
                test_model = 'deepfm'
            else:
                test_model = 'fm'
                    
            if wc == 'warm':
                i_result = np.array([test_model,np.mean(warm_auc),np.mean(warm_logloss),np.std(warm_auc),np.std(warm_logloss)]).reshape(1,-1)
                result = pd.DataFrame(i_result,columns=['model','warm_auc','warm_logloss','std_warm_auc','std_warm_logloss'])

            if wc == 'cold':
                i_result = np.array([test_model,np.mean(cold_auc),np.mean(cold_logloss),np.std(cold_auc),np.std(cold_logloss)]).reshape(1,-1)
                result = pd.DataFrame(i_result,columns=['model','cold_auc','cold_logloss','std_cold_auc','std_cold_logloss'])

            return result
    
        result_1 = (model_choose(True))
        result_2 = (model_choose(False))

        return result_1, result_2



