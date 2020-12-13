# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:21:22 2020

@author: suyu
"""

import numpy as np
import pandas as pd 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,roc_auc_score,mean_absolute_error,log_loss
import sys
sys.path.append('../')
from gammli.GAMMLI import GAMMLI
from gammli.DataReader import data_initialize

def rtest(wc, data, meta_info_ori, task_type="Regression", random_state=0, params=None):
    
    train, test = train_test_split(data, test_size=0.2, random_state=0)
    tr_x, tr_Xi, tr_y, tr_idx, te_x, te_Xi, te_y, val_x, val_Xi, val_y, val_idx, meta_info, model_info , sy, sy_t= data_initialize(train, test, meta_info_ori, task_type, 'warm', random_state=0, verbose=True)


    rank = params['rank']
    main_effect_epochs = params['main_effect_epochs']
    interaction_epochs = params['interaction_epochs']
    tuning_epochs = params['tuning_epochs']
    mf_training_iters = params['mf_training_iters']
    u_group_num = params['u_group_num']
    i_group_num = params['i_group_num']
    auto_tune = params['auto_tune']
    best_ratio = params['best_shrinkage']
    best_combine_range = params['best_combination']
    verbose = params['verbose']
    
    if task_type == "Regression":
        cold_mae = []
        cold_rmse = []
        warm_mae = []
        warm_rmse = []
        #gami_mae = []
        #gami_rmse = []
        if auto_tune:
        
            model = GAMMLI(model_info=model_info, meta_info=meta_info, subnet_arch=[8, 16],interact_arch=[20, 10],activation_func=tf.tanh, batch_size=1000, lr_bp=0.01, auto_tune=True,
                           interaction_epochs=interaction_epochs,main_effect_epochs=main_effect_epochs,tuning_epochs=tuning_epochs,loss_threshold_main=0.01,loss_threshold_inter=0.01,
                           verbose=False, early_stop_thres=100,interact_num=10,u_group_num=u_group_num,i_group_num=i_group_num,scale_ratio=1,n_power_iterations=5,n_oversamples=0,
                           mf_training_iters=mf_training_iters,change_mode=True,convergence_threshold=0.001,max_rank=rank,random_state=0, interaction_restrict='intra')
            
            model.fit(tr_x, val_x, tr_y, val_y, tr_Xi, val_Xi, tr_idx, val_idx)
        
            best_ratio = model.final_mf_model.best_ratio
            best_combine_range = model.final_mf_model.best_combine_range

        rank_li =[]
        for rank in range(1,11):
            
            print(rank)
            
            train, test = train_test_split(data, test_size=0.2, random_state=0)
            tr_x, tr_Xi, tr_y, tr_idx, te_x, te_Xi, te_y, val_x, val_Xi, val_y, val_idx, meta_info, model_info ,sy, sy_t= data_initialize(train, test, meta_info_ori, task_type, 'warm', random_state=0, verbose=False)            

            rank_li.append(rank)
            model = GAMMLI(model_info=model_info, meta_info=meta_info, subnet_arch=[8, 16],interact_arch=[20, 10],activation_func=tf.tanh, batch_size=1000, lr_bp=0.01, auto_tune=False,
                           interaction_epochs=interaction_epochs,main_effect_epochs=main_effect_epochs,tuning_epochs=tuning_epochs,loss_threshold_main=0.01,loss_threshold_inter=0.01,combine_range=best_combine_range,
                           verbose=verbose, early_stop_thres=100,interact_num=10,u_group_num=u_group_num,i_group_num=i_group_num,scale_ratio=best_ratio,n_power_iterations=5,n_oversamples=0,
                           mf_training_iters=mf_training_iters,change_mode=True,convergence_threshold=0.001,max_rank=rank,random_state=0,wc=wc, interaction_restrict='intra')
    
            model.fit(tr_x, val_x, tr_y, val_y, tr_Xi, val_Xi, tr_idx, val_idx)
            
            pred = model.predict(te_x, te_Xi)
            pred = sy.inverse_transform(pred.reshape(-1,1))
            te_y = sy_t.inverse_transform(te_y.reshape(-1,1))
        
            if wc == 'warm':
                if len([(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')])!=1:
                    warm_y = te_y[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                    warm_pred = pred[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                else:
                    warm_y = te_y
                    warm_pred= pred
                    
                
                warm_mae.append(mean_absolute_error(warm_y,warm_pred))
                warm_rmse.append(mean_squared_error(warm_y,warm_pred)**0.5)


                
            if wc == 'cold':
                if [(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')] != [True]:
                    cold_y = te_y[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                    cold_pred = pred[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                else:
                    print('no cold samples')
                    return
                cold_mae.append(mean_absolute_error(cold_y,cold_pred))
                cold_rmse.append(mean_squared_error(cold_y,cold_pred)**0.5)
                
                
        r = np.array(rank_li).reshape(-1,1)
        mae = np.array(warm_mae).reshape(-1,1)
        rmse = np.array(warm_rmse).reshape(-1,1)
        i_result = np.concatenate([r,mae,rmse],1)
        result = pd.DataFrame(i_result,columns=['rank','mae','rmse'])
           
        return result
    
    if task_type == "Classification":
        cold_auc = []
        cold_logloss = []
        warm_auc = []
        warm_logloss = []
        rank_li =[]
        #gami_auc = []
        #gami_logloss = []
        
        if auto_tune:
        
            model = GAMMLI(wc=wc,model_info=model_info, meta_info=meta_info, subnet_arch=[8, 16],interact_arch=[20, 10],activation_func=tf.tanh, batch_size=1000, lr_bp=0.01, auto_tune=True,
                           interaction_epochs=interaction_epochs,main_effect_epochs=main_effect_epochs,tuning_epochs=tuning_epochs,loss_threshold_main=0.01,loss_threshold_inter=0.01,
                           verbose=False, early_stop_thres=100,interact_num=10,u_group_num=u_group_num,i_group_num=i_group_num,scale_ratio=1,n_power_iterations=5,n_oversamples=0,
                           mf_training_iters=mf_training_iters,change_mode=True,convergence_threshold=0.001,max_rank=rank,random_state=0, interaction_restrict='intra')
    
            model.fit(tr_x, val_x, tr_y, val_y, tr_Xi, val_Xi, tr_idx, val_idx)
        
            best_ratio = model.final_mf_model.best_ratio
            best_combine_range = model.final_mf_model.best_combine_range

        else:
            best_ratio = 0.84
            best_combine_range = 0.8
        
        for rank in range(1,11):
            
            print(rank)
            rank_li.append(rank)
            model = GAMMLI(wc=wc,model_info=model_info, meta_info=meta_info, subnet_arch=[8, 16],interact_arch=[20, 10],activation_func=tf.tanh, batch_size=1000, lr_bp=0.01, auto_tune=False,
                           interaction_epochs=interaction_epochs,main_effect_epochs=main_effect_epochs,tuning_epochs=tuning_epochs,loss_threshold_main=0.01,loss_threshold_inter=0.01,combine_range=best_combine_range,
                           verbose=False, early_stop_thres=100,interact_num=10,u_group_num=u_group_num,i_group_num=i_group_num,scale_ratio=best_ratio,n_power_iterations=5,n_oversamples=0,
                           mf_training_iters=mf_training_iters,change_mode=True,convergence_threshold=0.001,max_rank=rank,random_state=0, interaction_restrict='intra')
            
            model.fit(tr_x, val_x, tr_y, val_y, tr_Xi, val_Xi, tr_idx, val_idx)
            
            pred = model.predict(te_x, te_Xi)
            
            if wc == 'warm':
                if len([(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')])!=1:
                    warm_y = te_y[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                    warm_pred = pred[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                else:
                    warm_y = te_y
                    warm_pred= pred            
                warm_auc.append(roc_auc_score(warm_y,warm_pred))
                warm_logloss.append(log_loss(warm_y,warm_pred))
                
            if wc == 'cold':
        
                cold_y = te_y[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                cold_pred = pred[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                cold_auc.append(roc_auc_score(cold_y,cold_pred))
                cold_logloss.append(log_loss(cold_y,cold_pred))
                

        r = np.array(rank_li).reshape(-1,1)
        mae = np.array(warm_auc).reshape(-1,1)
        rmse = np.array(warm_logloss).reshape(-1,1)
        i_result = np.concatenate([r,mae,rmse],1)
        result = pd.DataFrame(i_result,columns=['rank','auc','logloss'])
        
        return result


