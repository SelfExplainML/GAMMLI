import numpy as np
import pandas as pd 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,roc_auc_score,mean_absolute_error,log_loss
import sys
sys.path.append('../')
from gammli.gammli import GAMMLI
from gammli.dataReader import data_initialize

def gammli( wc, data, meta_info_ori, task_type="Regression", random_state=0, params=None):
    
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
    best_lambda = params['best_lambda']
    verbose = params['verbose']
    
    result = []
    
    if task_type == "Regression":
        cold_mae = []
        cold_rmse = []
        warm_mae = []
        warm_rmse = []
        #gami_mae = []
        #gami_rmse = []
        """
        if auto_tune:
        
            model = GAMMLI(model_info=model_info, meta_info=meta_info, subnet_arch=[8, 16],interact_arch=[20, 10],activation_func=tf.tanh, batch_size=min(500, int(0.2*tr_x.shape[0])), lr_bp=0.001, auto_tune=True,
                           interaction_epochs=interaction_epochs,main_effect_epochs=main_effect_epochs,tuning_epochs=tuning_epochs,loss_threshold_main=0.01,loss_threshold_inter=0.01,
                           verbose=False, early_stop_thres=100,interact_num=10,u_group_num=u_group_num,i_group_num=i_group_num,scale_ratio=1,n_power_iterations=5,n_oversamples=0,
                           mf_training_iters=mf_training_iters,change_mode=True,convergence_threshold=0.0001,max_rank=rank,wc=wc,interaction_restrict='intra',si_approach ='als',lambda_=lambda_)
            GAMMLI(wc='warm',model_info=model_info, meta_info=meta_info, subnet_arch=[20, 10],interact_arch=[20, 10],activation_func=tf.tanh, batch_size=min(500, int(0.2*tr_x.shape[0])), lr_bp=0.001, auto_tune=False,
               interaction_epochs=0,main_effect_epochs=50,tuning_epochs=20,loss_threshold_main=0.01,loss_threshold_inter=0.1,
              verbose=True, early_stop_thres=20,interact_num=10,n_power_iterations=5,n_oversamples=10, u_group_num=19, i_group_num=2, reg_clarity=1,
              mf_training_iters=200,change_mode=False,convergence_threshold=0.0001,max_rank=3,interaction_restrict='intra', si_approach ='als')
            
            model.fit(tr_x, val_x, tr_y, val_y, tr_Xi, val_Xi, tr_idx, val_idx)
        
            self.best_u_group_num = int(u_group_num)
            self.best_i_group_num = int(i_group_num)
            self.best_lambda = lambda_
        """
        

            
        train, test = train_test_split(data, test_size=0.2, random_state=random_state)
        tr_x, tr_Xi, tr_y, tr_idx, te_x, te_Xi, te_y, val_x, val_Xi, val_y, val_idx, meta_info, model_info ,sy, sy_t= data_initialize(train, test, meta_info_ori, task_type, 'warm', random_state=0, verbose=False)            

        model = GAMMLI(model_info=model_info, meta_info=meta_info, subnet_arch=[8, 16],interact_arch=[20, 10],activation_func=tf.tanh, batch_size=min(500, int(0.2*tr_x.shape[0])), lr_bp=0.001, auto_tune=auto_tune,
                       interaction_epochs=interaction_epochs,main_effect_epochs=main_effect_epochs,tuning_epochs=tuning_epochs,loss_threshold_main=0.01,loss_threshold_inter=0.01,
                       verbose=verbose, early_stop_thres=100,interact_num=10,u_group_num=u_group_num,i_group_num=i_group_num,n_power_iterations=5,lambda_=best_lambda,
                       mf_training_iters=mf_training_iters,change_mode=True,convergence_threshold=0.0001,max_rank=rank,wc='warm',interaction_restrict='intra', si_approach ='als')

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
            result.append(mean_absolute_error(warm_y,warm_pred))
            result.append(mean_squared_error(warm_y,warm_pred)**0.5)


        if wc == 'cold':
            try:
                [(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')] != [True]
                print('no cold samples')
            except:
                cold_y = te_y[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                cold_pred = pred[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]

                result.append(mean_absolute_error(cold_y,cold_pred))
                result.append(mean_squared_error(cold_y,cold_pred)**0.5)
                
                
    
        return result
    
    if task_type == "Classification":
        cold_auc = []
        cold_logloss = []
        warm_auc = []
        warm_logloss = []
        #gami_auc = []
        #gami_logloss = []
        
        """
        if auto_tune:
        
            model = GAMMLI(wc=wc,model_info=model_info, meta_info=meta_info, subnet_arch=[8, 16],interact_arch=[20, 10],activation_func=tf.tanh, batch_size=min(500, int(0.2*tr_x.shape[0])), lr_bp=0.01, auto_tune=True,
                           interaction_epochs=interaction_epochs,main_effect_epochs=main_effect_epochs,tuning_epochs=tuning_epochs,loss_threshold_main=0.01,loss_threshold_inter=0.01,
                           verbose=False, early_stop_thres=100,interact_num=10,u_group_num=u_group_num,i_group_num=i_group_num,n_power_iterations=5,lambda_=best_lambda,
                           mf_training_iters=mf_training_iters,change_mode=True,convergence_threshold=0.0001,max_rank=rank,interaction_restrict='intra', si_approach ='als')
    
            model.fit(tr_x, val_x, tr_y, val_y, tr_Xi, val_Xi, tr_idx, val_idx)
        
            best_ratio = model.final_mf_model.best_ratio
            best_combine_range = model.final_mf_model.best_combine_range
        """

        

        train, test = train_test_split(data, test_size=0.2, random_state=random_state)
        tr_x, tr_Xi, tr_y, tr_idx, te_x, te_Xi, te_y, val_x, val_Xi, val_y, val_idx, meta_info, model_info ,sy, sy_t= data_initialize(train, test, meta_info_ori, task_type, 'warm', random_state=0, verbose=False)


        model = GAMMLI(wc='warm',model_info=model_info, meta_info=meta_info, subnet_arch=[8, 16],interact_arch=[20, 10],activation_func=tf.tanh, batch_size=min(500, int(0.2*tr_x.shape[0])), lr_bp=0.001, auto_tune=auto_tune,
                       interaction_epochs=interaction_epochs,main_effect_epochs=main_effect_epochs,tuning_epochs=tuning_epochs,loss_threshold_main=0.01,loss_threshold_inter=0.01,
                       verbose=verbose, early_stop_thres=100,interact_num=10,u_group_num=u_group_num,i_group_num=i_group_num,n_power_iterations=5,lambda_=best_lambda,
                       mf_training_iters=mf_training_iters,change_mode=True,convergence_threshold=0.001,max_rank=rank,interaction_restrict='intra', si_approach ='als')

        model.fit(tr_x, val_x, tr_y, val_y, tr_Xi, val_Xi, tr_idx, val_idx)

        pred = model.predict(te_x, te_Xi)
            
        if wc == 'warm':
            if len([(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')])!=1:
                warm_y = te_y[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                warm_pred = pred[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
            else:
                warm_y = te_y
                warm_pred= pred            
            result.append(roc_auc_score(warm_y,warm_pred))
            result.append(log_loss(warm_y,warm_pred))

        if wc == 'cold':

            try:
                [(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')] != [True]
                print('no cold samples')
            except:
                cold_y = te_y[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                cold_pred = pred[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]

                result.append(roc_auc_score(cold_y,cold_pred))
                result.append(log_loss(cold_y,cold_pred))
                       

        return result