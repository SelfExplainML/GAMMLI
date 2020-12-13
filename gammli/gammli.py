import os
import time
import numpy as np
import tensorflow as tf
from .li import LatentVariable
from .gaminet import GAMINet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans 
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
from .utils import global_visualize_density
from itertools import product
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

class GAMMLI:
    """
    Generalized Addtive Model with Manifest and Latent Interactions
    
    :param dict model_info: model basic information.
    :param array subnet_arch: subnetwork architecture.
    :param array interact_arch: interact subnetwork architecture.
    :param func activation_func: activation_function.
    :param float lr_bp: learning rate.
    :param float loss_threshold_main: main_effect loss threshold.
    :param float loss_threshold_inter: interact_effect loss threshold.
    :param int main_grid_size: number of the sampling points for main_effect training..
    :param int interact_grid_size: number of the sampling points for interact_effect training.
    :param int batch_size: size of batch.
    :param int main_effect_epochs: main effect training stage epochs.
    :param int tuning_epochs: tuning stage epochs.
    :param int interaction_epochs: interact effect training stage epochs.
    :param int interact_num: the max interact pair number.
    :param str interaction_restrict: interaction restrict settings.
    :param int early_stop_thres: epoch for starting the early stop.
    :param float convergence_threshold: convergence threshold for latent effect training.
    :param int mf_training_iters: latent effect training stage epochs.
    :param int max_rank: max rank for the latent variable.
    :param bool change_mode: whether change the initial value for latent effect training.
    :param int u_group_num: number of user group.
    :param int i_group_num: number of item group.
    :param float scale_ratio: group range shrinkage ratio.
    :param float combine_range: group combination range.
    :param bool auto_tune: whether auto tune the hyperparameter.
    :param int random_state: number of user group.
    :param str wc: build model for 'warm start' or 'cold start'


    """
    def __init__(self,
                 meta_info=None,
                 model_info=None,
                 subnet_arch=[10, 6],
                 interact_arch=[20, 10],
                 activation_func=tf.tanh,
                 lr_bp=0.001,
                 loss_threshold_main=0.01,
                 loss_threshold_inter=0.01,
                 main_grid_size=41,
                 interact_grid_size=41,
                 batch_size=1000,
                 main_effect_epochs=10000, 
                 tuning_epochs=500,
                 interaction_epochs=20,
                 interact_num=20,
                 interaction_restrict=None,
                 verbose=False,
                 early_stop_thres=100,
                 shrinkage_value=2,
                 convergence_threshold=0.001,
                 mf_training_iters=20,
                 max_rank=None,
                 n_power_iterations=1,
                 n_oversamples=10,
                 init_fill_method="zero",
                 min_value=None,
                 max_value=None,
                 change_mode = False,
                 normalizer=None,
                 multi_type_num=0,
                 u_group_num=0,
                 i_group_num=0,
                 val_u_group = 0,
                 val_i_group = 0,
                 reg_clarity=0.001,
                 scale_ratio=1,
                 auto_tune=False,
                 random_state = 0,
                 combine_range=0.99,
                 wc = None,
                 lambda_ = 0,
                 si_approach = 'rsvd'):

        super(GAMMLI, self).__init__()

        self.meta_info = meta_info
        self.model_info = model_info
        
        self.subnet_arch = subnet_arch
        self.interact_arch = interact_arch
        self.activation_func = activation_func

        self.lr_bp = lr_bp
        self.loss_threshold_main = loss_threshold_main
        self.loss_threshold_inter = loss_threshold_inter
        self.main_grid_size = main_grid_size
        self.interact_grid_size = interact_grid_size
        self.batch_size = batch_size
        self.tuning_epochs = tuning_epochs
        self.main_effect_epochs = main_effect_epochs
        self.interaction_epochs = interaction_epochs
        self.interact_num = interact_num
        self.interaction_restrict = interaction_restrict

        self.verbose = verbose
        self.early_stop_thres = early_stop_thres

        self.fill_method = init_fill_method
        self.min_value = min_value
        self.max_value = max_value
        self.normalizer = normalizer
        self.shrinkage_value = shrinkage_value
        self.convergence_threshold = convergence_threshold
        self.mf_max_iters = mf_training_iters
        self.max_rank = max_rank
        self.change_mode =change_mode
        self.n_power_iterations = n_power_iterations
        self.n_oversamples = n_oversamples
        self.reg_clarity = reg_clarity

        self.multi_type_num = multi_type_num
        self.u_group_num = u_group_num
        self.i_group_num = i_group_num
        self.val_u_group = val_u_group
        self.val_i_group = val_i_group        
        self.scale_ratio = scale_ratio
        self.auto_tune = auto_tune
        self.random_state = random_state
        self.combine_range = combine_range
        self.wc = wc
        self.lambda_ = lambda_
        self.si_approach = si_approach


        tf.random.set_seed(self.random_state)
        simu_dir = "./results/gaminet/"
        #path = 'data/simulation/sim_0.9.csv'
        if not os.path.exists(simu_dir):
            os.makedirs(simu_dir)

        self.task_type = self.model_info['task_type']
        self.feat_dict = self.model_info['feat_dict']
        self.ui_shape = self.model_info['ui_shape']

        if self.task_type == "Regression":
            #self.loss_fn = tf.keras.losses.MeanSquaredError()
            self.loss_fn = tf.keras.losses.MeanAbsoluteError()
        elif self.task_type == "Classification":
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()

    #gam first mf second    
    def fit(self,tr_x, val_x, tr_y, val_y, tr_Xi, val_Xi, tr_idx, val_idx):
        
        """
        Build a GAMMLI model from the dataset (tr_x, val_x, tr_y, val_y, tr_Xi, val_Xi, tr_idx, val_idx).

        :param array tr_x: explict effect feature in training set.
        :param array val_x: explict effect feature in validation set.
        :param array tr_y: target variable in training set.
        :param array val_y: target variable in validation set.
        :param array tr_Xi: implicit effect feature in training set.
        :param array val_Xi: implicit effect feature in validation set.
        :param array tr_idx: training set index.
        :param array tr_idx: validation set index.
        
        :return: fitted GAMMLI model
        """

        """
        #initial cluster training
        self.user_feature_list = []
        self.item_feature_list = []
        for indice, (feature_name, feature_info) in enumerate(self.meta_info.items()):
            if feature_info["source"] == "user":
                self.user_feature_list.append(indice)
            elif feature_info["source"] == "item":
                self.item_feature_list.append(indice)


        user_feature = np.concatenate([tr_x[:,self.user_feature_list],tr_Xi[:,0].reshape(-1,1)],1)
        item_feature = np.concatenate([tr_x[:,self.item_feature_list],tr_Xi[:,1].reshape(-1,1)],1)
        user_feature = np.unique(user_feature,axis=0)
        item_feature = np.unique(item_feature,axis=0)
        user_feature = user_feature[np.argsort(user_feature[:,-1])]
        item_feature = item_feature[np.argsort(item_feature[:,-1])]
        self.user_id = user_feature[:,-1]
        self.item_id = item_feature[:,-1]
        user_feature = user_feature[:,:-1]
        item_feature = item_feature[:,:-1]
        
        val_user_feature = np.concatenate([val_x[:,self.user_feature_list],val_Xi[:,0].reshape(-1,1)],1)
        val_item_feature = np.concatenate([val_x[:,self.item_feature_list],val_Xi[:,1].reshape(-1,1)],1)
        val_user_feature = np.unique(val_user_feature,axis=0)
        val_item_feature = np.unique(val_item_feature,axis=0)
        val_user_feature = val_user_feature[np.argsort(val_user_feature[:,-1])]
        val_item_feature = val_item_feature[np.argsort(val_item_feature[:,-1])]
        val_user_feature = val_user_feature[:,:-1]
        val_item_feature = val_item_feature[:,:-1]

        if self.u_group_num != 0:
            self.u_group, self.u_group_model = self.main_effect_cluster(user_feature,self.u_group_num)
            self.val_u_group = self.u_group_model.predict(val_user_feature)
        else:
            self.u_group=0
        if self.i_group_num != 0:
            self.i_group, self.i_group_model = self.main_effect_cluster(item_feature,self.i_group_num)
            self.val_i_group = self.i_group_model.predict(val_item_feature)
        else:
            self.i_group = 0
            
        """

        error1=[]
        val_error1=[]
        error2=[]
        val_error2 =[]
        val_error = []


        #gam fit
        self.gami_model = GAMINet(meta_info=self.meta_info,interact_num=self.interact_num,interact_arch=self.interact_arch,
                                  subnet_arch=self.subnet_arch, task_type=self.task_type,
                                  activation_func=tf.tanh, batch_size=self.batch_size, lr_bp=self.lr_bp,
                                  main_effect_epochs=self.main_effect_epochs,tuning_epochs=self.tuning_epochs,
                                  interaction_epochs=self.interaction_epochs, reg_clarity=self.reg_clarity,interaction_restrict=self.interaction_restrict,
                                  verbose=self.verbose, early_stop_thres=self.early_stop_thres,random_state=self.random_state)


        model = self.gami_model
        st_time = time.time()

        model.fit(tr_x, val_x, tr_y, val_y, tr_idx, val_idx)

        fi_time = time.time()
        print('time cost:',fi_time-st_time)

        pred_train = model.predict(tr_x)
        pred_val = model.predict(val_x)   
        error1.append(self.loss_fn(tr_y.ravel(),pred_train.ravel()).numpy())
        val_error1.append(self.loss_fn(val_y.ravel(),pred_val.ravel()).numpy())
        if self.task_type == 'Classification':
            pred_train_initial = model.predict_initial(tr_x).numpy()
            pred_train_initial[pred_train_initial>np.log(9999)] = np.log(9999)
            pred_train_initial[pred_train_initial<np.log(1/9999)] = np.log(1/9999)
            pred_val_initial = model.predict_initial(val_x).numpy()
            pred_val_initial[pred_val_initial>np.log(9999)] = np.log(9999)
            pred_val_initial[pred_val_initial<np.log(1/9999)] = np.log(1/9999)

        print('After the gam stage, training error is %0.5f , validation error is %0.5f' %(error1[-1],val_error1[-1]))
        if self.task_type == 'Regression':
            residual = (tr_y.ravel() - pred_train.ravel()).reshape(-1,1)
            residual_val = (val_y.ravel() - pred_val.ravel()).reshape(-1,1)
        elif self.task_type == 'Classification':
            tr_y_temp = deepcopy(tr_y)
            val_y_temp = deepcopy(val_y)
            tr_y_temp[tr_y_temp==0]=-1
            val_y_temp[val_y_temp==0]=-1

            residual = (2*tr_y_temp.ravel()/(1 + np.exp(2*tr_y_temp.ravel()*pred_train_initial.ravel()))).reshape(-1,1)
            residual_val = (2*val_y_temp.ravel()/(1 + np.exp(2*val_y_temp.ravel()*pred_val_initial.ravel()))).reshape(-1,1)


        #mf fit
        if self.mf_max_iters !=0:
            
            if self.task_type == 'Classification':
                pred_train = pred_train_initial
                pred_val = pred_val_initial
            self.lv_model = LatentVariable(meta_info=self.meta_info,verbose = self.verbose,task_type=self.task_type,max_rank=self.max_rank,max_iters=self.mf_max_iters,
                                           change_mode=self.change_mode,auto_tune=self.auto_tune,
                                           convergence_threshold=self.convergence_threshold,n_oversamples=self.n_oversamples
                                           ,u_group_num=self.u_group_num, i_group_num=self.i_group_num
                                           ,scale_ratio=self.scale_ratio,pred_tr=pred_train,shrinkage_value=self.shrinkage_value,
                                           tr_y=tr_y,pred_val=pred_val,val_y=val_y, tr_Xi=tr_Xi,val_Xi=val_Xi,random_state=self.random_state
                                           ,combine_range=self.combine_range, wc = self.wc, lambda_= self.lambda_, si_approach = self.si_approach)
            model1 = self.lv_model
            st_time = time.time()
            model1.fit(tr_x,val_x,tr_Xi,val_Xi,residual,residual_val,self.ui_shape)
            fi_time = time.time()
            print('time cost:',fi_time-st_time)

            pred = model1.predict(tr_Xi)
            predval = model1.predict(val_Xi)
            if self.task_type == 'Classification':
                error2.append(self.loss_fn(tr_y.ravel(),tf.sigmoid(pred.ravel()+pred_train.ravel()).numpy()).numpy())
                val_error2.append(self.loss_fn(val_y.ravel(),tf.sigmoid(predval.ravel()+pred_val.ravel()).numpy()).numpy())
            else:
                error2.append(self.loss_fn(tr_y.ravel(),pred.ravel()+pred_train.ravel()).numpy())
                val_error2.append(self.loss_fn(val_y.ravel(),predval.ravel()+pred_val.ravel()).numpy())
            self.mf_tr_err = error2[-1]
            self.mf_val_err = val_error2[-1]
            print('After the matrix factor stage, training error is %0.5f, validation error is %0.5f' %(error2[-1],val_error2[-1]))


            val_error_bi = [val_error1[-1],val_error2[-1]]
            val_error = val_error + val_error_bi

   
        self.final_gam_model = self.gami_model
        self.final_mf_model = self.lv_model
        if self.si_approach == 'rsvd':
            self.s = np.diag(self.final_mf_model.s)
            self.u = self.final_mf_model.u
            self.v = self.final_mf_model.v.T
        elif self.si_approach == 'als':
            self.s = self.final_mf_model.s
            self.u = self.final_mf_model.u
            self.v = self.final_mf_model.v
        
        self.cur_rank = self.final_mf_model.cur_rank
        self.match_i = self.final_mf_model.match_i
        self.match_u = self.final_mf_model.match_u
        self.u_group_model = self.final_mf_model.u_group_model
        self.i_group_model = self.final_mf_model.i_group_model
        self.user_feature_list = self.final_mf_model.user_feature_list
        self.item_feature_list = self.final_mf_model.item_feature_list
        self.feature_list_ = self.gami_model.feature_list_
 
        


    def predict(self,xx,Xi):
        
        """
        predict result by fitted GAMMLI model.

        :param array xx: the explicit features of samples for predict.
        :param array Xi: the implicit features of samples for predict.
        
        :return: prediction result
        """

        if self.mf_max_iters == 0 or self.final_mf_model==None:

            pred = self.final_gam_model.predict(xx)

            return pred

        else:
            
            pred1 = self.final_gam_model.predict(xx)
            if self.task_type == 'Classification':
                pred1 = self.final_gam_model.predict_initial(xx).numpy()
            
            pred2 = []
            for i in range(Xi.shape[0]):
                if Xi[i,0] == 'cold':
                    g = self.u_group_model.predict(xx[i,self.user_feature_list].reshape(1,-1))[0]
                    u = self.match_u[g]
                    #u=np.zeros(u.shape)
                else:
                    u = self.u[int(Xi[i,0])]
                if Xi[i,1] == 'cold':
                    g = self.i_group_model.predict(xx[i,self.item_feature_list].reshape(1,-1))[0]
                    v = self.match_i[g]
                    #v =np.zeros(v.shape)
                else:
                    v =self.v[int(Xi[i,1])]

                pred_mf = np.dot(u, np.multiply(self.s, v))
                pred2.append(pred_mf)
            pred2 = np.array(pred2)

            pred = pred1.ravel()+ pred2.ravel()
            

            
            if self.task_type == 'Classification':
                #print(pred.shape)
                #pred = tf.nn.softmax(pred).numpy()
                #print(pred.shape)
                pred = tf.sigmoid(pred).numpy()


            return pred

    
    def linear_global_explain(self):
        self.final_gam_model.global_explain(folder="./results", name="demo", cols_per_row=3, main_density=3, save_png=False, save_eps=False)

    def local_explain(self,class_,ex_idx,xx,Xi,y,simu_dir = 'result'):


        mf_output = self.final_mf_model.predict(Xi[ex_idx].reshape(1,-1))
        data_dict_local = self.final_gam_model.local_explain(class_ ,mf_output,xx[[ex_idx],:], y[[ex_idx],:],save_dict=False)
        return data_dict_local

            
            




    def cold_start_analysis(self,xx,u_i,confi):
        
        if u_i == 'user':
            g = self.u_group_model.predict(xx[:,self.user_feature_list].reshape(1,-1))[0]
            group_pre_u = self.final_mf_model.pre_u
            g = self.new_group(g,group_pre_u)
            mean_g = self.match_u[g]
            std_g = self.var_u[g]**0.5

        if u_i == 'item':
            g = self.i_group_model.predict(xx[:,self.item_feature_list].reshape(1,-1))[0]
            group_pre_i = self.final_mf_model.pre_i
            g = self.new_group(g,group_pre_i)
            mean_g = self.match_i[g]
            std_g = self.var_i[g]**0.5

        upper = mean_g + confi * std_g
        lower = mean_g - confi * std_g
        
        print('The new '+u_i+' belong to group '+str(g)+'\n mean is '+str(mean_g)+'\n and std is '+ str(std_g)+
        '\n the confidence interval is ['+str(lower)+','+str(upper)+']')

        return mean_g, std_g, upper, lower
    
    def dash_board(self,data_dict, importance, simu_dir, save_eps=True):
        
        """
        Show a dashboard for global explain, which contains the explanation of main effects, manifest interactions and latent interactions.

        :param dict data_dict: explanation data for plotting.
        :param array importance: importance of each effects.
        :param str simu_dir: storage path.
        :param bool save_eps: whether save the eps of dashboard.
        
        :return: Showing dashboard.
        """
        
        im_list = importance.tolist()
        for i,j in data_dict.items():
            importance_ = im_list.pop(0)
            if data_dict[i]['importance'] !=0:
                data_dict[i]['importance'] = importance_

        importance = np.sum(im_list) *100
        global_visualize_density(data_dict, save_png=False,save_eps=save_eps, folder=simu_dir, name='s1_global')
        self.latent_graph(importance,save_eps=save_eps)

        
        
        
        
    def get_all_rank(self,Xi):
        
        """
        Get importance of each effects.

        :param array Xi: implicit effect feature in training set.
        
        :return: array of importance.
        """
        
        sorted_index, componment_scales_gam = self.final_gam_model.get_all_active_rank()
        componment_scales_gam = componment_scales_gam.reshape(-1,1)
        delta = np.array(self.final_mf_model.rank_norm(Xi)).reshape(-1,1)
        componment_coefs= np.vstack([componment_scales_gam, delta])
        
        componment_scales = (np.abs(componment_coefs) / np.sum(np.abs(componment_coefs))).reshape([-1])
        
        return componment_scales
    
            
    def latent_graph(self,importance,save_eps=False):

        s = self.s
        user=np.array(list(self.match_u.values()))
        item=np.array(list(self.match_i.values()))
        
        user_m = np.multiply(user,s)
        item_m = np.multiply(item,s)

        Z1 = linkage(user_m, 'ward')
        Z2 = linkage(item_m, 'ward')

        left_x,left_y=0.1,0.1
        width,height=1,0.5
        left_yh=left_y+0.1+0.1
        left_yhh=left_yh+0.1+0.1
        left_xh=left_x+0.1
        left_xhh=left_xh+0.2

        heatmap_area=[left_xhh,left_y,width,height+0.15]
        #user_dendro=[left_x,left_y,0.1,height]
        #item_dendro = [left_xhh,0.25+height,1,0.1]
        user_heat = [left_xh,left_y,0.1,height+0.15]
        item_heat = [left_xhh, 0.3+height,1,0.1]
        cbar = [left_xhh+1.05,left_y,0.1,height+0.15]
        plt.figure(figsize=(8,8))
        plt.suptitle('Latent Interactions (%.2f%%)' %importance,fontsize=18, x=0.9)

        area_1=plt.axes(heatmap_area)
        #area_2=plt.axes(user_dendro)
        #area_3=plt.axes(item_dendro)
        area_4 =plt.axes(user_heat)
        area_5 =plt.axes(item_heat)
        area_6 = plt.axes(cbar)

        #h1 =dendrogram(Z1,ax=area_2,orientation='left',no_labels=True)
        #h2 = dendrogram(Z2,ax=area_3,no_labels=True)

        #user_s = user_m[h1['leaves']][::-1]
        #item_s = item_m[h2['leaves']]

        f_user = [np.linalg.norm(i) for i in user_m.tolist()]
        f_item = [np.linalg.norm(j) for j in item_m.tolist()]

        inter = []
        for i,j in product(user_m, item_m):
            inter.append(np.dot(i,np.divide(j,s)))

        inter = np.array(inter).reshape(user_m.shape[0],-1)

        ax1 = area_1.imshow(inter,aspect='auto')
        area_1.yaxis.set_ticks_position('right')
        
        area_1.set_yticks(range(user_m.shape[0]))
        area_1.set_xticks(range(item_m.shape[0]))
        area_1.set_yticklabels([str(i) for i in range(user_m.shape[0])])
        area_1.set_xticklabels([str(i) for i in range(item_m.shape[0])])


        ax4 = area_4.imshow(np.array(f_user).reshape(-1,1),aspect='auto')
        #area_2.set_xticks([])
        area_4.set_xticks([])
        area_4.set_yticks([])
        ax5 = area_5.imshow(np.array(f_item).reshape(1,-1),aspect='auto')
        #area_3.set_yticks([])
        area_5.set_xticks([])
        area_5.set_yticks([])

        #area_2.spines['top'].set_visible(False)
        #area_2.spines['right'].set_visible(False)
        #area_2.spines['bottom'].set_visible(False)
        #area_2.spines['left'].set_visible(False)
        #area_3.spines['top'].set_visible(False)
        #area_3.spines['right'].set_visible(False)
        #area_3.spines['bottom'].set_visible(False)
        #area_3.spines['left'].set_visible(False)
        area_6.spines['top'].set_visible(False)
        area_6.spines['right'].set_visible(False)
        area_6.spines['bottom'].set_visible(False)
        area_6.spines['left'].set_visible(False)
        area_6.set_xticks([])
        area_6.set_yticks([])
        plt.colorbar(ax1,ax=area_6)
        
        if save_eps:
            plt.savefig("latent.eps", bbox_inches="tight", dpi=100)
            
            
    def radar_plot(self, ui_type='user', group_index=[0,1],save_eps=False):
        if ui_type == 'user':
            labels = np.array(self.feature_list_)[self.user_feature_list].tolist()
            k = len(self.user_feature_list)
            plot_data = self.u_group_model.cluster_centers_[group_index]


            angles = np.linspace(0, 2*np.pi, k, endpoint=False)
            plot_data = np.concatenate((plot_data, plot_data[:,[0]]), axis=1) 
            angles = np.concatenate((angles, [angles[0]])) 

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True) 
            for i in range(len(plot_data)):
                ax.plot(angles, plot_data[i], 'o-', label = 'Group'+str(group_index[i]), linewidth=2)

            ax.set_rgrids(np.arange(0, 1.6, 0.2), np.arange(0, 1))
            ax.set_thetagrids(angles * 180/np.pi, labels)
            plt.legend(loc = 4)
            
        elif ui_type == 'item':
            labels = np.array(self.feature_list_)[self.item_feature_list].tolist()
            k = len(self.item_feature_list)
            plot_data = self.i_group_model.cluster_centers_[group_index]


            angles = np.linspace(0, 2*np.pi, k, endpoint=False)
            plot_data = np.concatenate((plot_data, plot_data[:,[0]]), axis=1) 
            angles = np.concatenate((angles, [angles[0]])) 

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True) 
            for i in range(len(plot_data)):
                ax.plot(angles, plot_data[i], 'o-', label = 'Group'+str(group_index[i]), linewidth=2)

            ax.set_rgrids(np.arange(0, 1.6, 0.2), np.arange(0, 1))
            ax.set_thetagrids(angles * 180/np.pi, labels)
            plt.legend(loc = 4)
            
        if save_eps:
            plt.savefig("latent.eps", bbox_inches="tight", dpi=100)








