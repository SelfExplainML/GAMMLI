import warnings
import logging
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_array
import tensorflow as tf


F32PREC = np.finfo(np.float32).eps
 
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import numpy.linalg as npla
import matplotlib.pyplot as plt
import time


class SoftImpute_ALS(object):
    

    def __init__(
        self,
        task_type = None,
        combine = False,
        auto_tune = False,
        _Lambda=2,
        convergence_threshold=0.001,
        max_iters=20,
        max_rank=None,
        init_fill_method="zero",
        min_value=None,
        max_value=None,
        normalizer=None,
        change_mode=False,
        verbose=True,
        u_group = 0,
        i_group = 0,
        val_u_group = 0,
        val_i_group = 0,
        n_oversamples=10,
        pred_tr=None,
        tr_y=None,
        pred_val=None,
        val_y=None,
        tr_Xi=None,
        val_Xi=None,
        wc = None,
        m=0,
        n=0):
            
        
        self.task_type = task_type
        self.auto_tune = auto_tune
        self.normalizer = normalizer
        self.change_mode = change_mode
        self._Lambda = _Lambda
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters
        self.max_rank = max_rank
        self.verbose = verbose
        self.u_group = u_group
        self.i_group = i_group
        self.val_u_group = val_u_group
        self.val_i_group = val_i_group
        self.n_oversamples = n_oversamples
        self.fill_method = init_fill_method
        self.min_value = min_value
        self.max_value = max_value

        self.pred_tr=pred_tr
        self.tr_y=tr_y
        self.pred_val=pred_val
        self.val_y=val_y
        self.tr_Xi = tr_Xi
        self.val_Xi = val_Xi
        self.wc = wc

        self.m = m
        self.n = n
    
    def Initialize(self):
        #U is an mxr matrix
        R = np.random.RandomState(0)
        self.U = R.randn(self.m, self.max_rank)
        #Calling QR Factorization on U
        Q,R = npla.qr(self.U)
        self.U=Q[:,0:self.max_rank]
        #U is now a matrix with orthonormal columns

        #D is an rxr identity matrix of type numpy.ndarray
        self.D=np.identity(self.max_rank)
        self.D_sq=self.D**2  
        self.A=self.U.dot(self.D)

        #V is nxr
        self.V=np.zeros(self.n*self.max_rank).reshape(self.n,self.max_rank)
        #B is an nxr matrix
        self.B=self.V.dot(self.D)
        return self.A, self.B, self.U, self.V, self.D

    def user_cluster_mean(self, u_group, A):
        match_u = dict()
        A_avg = A.copy()
        
        if type(u_group) != int :
            if len(np.unique(u_group)) > 1:
                for i in np.unique(u_group):
                    cus = np.argwhere(u_group==i)
                    group = A[cus,:].reshape(-1,A.shape[1])
                    avg = np.mean(group,axis=0)
                    A_avg[cus,:] = avg
                    match_u[i] = avg

                    
        return A_avg, match_u
    
    def item_cluster_mean(self, i_group, B):
        B_avg = B.copy()
        match_i = dict()
                    
        if type(i_group) != int :
            if len(np.unique(i_group)) > 1:
                for i in np.unique(i_group):
                    cus = np.argwhere(i_group==i)
                    group = B[cus,:].reshape(-1,B.shape[1])
                    avg = np.mean(group,axis=0)
                    B_avg[cus,:] = avg
                    match_i[i] = avg
                    
        return B_avg, match_i
                    
        

    def _converged(self, X_old, X_new, missing_mask):
        # check for convergence
        #old_missing_values = X_old[missing_mask]
        #new_missing_values = X_new[missing_mask]
        difference = X_old - X_new
        ssd = np.sum(difference ** 2)
        old_norm = np.sqrt((X_old ** 2).sum())
        # edge cases
        if old_norm == 0 or (old_norm < F32PREC and np.sqrt(ssd) > F32PREC):
            return False
        else:
            if len(self.valloss_record)>4:
                if self.valloss_record[-1] > self.valloss_record[-2]:
                    if self.valloss_record[-2] > self.valloss_record[-3]:
                        if self.valloss_record[-3] > self.valloss_record[-4]:
                            return True
            return (np.sqrt(ssd) / old_norm) < self.convergence_threshold

    def _verbose(self, X_reconstruction, i):
        
        loss = self.loss_record[-1]
        val_loss = self.valloss_record[-1]
        sign = self.sign
        
        if self.auto_tune == False:
            print(
            "MF Iter %d: observed %s=%0.6f validation %s=%0.6f" % (
            i + 1,
            sign,
            loss,
            sign,
            val_loss))
            
    def predict(self,X_re,Xi):

        pred2 = []
        for i in range(Xi.shape[0]):
            pred2.append(X_re[int(Xi[i,0]),int(Xi[i,1])])
        pred2 = np.ravel(np.array(pred2))
        final_pred = pred2
        return final_pred
    
    def predict_cold(self,u,v,s):
        v=v.T
        pred2 = []
        Xi= self.val_Xi
        val_y = self.val_y
        for i in range(Xi.shape[0]):
            if Xi[i,0] != 'cold':
                u_ = u[int(Xi[i,0])]
            else:
                u_ = self.match_u[self.val_u_group[i]]
            
            if Xi[i,1] != 'cold':
                v_ = v[int(Xi[i,1])]
            else:
                v_ = self.match_i[self.val_i_group[i]]
                
            pred_mf = np.dot(u_, np.multiply(s, v_))
            if (Xi[i,0] != 'cold') & (Xi[i,1] != 'cold'):
                pred_mf = val_y[i]
            pred2.append(pred_mf)
        pred2 = np.array(pred2)
        return pred2
    
    def clip(self, X):
        """
        Clip values to fall within any global or column-wise min/max constraints
        """
        X = np.asarray(X)
        if self.min_value is not None:
            X[X < self.min_value] = self.min_value
        if self.max_value is not None:
            X[X > self.max_value] = self.max_value
        return X
            
    


    def fit(self, X):
        if self.task_type=="Regression":
            self.sign = "MAE"
            self.loss_fn = tf.keras.losses.MeanAbsoluteError()
        elif self.task_type == 'Classification': 
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()
            self.sign = "BCE"

        self.A, self.B, self.U, self.V, self.Dsq = self.Initialize()
        
        
        self.A_avg, self.match_u = self.user_cluster_mean(self.u_group, self.A)
        self.B_avg, self.match_i = self.item_cluster_mean(self.i_group, self.B)
        
        self.missing_mask = np.isnan(X)
        self.observed = ~self.missing_mask
        self.loss_record = []
        self.valloss_record = []
        
        X_init = X.copy()
        
        self.Best_A = None
        self.Best_B = None
        Best_record = 10
        X_filled_init = self.A.dot(self.B.T)
        X_init[self.missing_mask] = X_filled_init[self.missing_mask]
        self.X_filled = X_init
        #np.copyto(self.X_filled, 0, where=self.missing_mask)


        for i in range(self.max_iters):

            #B step
            
            ABt = self.A.dot(self.B.T)
            proj_ABt = np.multiply(ABt, self.observed)
            X_star = np.multiply(X_init, self.observed) - proj_ABt + ABt - self.A.dot(self.B_avg.T)
            left_side = self.Dsq**2 + (self._Lambda * np.eye(self.max_rank))
            right_side = np.dot(self.Dsq, np.dot(self.U.T, X_star))
            B_hat = npla.solve(left_side, right_side)
            B_tilde = B_hat.T + self.B_avg
            
            self.V[:, :], D, _ = npla.svd(np.dot(B_tilde, self.Dsq), False)
            self.Dsq[:, :] = np.diag(np.sqrt(D))
            self.B[:, :] = np.dot(self.V, self.Dsq)
            self.B_avg, self.match_i = self.item_cluster_mean(self.i_group, self.B)

            #A step
            BAt = self.B.dot(self.A.T)
            proj_BAt = np.multiply(BAt, self.observed.T)
            Xt_star = np.multiply(X_init.T, self.observed.T) - proj_BAt + BAt - self.B.dot(self.A_avg.T)
            left_side = self.Dsq**2 + (self._Lambda * np.eye(self.max_rank))
            right_side = np.dot(self.Dsq, np.dot(self.V.T, Xt_star))
            A_hat = npla.solve(left_side, right_side)
            A_tilde = A_hat.T + self.A_avg
            
            self.U[:,:], D, _ = npla.svd(np.dot(A_tilde, self.Dsq), False)
            self.Dsq[:, :] = np.diag(np.sqrt(D))
            self.A[:, :] = np.dot(self.U, self.Dsq)
            self.A_avg, self.match_u = self.user_cluster_mean(self.u_group, self.A)
            
            
            #X_reconstruction = self.A.dot(self.B.T)
            #X_reconstruction = self.clip(X_reconstruction)
            

            
            AB = self.A.dot(self.B.T)
            proj_AB = np.multiply(AB, self.observed)
            M = np.multiply(X_init, self.observed) - proj_AB + AB
            self.U[:,:], self.D, R = npla.svd(np.dot(M, self.V), False)
            self.Dsq[:, :] = np.diag(np.sqrt(D))
            self.V[:, :] = np.dot(self.V, R)
            self.A[:, :] = np.dot(self.U, self.Dsq)
            self.B[:, :] = np.dot(self.V, self.Dsq)
            
            
            X_reconstruction = self.A.dot(self.B.T)
            X_reconstruction = self.clip(X_reconstruction)
            
                
                
                
            pred = self.predict(X_reconstruction,self.tr_Xi) 
            if self.wc == 'warm':
                predval = self.predict(X_reconstruction,self.val_Xi)  
            else:
                predval = self.predict_cold(self.U,self.Dsq**2,self.V)
        
            if self.task_type == 'Classification':
                         
                self.loss_record.append(self.loss_fn(self.tr_y.ravel(),tf.sigmoid(pred.ravel()+self.pred_tr.ravel()).numpy()).numpy())
                self.valloss_record.append(self.loss_fn(self.val_y.ravel(),tf.sigmoid(predval.ravel()+self.pred_val.ravel()).numpy()).numpy())
            else:
                self.loss_record.append(self.loss_fn(self.tr_y.ravel(),pred.ravel()+self.pred_tr.ravel()).numpy())
                self.valloss_record.append(self.loss_fn(self.val_y.ravel(),predval.ravel()+self.pred_val.ravel()).numpy())

            if self.valloss_record[-1] < Best_record:
                Best_record = self.valloss_record[-1]
                self.Best_A = self.A
                self.Best_B = self.B
                X_reconstruction = self.Best_A.dot(self.Best_B.T)
                X_reconstruction = self.clip(X_reconstruction)
            else:
                X_reconstruction = self.Best_A.dot(self.Best_B.T)
                X_reconstruction = self.clip(X_reconstruction)
                self.A = self.Best_A
                self.B = self.Best_B

            
            # print error on observed data
            if self.verbose:
                self._verbose(X_reconstruction, i)
                
            converged = self._converged(
                X_old=self.X_filled,
                X_new=X_reconstruction,
                missing_mask=self.missing_mask)
            
            self.X_filled[self.missing_mask] = X_reconstruction[self.missing_mask]
           # print(X_reconstruction[observed_mask])

            if converged:
                break

            
            
    def get_UVD(self):
        #return copies so as to not corrupt internal structures
        return (self.U.copy(), self.V.copy(), self.D.copy())
    
    
    def get_reconstruction(self):
        
        return self.X_filled
    
    def get_avg_latent(self):
        
        return self.match_u, self.match_i
        
        









