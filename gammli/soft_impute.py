import warnings
import logging
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_array
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity


F32PREC = np.finfo(np.float32).eps


class SoftImpute(object):
    """
    Implementation of the SoftImpute algorithm from:
    "Spectral Regularization Algorithms for Learning Large Incomplete Matrices"
    by Mazumder, Hastie, and Tibshirani.
    """
    def __init__(
            self,
            task_type = None,
            combine = False,
            auto_tune = False,
            shrinkage_value=None,
            convergence_threshold=0.001,
            max_iters=20,
            max_rank=None,
            n_power_iterations=1,
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
            scale_ratio=1,
            n_oversamples=10,
            pred_tr=None,
            tr_y=None,
            pred_val=None,
            val_y=None,
            tr_Xi=None,
            val_Xi=None,
            combine_range=0.99,
            wc = None):
        """
        Parameters
        ----------
        shrinkage_value : float
            Value by which we shrink singular values on each iteration. If
            omitted then the default value will be the maximum singular
            value of the initialized matrix (zeros for missing values) divided
            by 100.

        convergence_threshold : float
            Minimum ration difference between iterations (as a fraction of
            the Frobenius norm of the current solution) before stopping.

        max_iters : int
            Maximum number of SVD iterations

        max_rank : int, optional
            Perform a truncated SVD on each iteration with this value as its
            rank.

        n_power_iterations : int
            Number of power iterations to perform with randomized SVD

        init_fill_method : str
            How to initialize missing values of data matrix, default is
            to fill them with zeros.

        min_value : float
            Smallest allowable value in the solution

        max_value : float
            Largest allowable value in the solution

        normalizer : object
            Any object (such as BiScaler) with fit() and transform() methods

        verbose : bool
            Print debugging info
        """

        self.task_type = task_type
        self.combine = combine
        self.auto_tune = auto_tune
        self.normalizer = normalizer
        self.change_mode = change_mode
        self.shrinkage_value = shrinkage_value
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters
        self.max_rank = max_rank
        self.n_power_iterations = n_power_iterations
        self.verbose = verbose
        self.u_group = u_group
        self.i_group = i_group
        self.val_u_group = val_u_group
        self.val_i_group = val_i_group
        self.scale_ratio = scale_ratio
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
        self.combine_range = combine_range
        self.wc = wc
        self.change_u=[]
        self.change_i=[]
    
        
    def masked_mae(self,X_true, X_pred, mask):
        masked_diff = X_true[mask] - X_pred[mask]
        return np.mean(np.abs(masked_diff))


    def masked_mse(self,X_true, X_pred, mask):
        masked_diff = X_true[mask] - X_pred[mask]
        return np.mean(masked_diff ** 2)
    

    def _converged(self, X_old, X_new, missing_mask):
        # check for convergence
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference ** 2)
        old_norm = np.sqrt((old_missing_values ** 2).sum())
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

    def _svd_step(self, X, shrinkage_value,tuning=False, max_rank=None):
        """
        Returns reconstructed X from low-rank thresholded SVD and
        the rank achieved.
        """
        if max_rank:
            # if we have a max rank then perform the faster randomized SVD
            (U, s, V) = randomized_svd(
                X,
                max_rank,
                power_iteration_normalizer='QR',
                n_oversamples=self.n_oversamples,
                n_iter=self.n_power_iterations)
        else:
            # perform a full rank SVD using ARPACK
            (U, s, V) = np.linalg.svd(
                X,
                full_matrices=False,
                compute_uv=True)

            # print(X_reconstruction[observed_mask])

        if tuning:
            U,V ,self.match_u, self.match_i,self.var_u, self.var_i, self.radius_u, self.radius_i = self.cluster_mean(U,V,self.u_group,self.i_group,self.u_max_d,self.i_max_d,self.scale_ratio)
            self.u_max_d,self.i_max_d = self.update_max_dis(U,V,self.u_group,self.i_group,self.u_max_d,self.i_max_d)
        else:
            self.u_max_d,self.i_max_d = self.get_max_dis(U,V,self.u_group,self.i_group)
        s_thresh = np.maximum(s - shrinkage_value, 0)
        rank = (s_thresh > 0).sum()
        s_thresh = s_thresh[:rank]
        U_thresh = U[:, :rank]
        V_thresh = V[:rank, :]
        S_thresh = np.diag(s_thresh)

        X_reconstruction = np.dot(U_thresh, np.dot(S_thresh, V_thresh))

        return X_reconstruction, rank, U_thresh, V_thresh, S_thresh

    def _max_singular_value(self, X_filled):
        # quick decomposition of X_filled into rank-1 SVD
        _, s, _ = randomized_svd(
            X_filled,
            1,
            n_iter=5)
        return s[0]
    
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
            
    

    '''
    def _verbose(self,X_init, X_val, X_reconstruction, observed_mask, val_mask, i, rank, sign):
        
        loss = self.loss_fn(
                    X_true=X_init,
                    X_pred=X_reconstruction,
                    mask=observed_mask).numpy()
        self.loss_record.append(loss)

        val_loss = self.loss_fn(
                    X_true = X_val,
                    X_pred = X_reconstruction,
                    mask = val_mask).numpy()
        self.valloss_record.append(val_loss)
        
        
        print(
            "[SoftImpute] Iter %d: observed %s=%0.6f validation %s=%0.6f,rank=%d" % (
            i + 1,
            sign,
            loss,
            sign,
            val_loss,
            rank))
        
    '''
    def _verbose(self, X_reconstruction, i, rank):
        
        loss = self.loss_record[-1]
        val_loss = self.valloss_record[-1]
        sign = self.sign
        
        if self.auto_tune == False:
            print(
            "[SoftImpute] Iter %d: observed %s=%0.6f validation %s=%0.6f,rank=%d" % (
            i + 1,
            sign,
            loss,
            sign,
            val_loss,
            rank))
    
    def solve(self, X, missing_mask):
        if self.task_type=="Regression":
            self.sign = "MAE"
            self.loss_fn = tf.keras.losses.MeanAbsoluteError()
        elif self.task_type == 'Classification': 
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()
            self.sign = "BCE"
        self.group_pre_u = {}
        self.group_pre_i = {}
        X = check_array(X, force_all_finite=False)

        X_init = X.copy()
        self.loss_record = []
        self.valloss_record = []

        X_filled = X
        observed_mask = ~missing_mask
        max_singular_value = self._max_singular_value(X_filled)
        if self.verbose:
            if self.auto_tune == False:
                print("[SoftImpute] Max Singular Value of X_init = %f" % (
                max_singular_value))

        if self.shrinkage_value:
            shrinkage_value = self.shrinkage_value
        else:
            # totally hackish heuristic: keep only components
            # with at least 1/50th the max singular value
            shrinkage_value = max_singular_value / 50.0

        if self.auto_tune == False:
            print('#####mf_training#####')

        X_reconstruction, rank, U_thresh, V_thresh, S_thresh = self._svd_step(
                X_filled,
                shrinkage_value,
                tuning=False,
                max_rank=self.max_rank)
        X_reconstruction = self.clip(X_reconstruction)

                    
        converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstruction,
                missing_mask=missing_mask)
        X_filled[missing_mask] = X_reconstruction[missing_mask]


        self.ini_u = X_filled

        for i in range(self.max_iters):
            X_reconstruction, rank, U_thresh, V_thresh, S_thresh = self._svd_step(
                X_filled,
                shrinkage_value,
                tuning=True,
                max_rank=self.max_rank)
            X_reconstruction = self.clip(X_reconstruction)
            
            pred = self.predict(X_reconstruction,self.tr_Xi) 
            if self.wc == 'warm':
                predval = self.predict(X_reconstruction,self.val_Xi)  
            else:
                predval = self.predict_cold(U_thresh,V_thresh,S_thresh)
        
            if self.task_type == 'Classification':
                #self.loss_record.append(tf.keras.losses.MeanAbsoluteError(self.tr_y.ravel(),tf.sigmoid(pred.ravel()+self.pred_tr.ravel()).numpy()).numpy())
                #self.valloss_record.append(tf.keras.losses.MeanAbsoluteError(self.val_y.ravel(),tf.sigmoid(predval.ravel()+self.pred_val.ravel()).numpy()).numpy())                
                self.loss_record.append(self.loss_fn(self.tr_y.ravel(),tf.sigmoid(pred.ravel()+self.pred_tr.ravel()).numpy()).numpy())
                self.valloss_record.append(self.loss_fn(self.val_y.ravel(),tf.sigmoid(predval.ravel()+self.pred_val.ravel()).numpy()).numpy())
            else:
                self.loss_record.append(self.loss_fn(self.tr_y.ravel(),pred.ravel()+self.pred_tr.ravel()).numpy())
                self.valloss_record.append(self.loss_fn(self.val_y.ravel(),predval.ravel()+self.pred_val.ravel()).numpy())


            # print error on observed data
            if self.verbose:
                self._verbose(X_reconstruction, i, rank)
                
            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstruction,
                missing_mask=missing_mask)
            X_filled[missing_mask] = X_reconstruction[missing_mask]
           # print(X_reconstruction[observed_mask])

            if converged:
                break



        
        if self.verbose:
            if self.auto_tune == False:
                print("[SoftImpute] Stopped after iteration %d for lambda=%f" % (
                i + 1,
                shrinkage_value))


        if self.change_mode:
            X_filled = X_reconstruction
            
        var_whole_u = np.var(U_thresh)
        var_whole_i = np.var(V_thresh.T)
        print('final num of user group:',len(self.match_u))
        print('final num of item group:',len(self.match_i))
        return X_filled, U_thresh, V_thresh, S_thresh ,self.loss_record, self.valloss_record, self.match_u, self.match_i, self.var_u, self.var_i, var_whole_u, var_whole_i, self.group_pre_u, self.group_pre_i

    def cluster_mean(self, u,v,u_group,i_group,u_max_d,i_max_d,scale_ratio):

        v=v.T
        match_u = dict()
        match_i = dict()
        var_u = dict()
        var_i = dict()
        radius_u = dict()
        radius_i = dict()


        def center_restrict(point_t,max_d,avg):
            new = point_t
            if point_t.shape[0] ==1:
                return new

            d=[]
            for i in range(point_t.shape[0]):
                d.append(np.sqrt(np.sum(np.square(point_t[i]-avg))))    
            d=np.array(d).reshape(-1,1)
            #count=0
            for j in range(point_t.shape[0]):

                if d[j] > max_d*scale_ratio:
                   # count = count+1
                    new[j] = ((1-scale_ratio)*(avg-point_t[j]))+point_t[j]
            #print(count)
            new = new.reshape(-1,1,point_t.shape[1])

            #new = (avg+(d/r*d.max())*(point_t-avg)).reshape(-1,1,3)


            return new

        def center_dis(x):
            dis = {}
            closest = {}
            adjusted = np.mean(np.array(list(x.values())),axis=0)
            for i in x.keys():
                x[i] = x[i] - adjusted
            for i in x.keys():                
                sim = []
                for j in x.keys():
                    sim.append(cosine_similarity(x[i].reshape(1,-1),x[j].reshape(1,-1))[0][0])
                sim = np.array(sim)
                similarity = np.concatenate([np.array([np.array(list(x.keys()))]).T,abs(sim).reshape(-1,1)],axis=1).T
                sorted_sim = similarity.T[np.lexsort(similarity)][:-1,:]
                dis[i] = sorted_sim
                closest[i] = similarity.T[np.lexsort(similarity)][-2,0]
                
                
            return dis, closest
        
        def center_combine(u_group, dis, closest, group_pre):

            combined = []
            changed = []
            for i in closest.keys():           
                if dis[i][dis[i][:,0]==closest[i]][:,-1] > self.combine_range:
                    if closest[i] not in combined:
                        u_group[u_group==i] = closest[i]
                        group_pre[i] = int(closest[i])
                        combined.append(i)
                        changed.append(int(closest[i]))
                    #combine[i].append(closest[i])
            return u_group, group_pre, changed


        if type(u_group) != int :
            if len(np.unique(u_group)) > 1:
                for i in np.unique(u_group):
                    cus = np.argwhere(u_group==i)
                    group = u[cus,:].reshape(-1,u.shape[1])
                    avg = np.mean(group,axis=0)
                    var = np.var(group,axis=0)
                    point_t = u[cus].reshape(-1,u.shape[1])
                    #u[cus] = center_move(point_t,u_max_d[i],avg)
                    u[cus] = center_restrict(point_t,u_max_d[i],avg)
                    #u[cus] = avg
                    match_u[i] = avg
                    var_u[i] = var
                    radius_u[i] =u_max_d[i]*scale_ratio
                if self.combine:
                    dis , closest = center_dis(match_u)
                    u_group, self.group_pre_u, self.change_u = center_combine(u_group, dis, closest, self.group_pre_u)
            else:
                i = np.unique(u_group)[0]
                cus = np.argwhere(u_group==i)
                group = u[cus,:].reshape(-1,u.shape[1])
                avg = np.mean(group,axis=0)
                var = np.var(group,axis=0)
                point_t = u[cus].reshape(-1,u.shape[1])
                #u[cus] = center_move(point_t,u_max_d[i],avg)
                u[cus] = center_restrict(point_t,u_max_d[i],avg)
                #u[cus] = avg
                match_u[i] = avg
                var_u[i] = var
                radius_u[i] =u_max_d[i]*scale_ratio
                self.change_u = []
            

        if type(i_group) != int :
            if len(np.unique(i_group)) > 1:
                for j in np.unique(i_group):
                    cus = np.argwhere(i_group==j)
                    group = v[cus,:].reshape(-1,v.shape[1])
                    avg = np.mean(group,axis=0)
                    var = np.var(group,axis=0)
                    point_t = v[cus].reshape(-1,v.shape[1])
                    #v[cus] = center_move(point_t,i_max_d[j],avg)
                    v[cus] = center_restrict(point_t,i_max_d[j],avg)
                    #v[cus] = avg
                    match_i[j] = avg
                    var_i[j] = var
                    radius_i[j] =i_max_d[j]*scale_ratio
                if self.combine:
                    dis , closest = center_dis(match_i)
                    i_group, self.group_pre_i, self.change_i = center_combine(i_group, dis, closest, self.group_pre_i)
        
            else:
                j = np.unique(i_group)[0]
                cus = np.argwhere(i_group==j)
                group = v[cus,:].reshape(-1,v.shape[1])
                avg = np.mean(group,axis=0)
                var = np.var(group,axis=0)
                point_t = v[cus].reshape(-1,v.shape[1])
                #v[cus] = center_move(point_t,i_max_d[j],avg)
                v[cus] = center_restrict(point_t,i_max_d[j],avg)
                #v[cus] = avg
                match_i[j] = avg
                var_i[j] = var
                radius_i[j] =i_max_d[j]*scale_ratio
                self.change_i = []
 
        v=v.T


        return u,v,match_u,match_i,var_u,var_i, radius_u, radius_i
    

    def get_max_dis(self, u,v,u_group,i_group):
        v=v.T

        def max_distance(point_t,avg):
            d=[]
            for i in range(point_t.shape[0]):
                d.append(np.sqrt(np.sum(np.square(point_t[i]-avg))))    
            d=np.array(d).reshape(-1,1)
            return np.percentile(d,95,interpolation='lower')

        u_max_d = {}
        if type(u_group) != int :
            for i in np.unique(u_group):
                cus = np.argwhere(u_group==i)
                group = u[cus,:].reshape(-1,u.shape[1])
                avg = np.mean(group,axis=0)           
                point_t = u[cus].reshape(-1,u.shape[1])
                u_max_d[i]=max_distance(point_t,avg)


        i_max_d = {}
        if type(i_group) != int :            
            for j in np.unique(i_group):
                cus = np.argwhere(i_group==j)
                group = v[cus,:].reshape(-1,v.shape[1])
                avg = np.mean(group,axis=0)
                point_t = v[cus].reshape(-1,v.shape[1])
                i_max_d[j]=max_distance(point_t,avg)
                #v[cus] = avg


        return u_max_d,i_max_d
    
    def update_max_dis(self, u,v,u_group,i_group,u_max_d,i_max_d):
        v=v.T
        

        
        def max_distance(point_t,avg):
            if np.isnan(avg[0]):
                return 0
            d=[]
            for i in range(point_t.shape[0]):
                d.append(np.sqrt(np.sum(np.square(point_t[i]-avg))))    
            d=np.array(d).reshape(-1,1)
            return np.percentile(d,95,interpolation='lower')
        
        if len(self.change_u)!=0:
            for i in self.change_u:
                cus = np.argwhere(u_group==i)
                group = u[cus,:].reshape(-1,u.shape[1])
                avg = np.mean(group,axis=0) 
                point_t = u[cus].reshape(-1,u.shape[1])
                u_max_d[i]=max_distance(point_t,avg)
            
        if len(self.change_i)!=0:
            for j in self.change_i:
                cus = np.argwhere(i_group==j)
                group = v[cus,:].reshape(-1,v.shape[1])
                avg = np.mean(group,axis=0)
                point_t = v[cus].reshape(-1,v.shape[1])
                i_max_d[j]=max_distance(point_t,avg)
        
        return u_max_d, i_max_d

    def prepare_input_data(self, X):
        """
        Check to make sure that the input matrix and its mask of missing
        values are valid. Returns X and missing mask.
        """
        X = check_array(X, force_all_finite=False)
        if X.dtype != "f" and X.dtype != "d":
            X = X.astype(float)

        self._check_input(X)
        missing_mask = np.isnan(X)
        self._check_missing_value_mask(missing_mask)
        return X, missing_mask

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

    def project_result(self, X):
        """
        First undo normalization and then clip to the user-specified min/max
        range.
        """
        X = np.asarray(X)
        if self.normalizer is not None:
            X = self.normalizer.inverse_transform(X)
        return self.clip(X)


    def generate_random_column_samples(column):
        col_mask = np.isnan(column)
        n_missing = np.sum(col_mask)
        if n_missing == len(column):
            logging.warn("No observed values in column")
            return np.zeros_like(column)

        mean = np.nanmean(column)
        std = np.nanstd(column)

        if np.isclose(std, 0):
            return np.array([mean] * n_missing)
        else:
            return np.random.randn(n_missing) * std + mean

    def fit_transform(self, X, y=None):
        """
        Fit the imputer and then transform input `X`

        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        X_original, missing_mask = self.prepare_input_data(X)
        observed_mask = ~missing_mask
        X = X_original.copy()
        if self.normalizer is not None:
            X = self.normalizer.fit_transform(X)
        X_filled = self.fill(X, missing_mask, inplace=True)
        if not isinstance(X_filled, np.ndarray):
            raise TypeError(
                "Expected %s.fill() to return NumPy array but got %s" % (
                    self.__class__.__name__,
                    type(X_filled)))

        X_result,U,V,S,loss_record,valloss_record,match_u,match_i, var_u, var_i, var_w_u, var_w_i, pre_u, pre_i= self.solve(X_filled, missing_mask)
        if not isinstance(X_result, np.ndarray):
            raise TypeError(
                "Expected %s.solve() to return NumPy array but got %s" % (
                    self.__class__.__name__,
                    type(X_result)))

        X_result = self.project_result(X=X_result)
        if self.auto_tune == False:
            print('change mode state :',self.change_mode)
        if self.change_mode==False:
            X_result[observed_mask] = X_original[observed_mask]
        return X_result, U, V, S , loss_record , valloss_record , match_u , match_i, var_u, var_i, var_w_u, var_w_i ,pre_u, pre_i

    def _check_input(self, X):
        if len(X.shape) != 2:
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))

    def _check_missing_value_mask(self, missing):
        if not missing.any():
            warnings.simplefilter("always")
            warnings.warn("Input matrix is not missing any values")
        if missing.all():
            raise ValueError("Input matrix must have some non-missing values")

    def _fill_columns_with_fn(self, X, missing_mask, col_fn):
        for col_idx in range(X.shape[1]):
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()
            if n_missing == 0:
                continue
            col_data = X[:, col_idx]
            fill_values = col_fn(col_data)
            if np.all(np.isnan(fill_values)):
                fill_values = 0
            X[missing_col, col_idx] = fill_values

    def fill(
            self,
            X,
            missing_mask,
            fill_method=None,
            inplace=False):
        """
        Parameters
        ----------
        X : np.array
            Data array containing NaN entries

        missing_mask : np.array
            Boolean array indicating where NaN entries are

        fill_method : str
            "zero": fill missing entries with zeros
            "mean": fill with column means
            "median" : fill with column medians
            "min": fill with min value per column
            "random": fill with gaussian samples according to mean/std of column

        inplace : bool
            Modify matrix or fill a copy
        """
        X = check_array(X, force_all_finite=False)

        if not inplace:
            X = X.copy()

        if not fill_method:
            fill_method = self.fill_method

        if fill_method not in ("zero", "mean", "median", "min", "random"):
            raise ValueError("Invalid fill method: '%s'" % (fill_method))
        elif fill_method == "zero":
            # replace NaN's with 0
            X[missing_mask] = 0
        elif fill_method == "mean":
            self._fill_columns_with_fn(X, missing_mask, np.nanmean)
        elif fill_method == "median":
            self._fill_columns_with_fn(X, missing_mask, np.nanmedian)
        elif fill_method == "min":
            self._fill_columns_with_fn(X, missing_mask, np.nanmin)
        elif fill_method == "random":
            self._fill_columns_with_fn(
                X,
                missing_mask,
                col_fn=generate_random_column_samples)
        return X






