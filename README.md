# GAMMLI
Explainable Recommendation Systems by Generalized Additive Models with Manifest and Latent Interactions

The following environments are required:

- Python 3.7 (anaconda is preferable)
- tensorflow 2.0

## Usage

Import library
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import time
from sklearn.metrics import mean_squared_error,roc_auc_score,mean_absolute_error,log_loss
import sys
sys.path.append('../')
from gammli.GAMMLI import GAMMLI
from gammli.DataReader import data_initialize
from gammli.utils import local_visualize
from gammli.utils import global_visualize_density
from gammli.utils import feature_importance_visualize
from gammli.utils import plot_trajectory
from gammli.utils import plot_regularization
import tensorflow as tf

```

Load data 
```python
random_state = 0
data= pd.read_csv('data/simulation/sim_0.9_new.csv')
task_type = "Regression"

meta_info = OrderedDict()

meta_info['user_x_1']={'type': 'continues','source':'user'}
meta_info['user_x_2']={'type': 'continues','source':'user'}
meta_info['user_x_3']={'type': 'continues','source':'user'}
meta_info['user_x_4']={'type': 'continues','source':'user'}
meta_info['user_x_5']={'type': 'continues','source':'user'}
meta_info['item_x_1']={'type': 'continues','source':'item'}
meta_info['item_x_2']={'type': 'continues','source':'item'}
meta_info['item_x_3']={'type': 'continues','source':'item'}
meta_info['item_x_4']={'type': 'continues','source':'item'}
meta_info['item_x_5']={'type': 'continues','source':'item'}
meta_info['user_id']={"type":"id",'source':'user'}
meta_info['item_id']={"type":"id",'source':'item'}
meta_info['target']={"type":"target",'source':''}
```

Run GAMMLI
```python
tr_x, tr_Xi, tr_y, tr_idx, te_x, te_Xi, te_y, val_x, val_Xi, val_y, val_idx, meta_info, model_info,sy,sy_t = data_initialize(train,test,meta_info,task_type ,'warm', random_state, True)
model = GAMMLI(wc='warm',model_info=model_info, meta_info=meta_info, subnet_arch=[20, 10],interact_arch=[20, 10],activation_func=tf.tanh, batch_size=min(500, int(0.2*tr_x.shape[0])), lr_bp=0.001, auto_tune=False,
               interaction_epochs=1000,main_effect_epochs=1000,tuning_epochs=200,loss_threshold_main=0.01,loss_threshold_inter=0.1,
              verbose=True, early_stop_thres=20,interact_num=10,n_power_iterations=5,n_oversamples=10, u_group_num=10, i_group_num=10, reg_clarity=10, lambda_=5,
              mf_training_iters=200,change_mode=False,convergence_threshold=0.0001,max_rank=3,interaction_restrict='intra', si_approach ='als')
model.fit(tr_x, val_x, tr_y, val_y, tr_Xi, val_Xi, tr_idx, val_idx)
```
Training Logs
```python 
simu_dir = "./results/"
if not os.path.exists(simu_dir):
    os.makedirs(simu_dir)

data_dict_logs = model.summary_logs(save_dict=False)
plot_trajectory(data_dict_logs, folder=simu_dir, name="s1_traj_plot", log_scale=True, save_png=True)
plot_regularization(data_dict_logs, folder=simu_dir, name="s1_regu_plot", log_scale=True, save_png=True)
```
![traj_visu_demo](https://github.com/gyf9712/GAMMLI/blob/main/examples/result/simulation/reg_tra.png)


Global Visualization
```python 
importance = model.get_all_rank(tr_Xi)
model.dash_board(data_dict, importance,simu_dir,True)
```
![global_visu_demo](https://github.com/gyf9712/GAMMLI/blob/main/examples/result/simulation/reg_glo.png)
![global latent_visu_demo](https://github.com/gyf9712/GAMMLI/blob/main/examples/result/simulation/reg_latent.png)
