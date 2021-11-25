import pickle as pkl
import sys
import os
import numpy as np
import time

# Add our source directory to the path as Python doesn't like sub-directories
source_path = os.path.join("..", "dlaas_source")
sys.path.insert(0, source_path)
from grid_search import GridSearch

rseed_list = list(np.random.randint(0, 10000, 1))
print(rseed_list)

s = '''#!/bin/bash

#SBATCH -p barak_ysinger_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH -t 800
#SBATCH -o /n/ANONYMOUS01/ANONYMOUS/sbatchfiles/noise_ens/check%j.out
#SBATCH -e /n/ANONYMOUS01/ANONYMOUS/sbatchfiles/noise_ens/check%j.err

module load Anaconda3/5.0.1-fasrc02
module load cuda/10.0.130-fasrc01 cudnn/7.4.1.5_cuda10.0-fasrc01

source activate py36torch04
cd /n/ANONYMOUS01/ANONYMOUS/main-exp/ANONYMOUS/

{0}


exit 0;
'''

project_name = sys.argv[1]

def sgd_const():
    search = GridSearch()
    search.add_list("rseed", rseed_list)
    search.add_static_var("gpu", '')

    search.add_list("dataname", ['CelebA-Arched_Eyebrows-Attractive', 'CelebA-Pointy_Nose-Attractive','CelebA-No_Beard-Attractive', 'CelebA-Rosy_Cheeks-Attractive', 'CelebA-Blond_Hair-Attractive', 'CelebA-Male-Attractive', 'CelebA-Eyeglasses-Attractive', 'CelebA-Blurry-High_Cheekbones'])
    search.add_static_var("dataroot", '../../supervised-disentangled/data/hdd_c/')
    search.add_static_var("numsamples", 50000) #[500, 1000, 2000, 5000, 10000, 20000]
    search.add_static_var("batchsize", 32)
    search.add_static_var("testbatchsize", 32)

    search.add_static_var("noise_method", 'cub')

    search.add_static_var("tr_workers", 2)
    search.add_static_var("val_workers", 2)
    search.add_static_var("test_workers", 2)
    search.add_static_var("tr_val_workers", 2)

    search.add_static_var("classifiername", 'cub_resnet50')

    search.add_static_var("numout", 2)
    search.add_static_var("weightscale", 1.0)
    
    search.add_static_var("optimname", 'sgd')
    search.add_static_var("losstype", 'ce')

    search.add_static_var("lr_sched_type", 'dynamic')
    search.add_list("lr", [0.001])
    search.add_static_var("lr_drop_factor", 0.1)
    search.add_static_var("lr_patience", 2000)
    search.add_static_var("lr_monitor", 'train_loss')
    search.add_static_var("weight_decay_coeff", 1e-4)
    
    search.add_static_var("total_iterations", 60000)
    search.add_static_var("tr_loss_thresh", 1e-6)

    search.add_static_var("wandb_project_name", 'celeba-trial-attributes')    
    
    search.add_static_var("eval_iter", 1000)
    search.add_static_var("verbose", '')
    search.add_static_var("saveplot", '')
    search.add_static_var("savefile", '')
    search.add_static_var("savepath", '..')
    search.add_static_var("gcs_root", 'logs')
    search.add_static_var("wandb_api_key_path", '/n/ANONYMOUS01/ANONYMOUS/settings/wandb_api_key.txt')
    search.add_static_var("gcs_key_path", '/n/ANONYMOUS01/ANONYMOUS/settings/ANONYMOUS-b1e9f085f023.json')

    return search.create_grid_search()

runs = sgd_const()
run_dict = []

for index, run_params in enumerate(runs):
    command = "python3 ../source/noise_ens_dev.py"
    for name in run_params:
        command = "%s --%s %s" % (command, name, str(run_params[name]))
    slurm_file_in = s.format(command)
    with open('tmp.slurm', 'w') as f:
        f.write(slurm_file_in)
    term_out = os.system('sbatch tmp.slurm')
    
    # Store run commands and slurm output
    dict = {}
    dict['slurm_file'] = slurm_file_in
    dict['term_out'] = term_out
    
    run_dict.append(dict)
    
savepath = '/n/ANONYMOUS01/ANONYMOUS/sbatchfiles/noise_ens/run_details/' +  project_name + '_'+time.strftime("%Y-%m-%d_%H-%M-%S")+'.pkl'
with open(savepath, 'wb') as f:
    pkl.dump(run_dict, f)
