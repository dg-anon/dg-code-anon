# Reproducing experiments for Distributional Generalization

### Overview of the main code
The file `source/noise_ens_dev.py` is the main code file that trains a model according to specifications
The flags that go into it include
- Dataset and accompanying parameters (CIFAR10/CIFAR100, number of samples etc)
- "noise_method" a string which specifies the type of noise added to the labels and associated parameters
- Model and associated params (architecture, width etc)
- Training method and associated params (learning rate, epochs etc)

The list of flags for all experiments are in individual files in the folder `run_configs`

For example, `run_configs/targeted/cifar10_wrn_oa_cat.py` contains the flags for Experiment 1
You can also run the following command to reproduce Experiment 1:

```python3 ../source/noise_ens_dev.py --gpu --dataname CIFAR10withInds --dataroot <CIFAR-10 directory path> --data_augment --numsamples -1 --batchsize 128 --testbatchsize 128 --noise_method fl_true_cm_supercat --noise_probability 0.3 --tr_workers 2 --val_workers 2 --test_workers 2 --tr_val_workers 2 --classifiername wideresnet_pic_myinit --numout 10 --weightscale 1.0 --optimname momentum --losstype ce --lr_sched_type at_epoch --lr 0.1 --lr_drop_factor 0.2 --lr_epoch_list 60 120 160 --momentum 0.9 --weight_decay_coeff 0.0005 --total_iterations 80000 --tr_loss_thresh 1e-06 --wandb_project_name supercat-cifar-wrn --eval_iter 2000 --verbose --saveplot --savefile --savepath .. --gcs_root logs --wandb_api_key_path <Your wandb key> --gcs_key_path <Your google cloud key JSON>```

### Additional Code
The code for the kernel experiments are in `kernels`.
The code for the decision-tree experiments on UCI are in `uci`.
The code for parsing imagenet results are in `imagenet`.
This code is provided as-is, anonymized, and so may require modifications to run on your specific compute setup.