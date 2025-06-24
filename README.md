# CalDiff
Code for manuscript *Step-wise and sequence-aware uncertainty calibration in diffusion model for reliable lesion segmentation*
submitted to IEEE JBHI

## Environment setup
```bash
conda create -n py39_pt19 python=3.9 -y
conda activate py39_pt19
pip install -r requirements.txt
```

## Dataset
### Lung nodule segmentation
Preprocessed dataset available at
https://github.com/stefanknegt/Probabilistic-Unet-Pytorch

### Multiple scelerosis (MS) lesion segmentation
https://smart-stats-tools.org/lesion-challenge-2015

### File Structure
```
data
└───training
│   └───0
│       │   image_0.npz
│       │   label0_.npz
│       │   label1_.npz
│       │   label2_.npz
│       │   label3_.npz
│   └───1
│       │  ...
└───testing
│   └───3
│       │   image_3.npz
│       │   label0_.npz
│       │   label1_.npz
│       │   label2_.npz
│       │   label3_.npz
│   └───4
│       │  ...
```

## Train, Val and Test 

Train and validation
``` bash
CUDA_VISIBLE_DEVICES=0,1,2 ./miniconda3/envs/py39_pt19/bin/python -m torch.distributed.launch --master_port 43980 --nproc_per_node=3 --nnodes=1 scripts/segmentation_train.py --data_dir ./data/training --image_size 128 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --train_stop 20000 --batch_size 16
```

Test including evaluation metrics
``` bash
CUDA_VISIBLE_DEVICES=1 ./miniconda3/envs/py39_pt19/bin/python -m torch.distributed.launch --master_port 49965 --nproc_per_node=1 --nnodes=1 scripts/segmentation_sample.py  --isblur 0 --data_dir ./data/testing  --model_path ./results/savedmodel100000.pt --num_ensemble 16 --image_size 128 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False
```

## Dual Mechanism

Function training_losses_segmentation() from guided_diffusion/gaussian_diffusion.py
where $L_{\text{KL}}$ and $L_{\text{KL}}$ are defined and weighted

Calibration module at guided_diffusion/distribution.py

## Reference codes

https://github.com/stefanknegt/Probabilistic-Unet-Pytorch

https://github.com/JuliaWolleb/Diffusion-based-Segmentation

https://github.com/aimansnigdha/Ambiguous-Medical-Image-Segmentation-using-Diffusion-Models
