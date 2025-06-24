"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import cv2
from datetime import datetime
import csv
from time import gmtime, strftime
from csv import writer
from medpy.metric import jc
import argparse
import os
import torch
import nibabel as nib
from visdom import Visdom
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.lidcloader import LIDCDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    if pred.sum()+targs.sum()==0:
        return 1
    else:
        pred = (pred>0)*1
        return 2. * (pred*targs).sum() / (pred.sum()+targs.sum())
    
def dice_max(samples,gts):
    num_labels=gts.shape[0]
    num_ensembles=samples.shape[0]
    per_D_max_list = []
    for ii in range(num_labels):
        per_expert_D_max=0
        for jj in range(num_ensembles):
            now_D_max = dice_score(samples[jj],gts[ii]) 
            if now_D_max > per_expert_D_max:
                per_expert_D_max = now_D_max 
        per_D_max_list.append(per_expert_D_max)
    return np.mean(np.array(per_D_max_list))

def diversity_agreement(samples,gts):
    def max_min_var(samples):
        max_dice=0
        min_dice=1
        for ii in range(samples.shape[0]):
            for jj in range(ii+1,samples.shape[0]):
                now_dice = dice_score(samples[ii],samples[jj])
                if now_dice > max_dice:
                    max_dice = now_dice
                if now_dice < min_dice:
                    min_dice = now_dice
        max_var = 1 - min_dice
        min_var = 1 - max_dice
        return max_var, min_var
    gt_max_var, gt_min_var = max_min_var(gts)
    pred_max_var, pred_min_var = max_min_var(samples)
    return 1 - (np.abs(gt_max_var-pred_max_var)+np.abs(gt_min_var-pred_min_var))/2
        
def sensitivity_combined(samples,gts):
    pred_c = (np.sum(samples,axis=0)>0)*1
    gt_c = (np.sum(gts,axis=0)>0)*1
    if np.sum(gt_c+pred_c)==0:
        return 1
    else:
        return np.sum(pred_c*gt_c)/np.sum(gt_c)

def generalised_energy_distance(samples, gts):
    def mask_IoU(prediction, groundtruth):
        intersection = np.logical_and(groundtruth, prediction)
        union = np.logical_or(groundtruth, prediction)
        if np.sum(union) == 0:
            return 1
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score
    
    D1 = 0
    for i in range(np.size(samples,0)):
        for j in range(np.size(gts,0)):
            D1 += 1 - mask_IoU(samples[i], gts[j])
    D2 = 0
    for i in range(np.size(samples,0)):
        for j in range(np.size(samples,0)):
            D2 += 1 - mask_IoU(samples[i], samples[j])
    D3 = 0
    for i in range(np.size(gts,0)):
        for j in range(np.size(gts,0)):
            D3 += 1 - mask_IoU(gts[i], gts[j])
            
    n, m = np.size(samples,0), np.size(gts,0)
    D = 2/(n*m)*D1 - 1/(n*n)*D2 - 1/(m*m)*D3
    return D,(D2+0.00001)/(n*n)

def ncc(a, v, zero_norm=True):
    a = a.flatten()
    v = v.flatten()

    if zero_norm:
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)
    else:
        a = a / (np.std(a) * len(a))
        v = v / np.std(v)

    return np.correlate(a, v)

def pixel_wise_xent(m_samp, m_gt, eps=1e-8):
    log_samples_p = np.log(m_samp + eps)
    log_samples_n = np.log((1 - m_samp) + eps)
    return -1.0 * (m_gt * log_samples_p + (1 - m_gt) * log_samples_n)

def variance_ncc_dist(samples, groundtruths):
    mean_seg = np.mean(samples, axis=0)

    N = np.size(samples,0)
    M = np.size(groundtruths,0)

    sX = np.size(samples,1)
    sY = np.size(samples,2)

    E_ss_arr = np.zeros((N, sX, sY))
    for i in range(N):
        E_ss_arr[i, ...] = pixel_wise_xent(samples[i, ...], mean_seg)

    E_ss = np.mean(E_ss_arr, axis=0)

    E_sy_arr = np.zeros((M, N, sX, sY))
    for j in range(M):
        for i in range(N):
            E_sy_arr[j, i, ...] = pixel_wise_xent(samples[i, ...], groundtruths[j, ...])

    E_sy = np.mean(E_sy_arr, axis=1)

    ncc_list = []
    for j in range(M):
        ncc_list.append(ncc(E_ss, E_sy[j, ...]))

    return (1 / M) * sum(ncc_list)


def main():
    args = create_argparser().parse_args()
    args = create_argparser().parse_args()
    
    world_size = args.ngpu
    
    torch.distributed.init_process_group('nccl')
    
    logger.configure(dir='./results_'+strftime("%m-%d-%H-%M", gmtime()))

    logger.log("creating model and diffusion...")
    model, diffusion, prior, posterior = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
        
    torch.cuda.set_device(args.local_rank)
    
    model.to(dist_util.dev())
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
)
    
    ds = LIDCDataset(args.data_dir, test_flag=True)
    
        
    sampler = torch.utils.data.distributed.DistributedSampler(
    ds,
    num_replicas=args.ngpu,
    rank=args.local_rank,
)
  
    
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        sampler = sampler,
        shuffle=False)
    data = iter(datal)
    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    GED=[]
    D_max=[]
    CI=[]
    D_a=[]
    S_c=[]
    NCC=[]
    DIV=[]

    new_row=[str(args.model_path), str(args.isblur), 'per_GED', 'per_CI', 'per_D_max', 'per_D_a', 'per_S_c','per_NCC']
    with open('metrics_bs_8_100k.csv','a') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(new_row)
        f_object.close()

    try:
        while len(all_images) * args.batch_size < args.num_samples:
            b, label, path = next(data)  #should return an image from the dataloader "data"
            c = th.randn_like(b[:, :1, ...])

            img = th.cat((b, c), dim=1)     #add a noise channel$

            slice_ID=path[0].split("/", -1)[6]
            
            gts=label[0,:,:,:].detach().cpu().numpy()
            if np.max(gts)==0:
                print(slice_ID)

            logger.log("sampling...")

            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)

            samples = np.zeros([args.num_ensemble,args.image_size,args.image_size])
            
            for i in range(args.num_ensemble):  #this is for the generation of an ensemble of n masks.
                model_kwargs = {}
                start.record()
                sample_fn = (
                    diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                )
                
                sample, x_noisy, org = sample_fn(
                    model,
                    (args.batch_size, img.shape[1], args.image_size, args.image_size), img,#iimg=b[:, :1, ...], mask=None,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )

                end.record()
                th.cuda.synchronize()
                print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample
                if args.isblur>1:
                    sample_trh=np.uint8((sample[0,0,:,:].detach().cpu().numpy()>0.5)*255)
                    sample_blur = cv2.medianBlur(sample_trh, int(args.isblur))/255
                    samples[i]=sample_blur

                else:
                    sample_trh=np.uint8((sample[0,0,:,:].detach().cpu().numpy()>0.5)*1)
                    samples[i]=sample_trh

            
            per_S_c = sensitivity_combined(samples,gts)

            per_GED,per_div = generalised_energy_distance(samples,gts)
            per_D_max= dice_max(samples,gts)
            per_D_a = diversity_agreement(samples,gts)
            
            per_CI = 3.0*per_S_c*per_D_a*per_D_max/(per_S_c+per_D_a+per_D_max)

            per_NCC = variance_ncc_dist(samples, gts)[0]

            GED.append(per_GED)
            D_max.append(per_D_max)
            D_a.append(per_D_a)
            S_c.append(per_S_c)
            CI.append(per_CI)
            NCC.append(per_NCC)
            DIV.append(per_div)

            print('GED ', per_GED)
            print('CI ', per_CI)
            print('D_max ', per_D_max)
            print('D_a ', per_D_a)
            print('S_c ', per_S_c)
            print('NCC ', per_NCC)
            print('DIV ', per_div)

            new_row=['', str(slice_ID), per_GED, per_CI, per_D_max, per_D_a, per_S_c, per_NCC,per_div]
            with open('metrics.csv','a') as f_object:
                writer_object = csv.writer(f_object)
                writer_object.writerow(new_row)
                f_object.close()

    except StopIteration:    
        pass
        # GED_score = np.mean(np.array(GED))
        # D_max_score = np.mean(np.array(D_max))
        # D_a_score = np.mean(np.array(D_a))
        # S_c_score = np.mean(np.array(S_c))
        # CI_score = np.mean(np.array(CI))
        # NCC_score = np.mean(np.array(NCC))
        # DIV_score = np.mean(np.array(DIV))

        # now = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        # model_path = args.model_path
        # new_row=[now, model_path, GED_score, CI_score, D_max_score, D_a_score, S_c_score, NCC_score,DIV_score]        
        # with open('metrics.csv','a') as f_object:
        #     writer_object = csv.writer(f_object)
        #     writer_object.writerow(new_row)
        #     f_object.close()



def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=1)
    parser.add_argument('--ngpu', type=int, default=1)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
