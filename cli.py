import numpy as np
import time
import re
import requests
import io
import hashlib
from subprocess import Popen

import torch
from torch import nn
import torch.nn.functional as F

from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torchvision.utils import make_grid
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm, trange
from functools import partial
from IPython.display import display
from ipywidgets import widgets

from ldm.util import instantiate_from_config
import argparse
import k_diffusion as K

from fetch_model import *
from functions import *
from load_model import load_model_from_config
from run_model import run
from generate_image import generate_image


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Generate image with stable diffusion upscaler.")
    parser.add_argument(
        "--image_url",
        type=str,
        default="https://models.rivershavewings.workers.dev/assets/sd_2x_upscaler_demo.png",
        required=False,
        help="image url", #TODO: fix the comment
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="the temple of fire by Ross Tran and Gerardo Dottori, oil on canvas",
        required=False,
        help=("prompt"),# TODO: fix the comment
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        required=False,
        help=("num_samples"),# TODO: fix the comment
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        required=False,
        help=("batch_size"),# TODO: fix the comment
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="finetuned_840k",
        required=False,
        help=("decoder"),# TODO: fix the comment
    )
    parser.add_argument(
        "--guidance_scale",
        type=int,
        default=1,
        required=False,
        help=("guidance_scale"),# TODO: fix the comment
    )
    parser.add_argument(
        "--noise_aug_level",
        type=int,
        default=0,
        required=False,
        help=("noise_aug_level"),# TODO: fix the comment
    )
    parser.add_argument(
        "--noise_aug_type",
        type=str,
        default='gaussian',
        required=False,
        help=("noise_aug_type"),# TODO: fix the comment
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default='k_dpm_adaptive',
        required=False,
        help=("sampler"),# TODO: fix the comment
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        required=False,
        help=("steps"),# TODO: fix the comment
    )
    parser.add_argument(
        "--tol_scale",
        type=float,
        default=0.25,
        required=False,
        help=("tol_scale"),# TODO: fix the comment
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        required=False,
        help=("eta"),# TODO: fix the comment
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        required=False,
        help=("seed"),# TODO: fix the comment
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    #save_to_drive(save_to_drive, save_location)
  
    model_up = make_upscaler_model(
            fetch('https://models.rivershavewings.workers.dev/config_laion_text_cond_latent_upscaler_2.json'),
            fetch('https://models.rivershavewings.workers.dev/laion_text_cond_latent_upscaler_2_1_00470000_slim.pth'))
    vae_840k_model_path = download_from_huggingface(
            "stabilityai/sd-vae-ft-mse-original", 
            "vae-ft-mse-840000-ema-pruned.ckpt")
    vae_560k_model_path = download_from_huggingface(
            "stabilityai/sd-vae-ft-ema-original",
            "vae-ft-ema-560000-ema-pruned.ckpt")
  
    cpu = torch.device("cpu")
    device = torch.device("cuda")
  
    # sd_model = load_model_from_config(
    #    "stable-diffusion/configs/stable-diffusion/v1-inference.yaml", 
    #    sd_model_path)
    vae_model_840k = load_model_from_config(
            "latent-diffusion/models/first_stage_models/kl-f8/config.yaml",
            vae_840k_model_path,
            cpu,
            device)
    vae_model_560k = load_model_from_config(
            "latent-diffusion/models/first_stage_models/kl-f8/config.yaml", 
            vae_560k_model_path,
            cpu,
            device)
  
    # sd_model = sd_model.to(device)
    vae_model_840k = vae_model_840k.to(device)
    vae_model_560k = vae_model_560k.to(device)
    model_up = model_up.to(device)
  
    # Not strictly required but can subtly affect the upscaling result.
    prompt = args.prompt
    num_samples = args.num_samples
    batch_size = args.batch_size 
    
    decoder = args.decoder #["finetuned_840k", "finetuned_560k"]
    
    guidance_scale = args.guidance_scale # min: 0.0, max: 10.0
    
    # Add noise to the latent vectors before upscaling. 
    # This theoretically can make the model work better on out-of-distribution inputs, 
    # but mostly just seems to make it match the input less, so it's turned off by default.
    noise_aug_level = args.noise_aug_level # min: 0.0, max: 0.6
    noise_aug_type = args.noise_aug_type # ["gaussian", "fake"]
    
    # Sampler settings. `k_dpm_adaptive` uses an adaptive solver with error tolerance `tol_scale`, 
    # all other use a fixed number of steps.
    sampler = args.sampler # ["k_euler", "k_euler_ancestral", "k_dpm_2_ancestral", "k_dpm_fast", "k_dpm_adaptive"]
    steps = args.steps
    tol_scale = args.tol_scale
    # Amount of noise to add per step (0.0=deterministic). Used in all samplers except `k_euler`.
    eta = args.eta #@param {type: 'number'}
    
    # Set seed to 0 to use the current time:
    seed = args.seed 

    #input_image = Image.open(fetch(args.image_url)).convert('RGB')
    input_image = generate_image(prompt)
    run(
        input_image, seed, batch_size, prompt, noise_aug_level, device, decoder, 
        guidance_scale, num_samples, noise_aug_type, sampler, steps, tol_scale,
        eta, model_up, vae_model_840k, vae_model_560k)

if __name__ == "__main__":
    args = parse_args()
    main(args)


