import k_diffusion as K
import numpy as np
import time
import torch
from pytorch_lightning import seed_everything
from torchvision.transforms import functional as TF
from functions import *
from save_images import *

# Model configuration values
SD_C = 4 # Latent dimension
SD_F = 8 # Latent patch size (pixels per latent)
SD_Q = 0.18215 # sd_model.scale_factor; scaling for latents in first stage models

@torch.no_grad()
def condition_up(prompts, text_encoder_up, tok_up):
  return text_encoder_up(tok_up(prompts))

@torch.no_grad()
def run(
  input_image, seed, batch_size, prompt, noise_aug_level, device, decoder, 
  guidance_scale, num_samples, noise_aug_type, sampler, steps, tol_scale, eta,
  model_up, vae_model_840k, vae_model_560k):
  timestamp = int(time.time())
  if not seed:
    print('No seed was provided, using the current time.')
    seed = timestamp
  print(f'Generating with seed={seed}')
  seed_everything(seed)

  tok_up = CLIPTokenizerTransform()
  text_encoder_up = CLIPEmbedder(device=device)

  uc = condition_up(batch_size * [""], text_encoder_up, tok_up)
  c = condition_up(batch_size * [prompt], text_encoder_up, tok_up)

  if decoder == 'finetuned_840k':
    vae = vae_model_840k
  elif decoder == 'finetuned_560k':
    vae = vae_model_560k

  # image = Image.open(fetch(input_file)).convert('RGB')
  image = input_image
  image = TF.to_tensor(image).to(device) * 2 - 1
  low_res_latent = vae.encode(image.unsqueeze(0)).sample() * SD_Q
  low_res_decoded = vae.decode(low_res_latent/SD_Q)

  [_, C, H, W] = low_res_latent.shape

  # Noise levels from stable diffusion.
  sigma_min, sigma_max = 0.029167532920837402, 14.614642143249512

  model_wrap = CFGUpscaler(model_up, uc, cond_scale=guidance_scale)
  low_res_sigma = torch.full([batch_size], noise_aug_level, device=device)
  x_shape = [batch_size, C, 2*H, 2*W]

  def do_sample(noise, extra_args):
    # We take log-linear steps in noise-level from sigma_max to sigma_min, using one of the k diffusion samplers.
    sigmas = torch.linspace(np.log(sigma_max), np.log(sigma_min), steps+1).exp().to(device)
    if sampler == 'k_euler':
      return K.sampling.sample_euler(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args)
    elif sampler == 'k_euler_ancestral':
      return K.sampling.sample_euler_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta)
    elif sampler == 'k_dpm_2_ancestral':
      return K.sampling.sample_dpm_2_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta)
    elif sampler == 'k_dpm_fast':
      return K.sampling.sample_dpm_fast(model_wrap, noise * sigma_max, sigma_min, sigma_max, steps, extra_args=extra_args, eta=eta)
    elif sampler == 'k_dpm_adaptive':
      sampler_opts = dict(s_noise=1., rtol=tol_scale * 0.05, atol=tol_scale / 127.5, pcoeff=0.2, icoeff=0.4, dcoeff=0)
      return K.sampling.sample_dpm_adaptive(model_wrap, noise * sigma_max, sigma_min, sigma_max, extra_args=extra_args, eta=eta, **sampler_opts)

  image_id = 0
  for _ in range((num_samples-1)//batch_size + 1):
    if noise_aug_type == 'gaussian':
      latent_noised = low_res_latent + noise_aug_level * torch.randn_like(low_res_latent)
    elif noise_aug_type == 'fake':
      latent_noised = low_res_latent * (noise_aug_level ** 2 + 1)**0.5
    extra_args = {'low_res': latent_noised, 'low_res_sigma': low_res_sigma, 'c': c}
    noise = torch.randn(x_shape, device=device)
    up_latents = do_sample(noise, extra_args)

    pixels = vae.decode(up_latents/SD_Q) # equivalent to sd_model.decode_first_stage(up_latents)
    pixels = pixels.add(1).div(2).clamp(0,1)


    # Display and save samples.
    # display(TF.to_pil_image(make_grid(pixels, batch_size)))
    for j in range(pixels.shape[0]):
      img = TF.to_pil_image(pixels[j])
      save_image(img, timestamp=timestamp, index=image_id, prompt=prompt, seed=seed)
      image_id += 1
