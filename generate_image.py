import torch
from diffusers import StableDiffusionPipeline


pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)

pipe = pipe.to('cuda')

def generate_image(prompt):
    image = pipe(prompt).image[0]
    return image

