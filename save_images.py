import os
import re

save_to_drive = False
save_location = 'stable-diffusion-upscaler/%T-%I-%P.png'

if save_to_drive:
  from google.colab import drive
  drive.mount('/content/drive')
  save_location = '/content/drive/MyDrive/' + save_location

def clean_prompt(prompt):
  badchars = re.compile(r'[/\\]')
  prompt = badchars.sub('_', prompt)
  if len(prompt) > 100:
    prompt = prompt[:100] + '…'
  return prompt

def format_filename(timestamp, seed, index, prompt):
  string = save_location
  string = string.replace('%T', f'{timestamp}')
  string = string.replace('%S', f'{seed}')
  string = string.replace('%I', f'{index:02}')
  string = string.replace('%P', clean_prompt(prompt))
  return string

def save_image(image, **kwargs):
  filename = format_filename(**kwargs)
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  image.save(filename)
