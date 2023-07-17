## How to run

```
git clone --recursive https://github.com/chie8842/stable_diffusion_upscaler
poetry install --no-root
poetry run python cli.py --prompt "photograph of an astronaut riding a horse"
```

Here is other options.
```
poetry run python cli.py --help
usage: cli.py [-h] [--image_url IMAGE_URL] [--prompt PROMPT] [--num_samples NUM_SAMPLES] [--batch_size BATCH_SIZE] [--decoder DECODER] [--guidance_scale GUIDANCE_SCALE]
              [--noise_aug_level NOISE_AUG_LEVEL] [--noise_aug_type NOISE_AUG_TYPE] [--sampler SAMPLER] [--steps STEPS] [--tol_scale TOL_SCALE] [--eta ETA] [--seed SEED]

Generate image with stable diffusion upscaler.

options:
  -h, --help            show this help message and exit
  --image_url IMAGE_URL
                        image url
  --prompt PROMPT       prompt
  --num_samples NUM_SAMPLES
                        num_samples
  --batch_size BATCH_SIZE
                        batch_size
  --decoder DECODER     decoder
  --guidance_scale GUIDANCE_SCALE
                        guidance_scale
  --noise_aug_level NOISE_AUG_LEVEL
                        noise_aug_level
  --noise_aug_type NOISE_AUG_TYPE
                        noise_aug_type
  --sampler SAMPLER     sampler
  --steps STEPS         steps
  --tol_scale TOL_SCALE
                        tol_scale
  --eta ETA             eta
  --seed SEED           seed
```
