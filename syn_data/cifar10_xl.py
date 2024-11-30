import torch
import tqdm
import os
from diffusers import DiffusionPipeline

class NoWatermark:
    def apply_watermark(self, img):
        return img

def generate_image(class_name, base, refiner):
    prompt = f"an image of a {class_name}"
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    return image

def get_path(version, class_idx, image_idx):
    return f"/home/User/<PATH/TO/SYNTHETIC/DATA>/{version}/{class_idx}/{image_idx}.png"

fine_labels = [
    'airplane',  # id 0
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.safety_checker = lambda images, clip_input: (images, False)
base.watermark = NoWatermark()
base.to("cuda")
base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.safety_checker = lambda images, clip_input: (images, False)
refiner.watermark = NoWatermark()
refiner.to("cuda")
refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

for class_idx, class_name in tqdm.tqdm(enumerate(fine_labels)):
    for i in range(5000):
        path = get_path("xl", class_idx, i)
        image = generate_image(class_name, base, refiner).resize((32,32))
        image.save(path)
        print('class_idx:', class_idx, 'image_idx:', i)
