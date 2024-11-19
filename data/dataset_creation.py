import torch
import torchvision
import random 
from PIL import Image
import numpy as np
from datasets import load_dataset
import io

ds = load_dataset("nlphuji/flickr30k", split="test") 
patch_size = 16


# Function to convert each image to 16x16 patches
def patch_image(image_bytes, patch_size):
    image = image_bytes.convert("RGB")
    width, height = image.size
    
    # Ensure dimensions are divisible by patch_size
    if width % patch_size != 0 or height % patch_size != 0:
        width = (width // patch_size) * patch_size
        height = (height // patch_size) * patch_size
        image = image.resize((width, height))

    # Convert image to NumPy array for patching
    img_array = np.array(image)
    patches = []

    # Slide through the image and create patches
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = img_array[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    return np.array(patches)  # Return an array of patches

# Apply the patching function to each image in the dataset
ds = ds.map(lambda x: {"patches": patch_image(x["image"], patch_size)})

# Example: Check the patches for the first image
print(f"Number of patches in the first image: {len(ds[0]['patches'])}")

torch.save(ds, 'Transformer_project/flickr30k_patches.pt')