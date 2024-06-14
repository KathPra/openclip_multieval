## conda env: openclip

# load packages
import torch
from PIL import Image, ImageFile
import os
#from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import glob
import numpy as np
import open_clip
import matplotlib.pyplot as plt
import tqdm

# allow to load images that exceed max size
ImageFile.LOAD_TRUNCATED_IMAGES = True

# load model
model_name = "ViT-B-32"
pretrain_dataset = "laion400m_e32"

model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrain_dataset) # quickgelu is used by openai
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Function to compute and save embeddings
def compute_embeddings(img_dir, labels_dict, model, transform,output_file):
    embeddings = {}
    for image_name in tqdm.tqdm(labels_dict.keys()):
        image_path = img_dir[image_name]

        image = Image.open(image_path)

        # Handle images with transparency
        if image.mode in ('P', 'RGBA'): image = image.convert("RGBA")
        else: image = image.convert("RGB")

        image = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            embedding = model.encode_image(image.to(device)).squeeze()#.numpy()  # Remove batch dimension and convert to numpy

        embeddings[image_name] = embedding
    
    torch.save(embeddings, output_file)

def find_all_image_paths(root_dir, img_list):
        image_paths = {}
        for dirpath, _, filenames in os.walk(root_dir): # root_dir = ../../Images/ClimateTV
            img_filenames = list(set(filenames) & set(img_list))
            for filename in img_filenames:
                image_paths[filename] = os.path.join(dirpath, filename)
        print(len(image_paths)) # 1,707,379
        return image_paths


output_file = f"../embeddings/tv_{model_name}_{pretrain_dataset}.torch"
labels_dict = torch.load(f"../../Labels/llama3/animals_llama3_final.torch")
label_list = list(labels_dict.keys())
img_dir = "../../Images/ClimateTV/"

img_path_dir = find_all_image_paths(img_dir, label_list)
print(len(img_path_dir))

compute_embeddings(img_path_dir, labels_dict, model, preprocess, output_file)
