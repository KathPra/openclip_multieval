## conda env: openclip

# load packages
import torch
from PIL import Image
import open_clip
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import glob
from collections import defaultdict

# load model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

# Create custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, txt_file, img_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with image names and labels.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        label_counter = 0

        with open(txt_file, 'r') as file:
            for line in file:
                image_name, label = line.strip().split(";")
                if label not in self.label_to_idx:
                    self.label_to_idx[label] = label_counter
                    self.idx_to_label[label_counter] = label
                    label_counter += 1
                self.img_labels.append((image_name, self.label_to_idx[label]))
                
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path+".jpg").convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_label_mappings(self):
        return self.label_to_idx, self.idx_to_label

# encode the labels
text = {}
cat_dir = glob.glob("../../ClimateVisions_Dataset/categories/*.txt")
for i in cat_dir:
    cat = i[i.find("/tum")+5:-4]
    with open(i) as f:
        labels = f.readlines()
        text[cat] = open_clip.tokenize(labels)


# Load the images
img_dir = "../../Images/ClimateCT"


# Iterate through the data loader (example)
for i in text.keys():
    print(i)
    acc = 0.0
    total = 0
    correct = 0
    txt_file = f"../../ClimateVisions_Dataset/final_labels/tum_{i}.txt"


    dataset = CustomImageDataset(txt_file=txt_file, img_dir=img_dir, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Access the label to int mapping
    label_to_idx, idx_to_label = dataset.get_label_mappings()

    for images, labels in dataloader:
        # Training code here
        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(text[i])
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            class_preds = torch.argmax(text_probs, dim=1)
            total += labels.size(0)
            correct += (class_preds == labels).sum().item()
    #print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
    acc = correct / total
    print(acc)








# Iterating through the DataLoader
for images, labels in dataloader:
    # Your training code here
    pass
