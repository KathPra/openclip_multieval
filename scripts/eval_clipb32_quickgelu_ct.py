## conda env: openclip

# load packages
import torch
from PIL import Image
import open_clip
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import glob
from collections import defaultdict, Counter
import numpy as np

# load model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

# create a custom dataset class which takes a folder of images and a text file of img file names and corresponding labels
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
        img_path = os.path.join(self.img_dir, img_name + ".jpg")
        image = Image.open(img_path)
        image = image.convert("RGBA" if image.mode in ['P', 'RGBA'] and image.info.get('transparency') is not None else "RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_label_mappings(self):
        return self.label_to_idx, self.idx_to_label

# open the description of the super-classes
text = {} # contains queries
class_overview = {} #  maps query to class
cat_dir = glob.glob("../../ClimateVisions_Dataset/categories/*.txt")
for i in cat_dir:
    cat = i[i.find("/tum")+5:-4]
    with open(i) as f:
        class_desc = f.readlines()
        class_desc = [i.replace("\n","").split(":") for i in class_desc ]
        queries = [i[1] for i in class_desc]
        text[cat] = open_clip.tokenize(queries)

        # create mapping from query to classes, as some classes have multiple queries
        classes = [i[0] for i in class_desc]
        class_set = list( dict.fromkeys(classes) )
        query_int = np.arange(0, len(queries), 1, dtype=int).tolist()
        class_int = np.arange(0, len(class_set), 1, dtype=int).tolist()
        class_dict = dict(zip(class_set, class_int))
        class_int2 = [class_dict[i] for i in classes]
        class_overview[cat] = dict(zip(query_int, class_int2))
        

# Load the images
img_dir = "../../Images/ClimateCT"


# iterate through super-categories
for i in text.keys():
    print(i)
    acc = 0.0
    total = 0
    correct = 0
    
    # load the data corresponding to the current super-class
    txt_file = f"../../ClimateVisions_Dataset/final_labels/tum_{i}.txt"
    dataset = CustomImageDataset(txt_file=txt_file, img_dir=img_dir, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Access the label to int mapping
    label_to_idx, idx_to_label = dataset.get_label_mappings()
    acc_cw = {c:0 for c in label_to_idx.keys()}

    # prepare to map the queries to classes
    query_class_mapping = class_overview[i]
    
    for images, labels in dataloader:
        label_stats = Counter(np.array(labels))
        label_stats = {idx_to_label[i]:v for i,v in label_stats.items()}
        # Training code here
        with torch.no_grad():
            # embed image and text
            image_features = model.encode_image(images)
            text_features = model.encode_text(text[i])
            # normalize embeddings
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # map images to queries and find closest query
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            query_preds = torch.argmax(text_probs, dim=1).tolist()
            # map queries to classes
            class_preds = [query_class_mapping[i] for i in query_preds]
            class_preds = torch.tensor(class_preds)
            # map queries to classes (some classes contain multiple queries)
            pred_stats = Counter(np.array(class_preds))
            pred_stats = Counter(np.array(class_preds))
            pred_stats = {idx_to_label[i]:v for i,v in pred_stats.items()}
            total += labels.size(0)
            correct += (class_preds == labels).sum().item()

            #for c in pred_stats.keys():
            #    acc_cw[c] += ((class_preds == labels) * (labels == class_int)).float().sum() / (max(labels == class_int).sum(), 1)
            
            
    #print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
    acc = correct / total
    print(acc)
    #print(acc_cw)
    


