## conda env: clip

# load packages
import torch
from PIL import Image
import os
#from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import glob
from collections import Counter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import numpy as np
import clip
import matplotlib.pyplot as plt

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
#tokenizer = open_clip.get_tokenizer('ViT-B-32')

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

# prepare queries, classes, and the query-to-class mapping
def process_text_descriptions(cat_dir):
    text = {}
    class_overview = {}
    for file_path in cat_dir:
        category = os.path.basename(file_path).replace('.txt', '').replace("tum_","")
        with open(file_path) as f:
            class_desc = [line.strip().split(":") for line in f]
        queries = [desc[1][1:] for desc in class_desc] # remove white space before query
        text[category] = clip.tokenize(queries)
        # create a mapping from query to class
        class_overview[category] ={desc[1][1:]: desc[0] for desc in class_desc}
    return text, class_overview

# Evaluate a super-class
def evaluate_category(category, text, class_overview, img_dir, preprocess, top_k):
    # load annotations
    txt_file = f"../../ClimateVisions_Dataset/final_labels/tum_{category}.txt"
    dataset = CustomImageDataset(txt_file=txt_file, img_dir=img_dir, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    label_to_idx, idx_to_label = dataset.get_label_mappings()
    query_class_overview = class_overview[category]

    class_list = list(dict.fromkeys(idx_to_label))
    # create mapping two combine results for classes with multiple queries
    query_class_map = dict(zip(np.arange(0,len(query_class_overview),1),query_class_overview.values()))
    query_class_mapping = {k:label_to_idx[v] for (k,v) in query_class_map.items()}

    # required for class_wise accuracy
    total_per_class = Counter()
    correct_per_class = Counter()

    # required for recall, precision and F1
    true_labels = []
    predicted_labels = []
    predicted_labels_top_k = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            # load embeddings
            images, labels = images.to(device), labels.to(device)
            image_features = model.encode_image(images)
            text_features = model.encode_text(text[category].to(device))
            # normalize embeddings
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            _, top_k_preds = torch.topk(text_probs, top_k, dim=1)
            class_preds_top_k = torch.tensor([[query_class_mapping[q.item()] for q in top_k_pred] for top_k_pred in top_k_preds]).to(device)
            
            query_preds = torch.argmax(text_probs, dim=1).tolist()
            class_preds = torch.tensor([query_class_mapping[q] for q in query_preds]).to(device)
            
            # other metrics
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(class_preds.cpu().numpy())
            predicted_labels_top_k.extend(class_preds_top_k.cpu().numpy())

            # class-wise accuracy
            for label, pred in zip(labels, class_preds):
                total_per_class[idx_to_label[label.item()]] += 1
                if label == pred:
                    correct_per_class[idx_to_label[label.item()]] += 1
    
    
    overall_metrics = compute_overall_metrics(true_labels, predicted_labels, category)

    # class-wise metrics
    class_wise_accuracy = {class_name: correct_per_class[class_name] / total_per_class[class_name] 
                           for class_name in total_per_class}
    print(class_wise_accuracy)
    class_metrics = compute_classwise_metrics(true_labels, predicted_labels, idx_to_label, category)

    
    return class_metrics, overall_metrics

def compute_overall_metrics(true_labels, pred_labels, cat):
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"Accuracy for {cat}: {accuracy:.4f}")

    balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro")

    overall_metrics = {
        'Balanced Accuracy': balanced_accuracy,
        'Precision (Macro)': precision_macro,
        'Recall (Macro)': recall_macro,
        'F1 Score (Macro)': f1_macro#,
        #'Top-k Accuracy': top_k_accuracy
        }

    return overall_metrics

def compute_classwise_metrics(true_labels, pred_labels, mapping, topic):
    prec, recall, f1, support = precision_recall_fscore_support(true_labels, pred_labels, average=None)
    print(mapping)

    # compuate and save confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=list(mapping.keys()), normalize="true")
    class_wise_acc = np.diag(conf_matrix)
    save_convmat(conf_matrix, topic, list(mapping.values()))

    #top_k_accuracy = sum(1 for true, pred in zip(true_labels, predicted_labels_top_k) if true in pred) / len(true_labels)
        
    class_metrics = {
        mapping[i]: {'Accuracy':class_wise_acc[i],'Precision': prec[i], 'Recall': recall[i], 'F1 Score': f1[i], 'Support':support[i]}
        for i in range(len(prec))}
    
    return class_metrics

def save_convmat(conv_mat, topic, label_overview):
        plt.rcParams.update({'font.size': 20})
        disp = ConfusionMatrixDisplay(confusion_matrix=conv_mat, display_labels=label_overview)
        _, ax = plt.subplots(figsize=(10,10))
        
        disp.plot(ax=ax, xticks_rotation="vertical", colorbar=False, cmap="Greens",values_format=".1f")

        for i in disp.text_.flatten(): i._text = i._text.replace("0.", ".")

        #plt.title("Normalized confusion matrix")
        plt.savefig(f"conf_mat/openai_clip_vitb32/conv_mat_{topic}_norm.png", bbox_inches="tight")
        plt.close()

# Main execution
cat_dir = glob.glob("../../ClimateVisions_Dataset/categories/*.txt")
text, class_overview = process_text_descriptions(cat_dir)
img_dir = "../../Images/ClimateCT"

for category in text.keys():
    class_metrics, overall_metrics  = evaluate_category(category, text, class_overview, img_dir, preprocess, top_k=3)
    #overall_top_k_accuracy.append(overall_metrics['Top-k Accuracy'])
    
    print(f"Class-wise Metrics for {category}:")
    for class_name, metrics in class_metrics.items():
        print(f"  {class_name}: Accuracy={metrics['Accuracy']:.4f}, Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, F1 Score={metrics['F1 Score']:.4f}, Support={metrics['Support']}")
    print()
    print(f"  Overall balanced accuracy (Macro): {overall_metrics['Balanced Accuracy']:.4f}")
    print(f"  Overall Precision (Macro): {overall_metrics['Precision (Macro)']:.4f}")
    print(f"  Overall Recall (Macro): {overall_metrics['Recall (Macro)']:.4f}")
    print(f"  Overall F1 Score (Macro): {overall_metrics['F1 Score (Macro)']:.4f}")
    #print(f"  Top-3 Accuracy: {overall_metrics['Top-k Accuracy']:.4f}")
    print()
    break

#average_top_k_accuracy = sum(overall_top_k_accuracy) / len(overall_top_k_accuracy)
#print("Average Top-k Accuracy across all categories:", average_top_k_accuracy)
