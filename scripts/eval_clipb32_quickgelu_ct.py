## conda env: openclip

# load packages
import torch
from PIL import Image
import open_clip
import os
#from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import glob
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
from torch.utils.tensorboard import SummaryWriter

# load model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

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
            queries = [desc[1] for desc in class_desc]
            text[category] = tokenizer(queries)
            
            classes = [desc[0] for desc in class_desc]
            class_set = list(dict.fromkeys(classes))
            class_dict = {cls: idx for idx, cls in enumerate(class_set)}
            query_class_mapping = [class_dict[cls] for cls in classes]
            class_overview[category] = dict(zip(range(len(queries)), query_class_mapping))
    return text, class_overview

# Evaluate a super-class
def evaluate_category(category, text, class_overview, img_dir, preprocess, top_k, writer = None):
    # initialize tensorboard
    if writer is None:
        writer = SummaryWriter()
    # load annotations
    txt_file = f"../../ClimateVisions_Dataset/final_labels/tum_{category}.txt"
    dataset = CustomImageDataset(txt_file=txt_file, img_dir=img_dir, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    label_to_idx, idx_to_label = dataset.get_label_mappings()
    query_class_mapping = class_overview[category]

    # required for class_wise accuracy
    total_per_class = Counter()
    correct_per_class = Counter()

    # required for recall, precision and F1
    true_labels = []
    predicted_labels = []
    predicted_labels_top_k = []
    
    # required for total accuracy
    correct, total = 0, 0

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
            
            # overall accuracy
            total += labels.size(0)
            correct += (class_preds == labels).sum().item()

            # other metrics
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(class_preds.cpu().numpy())
            predicted_labels_top_k.extend(class_preds_top_k.cpu().numpy())

            # class-wise accuracy
            for label, pred in zip(labels, class_preds):
                total_per_class[idx_to_label[label.item()]] += 1
                if label == pred:
                    correct_per_class[idx_to_label[label.item()]] += 1
    
    accuracy = correct / total
    print(f"Accuracy for {category}: {accuracy:.4f}")
    accuracy = accuracy_score(true_labels, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)

    class_wise_accuracy = {class_name: correct_per_class[class_name] / total_per_class[class_name] 
                           for class_name in total_per_class}
    
    precision = precision_score(true_labels, predicted_labels, average=None)
    recall = recall_score(true_labels, predicted_labels, average=None)
    f1 = f1_score(true_labels, predicted_labels, average=None)
    acc = accuracy_score(true_labels, predicted_labels, average=None)
    #top_k_accuracy = sum(1 for true, pred in zip(true_labels, predicted_labels_top_k) if true in pred) / len(true_labels)
    
    precision_macro = precision_score(true_labels, predicted_labels, average='macro')
    recall_macro = recall_score(true_labels, predicted_labels, average='macro')
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    
    
    class_metrics = {
        idx_to_label[i]: {'Precision': precision[i], 'Recall': recall[i], 'F1 Score': f1[i]}
        for i in range(len(precision))
    }

    # log results to tensorboard
    overall_metrics = {
        'Accuracy': accuracy,
        'Balanced Accuracy': balanced_accuracy,
        'Precision (Macro)': precision_macro,
        'Recall (Macro)': recall_macro,
        'F1 Score (Macro)': f1_macro#,
        #'Top-k Accuracy': top_k_accuracy
    }

    if writer is not None:
        for class_name, metrics in class_metrics.items():
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(f'{category}/{class_name}/{metric_name}', metric_value)
    
    return class_metrics, overall_metrics
      

# Main execution
cat_dir = glob.glob("../../ClimateVisions_Dataset/categories/*.txt")
text, class_overview = process_text_descriptions(cat_dir)
img_dir = "../../Images/ClimateCT"

overall_top_k_accuracy = []

# Initialize SummaryWriter for Tensorboard
writer = SummaryWriter()

for category in text.keys():
    class_metrics, overall_metrics  = evaluate_category(category, text, class_overview, img_dir, preprocess, top_k=3)
    #overall_top_k_accuracy.append(overall_metrics['Top-k Accuracy'])
    
    print(f"Metrics for {category}:")
    for class_name, metrics in class_metrics.items():
        print(f"  {class_name}: Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, F1 Score={metrics['F1 Score']:.4f}")
    print(f"  Overall Precision (Macro): {overall_metrics['Precision (Macro)']:.4f}")
    print(f"  Overall Recall (Macro): {overall_metrics['Recall (Macro)']:.4f}")
    print(f"  Overall F1 Score (Macro): {overall_metrics['F1 Score (Macro)']:.4f}")
    #print(f"  Top-3 Accuracy: {overall_metrics['Top-k Accuracy']:.4f}")
    print()

#average_top_k_accuracy = sum(overall_top_k_accuracy) / len(overall_top_k_accuracy)
#print("Average Top-k Accuracy across all categories:", average_top_k_accuracy)

# Close SummaryWriter
writer.close()