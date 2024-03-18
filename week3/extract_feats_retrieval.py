import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from utils import *
from tqdm import tqdm
from PIL import Image
import faiss
from sklearn.metrics import precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

DATASET_DIR = '/ghome/group02/mcv/datasets/C3/MIT_split'
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class_to_int = {'Opencountry': 0,
                'coast': 1,
                'forest': 2,
                'highway': 3,
                'inside_city': 4,
                'mountain': 5,
                'street': 6,
                'tallbuilding': 7}

int_to_class = {0: 'Opencountry',
                1: 'coast',
                2:'forest',
                3: 'highway',
                4: 'inside_city',
                5: 'mountain',
                6: 'street',
                7: 'tallbuilding'}


model = models.densenet121(pretrained=True)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 8) #Add layer with 8 classes
model.load_state_dict(torch.load('/ghome/group02/C5-G2/Week3/weights/best_model_backup.pth'))

model = torch.nn.Sequential(*list(model.children())[:-1]) #Remove last layer
# Add global average pooling layer
global_avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))

# Combine DenseNet features and global average pooling
feature_extractor = torch.nn.Sequential(model, global_avg_pooling)
feature_extractor.to(device)
feature_extractor.eval()  # Set model to evaluation mode

transform = transforms.Compose([
                    transforms.Resize((224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )])

train_imgs_paths = get_imgs_paths(DATASET_DIR, 'train')
test_imgs_paths = get_imgs_paths(DATASET_DIR, 'test')

for split in ['train', 'test']:

    imgs_paths = get_imgs_paths(split)
    extracted_feats = []

    for idx, img_path in tqdm(enumerate(imgs_paths)):
            # Preprocess the batch of images
            # preprocessed_images = torch.stack([transform(image) for image in images])
            label = img_path.split('/')[-2]
            int_label = class_to_int[label]

            # print(label, int_label)

            img = Image.open(img_path).convert("RGB")
            img = transform(img)

            # Extract features using the model
            with torch.no_grad():
                features = feature_extractor(img.to(device).unsqueeze(0))
                features = features.squeeze().cpu().detach().numpy()
                    
            # Store the extracted features and their corresponding labels (if available)
            extracted_feats.append([features, label, int_label])
    
    if split == 'train':
        feats_train = extracted_feats
    else:
        
        feats_test = extracted_feats

        #Create a dictionary with binary labels for the GT

        gt = {}
        for category in class_to_int:
            gt[category] = [1 if label == category else 0 for label in [sublist[1] for sublist in feats_train]]
            

index = faiss.IndexFlatL2(1024)
for feat in feats_train:
    # print(feat)
    index.add(np.array([feat[0]]))

# Initialize and fit KNN model
k = 5  # Number of nearest neighbors to retrieve
knn_model = NearestNeighbors(n_neighbors=k, algorithm='auto')
knn_model.fit([sublist[0] for sublist in feats_train])

# Aggregate Precision and Recall values across all queries

total_ap = {}
total_precision = {}
total_recall = {}



for feat in feats_test:
    distances, indices = index.search(np.array([feat[0]]), k=len(feats_train))
    # distances_knn, indices_knn = knn_model.kneighbors(np.array(feat[0]).reshape(1, -1))
    extracted_elements = [feats_train[i][1] for i in indices[0]]
    # extracted_elements_knn = [feats_train[i][1] for i in indices_knn[0]]

    if feat[1] not in total_ap:
        total_ap[feat[1]] = []
        total_precision[feat[1]] = []
        total_recall[feat[1]] = []

    ap, precision, recall = compute_ap(extracted_elements, feat[1], np.sum(np.array(gt[feat[1]]) == 1))
    print(f"Manual AP: {ap}")
    skap = average_precision_score([1 if feats_train[i][1] == feat[1] else 0 for i in indices[0]], (distances.max() - distances)/distances.max())
    print(f"sklearn AP: {skap}")
    print()


    total_ap[feat[1]].append(ap)
    total_precision[feat[1]].append(precision)
    total_recall[feat[1]].append(recall)

    # print(f'Query: {feat[1]} | Retrieved 1: {extracted_elements[0]}')

plt.figure(figsize=(8, 6))
map_total = 0
p_1 = 0
p_5 = 0
for clase in total_ap:

    mean_ap_class = np.mean(total_ap[clase], axis=0)
    mean_pr_class = np.mean(total_precision[clase], axis=0)
    mean_rc_class = np.mean(total_recall[clase], axis=0)

    print(f'Class {clase}: mAP->{mean_ap_class} | P@1->{mean_pr_class[0]} | P@5->{mean_pr_class[4]}')
    map_total += mean_ap_class
    p_1 += mean_pr_class[0]
    p_5 += mean_pr_class[4]
    
    plt.plot(np.array(mean_rc_class), np.array(mean_pr_class), label=clase)
    
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig(f'/ghome/group02/C5-G2/Week3/Precision-Recall_Curve/Precision-Recall_general.png')

map_total = map_total / 8
p_1 = p_1 / 8
p_5 = p_5 / 8
print(f'Total: mAP->{map_total} | P@1->{p_1} | P@5->{p_5}')

# Perform PCA for dimensionality reduction
# pca = PCA(n_components=50)  # Reduce to 50 components
# X_pca = pca.fit_transform([sublist[0] for sublist in feats_train])

# Perform t-SNE embedding
tsne = TSNE(n_components=2, random_state=0)
X_embedded = tsne.fit_transform([sublist[0] for sublist in feats_train])
class_labels = np.array([sublist[2] for sublist in feats_train])

# Visualize the embedded data
plt.figure(figsize=(8, 6))
for i in range(8):
    class_indices = np.where(class_labels == i)[0]  # Get indices of points belonging to class i
    plt.scatter(X_embedded[class_indices, 0], X_embedded[class_indices, 1], label=int_to_class[i], s=10)
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()

# Save the plot as a PNG file
plt.savefig('tsne_plot.png')

