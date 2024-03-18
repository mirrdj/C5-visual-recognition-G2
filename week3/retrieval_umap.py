import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from utils import *
from tqdm import tqdm
from PIL import Image
import faiss
import yaml
import umap

from sklearn.metrics import precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def retrieval(embedding_dimension, name, weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('config.yml', 'r') as file:
        data = yaml.safe_load(file)

    dataset_dir = data['DATASET_DIR']
    dataset_train = data['DATASET_TRAIN']
    dataset_test = data['DATASET_TEST']

    class_to_int = data['class_to_int']
    int_to_class = data['int_to_class']

    # embedding_dimension = 1024

    model = models.densenet121(pretrained=True)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, embedding_dimension)  # Add layer with 8 classes
    # TODO: Change to name of model
    model.load_state_dict(torch.load(weights))

    # model = torch.nn.Sequential(*list(model.children())[:-1]) #Remove last layer
    # # Add global average pooling layer
    # global_avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))

    # # Combine DenseNet features and global average pooling
    # feature_extractor = torch.nn.Sequential(model, global_avg_pooling)
    feature_extractor = model
    feature_extractor.to(device)
    feature_extractor.eval()  # Set model to evaluation mode

    transform = transforms.Compose([
        transforms.Resize((224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )])

    train_imgs_paths = get_imgs_paths(dataset_dir, 'train')
    test_imgs_paths = get_imgs_paths(dataset_dir, 'test')

    for split in ['train', 'test']:

        imgs_paths = get_imgs_paths(dataset_dir, split)
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
            extracted_feats.append([features, label, int_label, img_path])

        if split == 'train':
            feats_train = extracted_feats
        else:

            feats_test = extracted_feats

            # Create a dictionary with binary labels for the GT

            gt = {}
            for category in class_to_int:
                gt[category] = [1 if label == category else 0 for label in [sublist[1] for sublist in feats_train]]

    index = faiss.IndexFlatL2(embedding_dimension)
    feats_embedd = np.array([sublist[0] for sublist in feats_train])
    feats_query = np.array([sublist[0] for sublist in feats_test])
    faiss.normalize_L2(feats_embedd)
    faiss.normalize_L2(feats_query)
    index.add(feats_embedd)

    # Initialize and fit KNN model
    k = len(feats_train)  # Number of nearest neighbors to retrieve
    knn_model = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_model.fit([sublist[0] for sublist in feats_train])

    # Aggregate Precision and Recall values across all queries

    total_ap = {}
    total_precision = {}
    total_recall = {}

    for feat, query in zip(feats_test, feats_query):
        distances, indices = index.search(np.array([query]), k=len(feats_train))
        # distances, indices = index.search(np.array([feat[0]]), k=len(feats_train))
        # distances_knn, indices_knn = knn_model.kneighbors(np.array(feat[0]).reshape(1, -1))
        extracted_elements = [feats_train[i][1] for i in indices[0]]
        ordered_paths = [feats_train[i][3] for i in indices[0]]
        # extracted_elements_knn = [feats_train[i][1] for i in indices_knn[0]]

        if feat[1] not in total_ap:
            total_ap[feat[1]] = []
            total_precision[feat[1]] = []
            total_recall[feat[1]] = []

        ap, precision, recall = compute_ap(extracted_elements, feat[1], np.sum(np.array(gt[feat[1]]) == 1))
        
        print(f'P@1 of class {feat[1]}: {precision[0]}')

        if int(precision[0]) == 0:
            print('Outlier!!!!!!!!')
            print(f'The query path is: ')


        total_ap[feat[1]].append(ap)
        total_precision[feat[1]].append(precision)
        total_recall[feat[1]].append(recall)

        # print(f'Query: {feat[1]} | Retrieved 1: {extracted_elements[0]}')

    plt.figure(figsize=(8, 6))
    map_total = 0
    p_1 = 0
    p_5 = 0
    all_classes = []

    # map_total = compute_overall_mean(total_ap)
    for clase in total_ap:
        all_classes.extend(total_ap[clase])
    mean_ap_all = np.mean(all_classes, axis=0)

    for clase in total_ap:
        mean_ap_class = np.mean(total_ap[clase], axis=0)
        mean_pr_class = np.mean(total_precision[clase], axis=0)
        mean_rc_class = np.mean(total_recall[clase], axis=0)

        print(f'Class {clase}: mAP->{mean_ap_class} | P@1->{mean_pr_class[0]} | P@5->{mean_pr_class[4]}')
        # map_total += mean_ap_class
        # map_total = mean_ap_all
        p_1 += mean_pr_class[0]
        p_5 += mean_pr_class[4]
        # print(mean_pr_class)
        # print('RECALL: ', mean_rc_class)

        plt.plot(np.array(mean_rc_class), np.array(mean_pr_class), label=clase)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(f'/ghome/group02/C5-G2/Week3/Precision-Recall_Curve/Precision-Recall_{name}.png')

    # map_total = map_total / 8
    map_total = mean_ap_all
    p_1 = p_1 / 8
    p_5 = p_5 / 8
    print(f'Total: mAP->{map_total} | P@1->{p_1} | P@5->{p_5}')

    # Perform PCA for dimensionality reduction
    # pca = PCA(n_components=50)  # Reduce to 50 components
    # X_pca = pca.fit_transform([sublist[0] for sublist in feats_train])

    for split in ['train', 'test']:
        if split == 'train':
            feats = feats_train
            embed = feats_embedd
        else:
            feats = feats_test
            embed = feats_query

        tsne_visualization(feats, embed, name, split, int_to_class)
        umap_visualization(feats, embed, name, split, int_to_class)
        pca_visualization(feats, embed, name, split, int_to_class)

    return map_total, p_1, p_5


def tsne_visualization(feats, embed, name, split, int_to_class):
    # Perform t-SNE embedding
    tsne = TSNE(n_components=2, random_state=0)
    X_embedded = tsne.fit_transform(embed)
    class_labels = np.array([sublist[2] for sublist in feats])

    # Visualize the embedded data
    plt.figure(figsize=(8, 6))
    for i in range(8):
        class_indices = np.where(class_labels == i)[0]  # Get indices of points belonging to class i
        plt.scatter(X_embedded[class_indices, 0], X_embedded[class_indices, 1], label=int_to_class[i], s=10)
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.title(f't-SNE Visualization {split}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(f'./TSNE_plot/tsne_plot_{split}_{name}.png')


def umap_visualization(feats,embed, name, split, int_to_class):
    standard_embedding = umap.UMAP(n_components=2, random_state=0).fit_transform(
        embed)
    class_labels = np.array([sublist[2] for sublist in feats])

    # Visualize the embedded data
    plt.figure(figsize=(8, 6))
    for i in range(8):
        class_indices = np.where(class_labels == i)[0]  # Get indices of points belonging to class i
        plt.scatter(standard_embedding[class_indices, 0], standard_embedding[class_indices, 1], label=int_to_class[i],
                    s=10)
    plt.title(f'UMAP Visualization {split}')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend()
    plt.savefig(f'./UMAP_plot/UMAP_plot_{split}_{name}.png')


def pca_visualization(feats, embed, name, split, int_to_class):
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(feats)

    pca = PCA(n_components=2, random_state=0)
    standard_embedding = pca.fit_transform(embed)
    class_labels = np.array([sublist[2] for sublist in feats])

    # Visualize the embedded data
    plt.figure(figsize=(8, 6))
    for i in range(8):
        class_indices = np.where(class_labels == i)[0]  # Get indices of points belonging to class i
        plt.scatter(standard_embedding[class_indices, 0], standard_embedding[class_indices, 1], label=int_to_class[i],
                    s=10)
    plt.title(f'PCA Visualization {split}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.savefig(f'./PCA_plot/PCA_plot_{split}_{name}.png')


retrieval(8, '33044_1','/export/home/group02/C5-G2/Week3/weights/best_siamese_IKER_miner_EASY_33044_1.pth')
    
# retrieval(8, 'L1', '/ghome/group02/C5-G2/Week3/weights/best_siamese_IKER_miner_EASY_33049_1.pth')

# def retrieval_COCO(embedding_dimension, model, annotations, mode):
    
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # with open('config.yml', 'r') as file:
#     #     data = yaml.safe_load(file)

#     dataset_train = '/ghome/group02/mcv/datasets/C5/COCO/'
#     dataset_test = '/ghome/group02/mcv/datasets/C5/COCO/train2014'

#     # class_to_int = data['class_to_int']
#     # int_to_class = data['int_to_class']

#     # # embedding_dimension = 1024

#     # model = models.densenet121(pretrained=True)
#     # num_features = model.classifier.in_features
#     # model.classifier = nn.Linear(num_features, embedding_dimension)  # Add layer with 8 classes
#     # # TODO: Change to name of model
#     # model.load_state_dict(torch.load(weights))

#     # model = torch.nn.Sequential(*list(model.children())[:-1]) #Remove last layer
#     # # Add global average pooling layer
#     # global_avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))

#     # # Combine DenseNet features and global average pooling
#     # feature_extractor = torch.nn.Sequential(model, global_avg_pooling)
#     # feature_extractor = model
#     # feature_extractor.to(device)
#     feature_extractor.eval()  # Set model to evaluation mode

    

#     transform = transforms.Compose([
#         transforms.Resize((224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#         )])
    

#     img_to_index = {}
#     for i, id in enumerate(imgs_lbls_query.keys()):
#         filename = f'COCO_{dir}_{id:012}.jpg'
#         img_to_index[filename] = i

#     for split in ['train2014', 'val2014']:
        
#         extracted_feats = []

#         if split == 'train2014':
#             imgs_lbls = get_imgs_lbls_dict(annotations['database'])
#         else:
#             imgs_lbls = get_imgs_lbls_dict(annotations[mode])

#         for img_id, objects in tqdm(imgs_lbls.items()):
#             img = Image.open(os.path.join('/ghome/group02/mcv/datasets/C5/COCO/', split, "COCO_train2014_" + str(img_id).zfill(12) + ".jpg")).convert("RGB")
#             img = transform(img)

#             # Extract features using the model
#             with torch.no_grad():
#                 features = feature_extractor(img.to(device).unsqueeze(0))
#                 features = features.squeeze().cpu().detach().numpy()

#             # Store the extracted features and their corresponding labels (if available)
#             extracted_feats.append([features, objects])

#             if split == 'train2014':
#                 feats_train = extracted_feats
#             else:
#                 feats_test = extracted_feats

#     index = faiss.IndexFlatL2(embedding_dimension)
#     feats_embedd = np.array([sublist[0] for sublist in feats_train])
#     feats_query = np.array([sublist[0] for sublist in feats_test])
#     faiss.normalize_L2(feats_embedd)
#     faiss.normalize_L2(feats_query)
#     index.add(feats_embedd)

#     # Initialize and fit KNN model
#     # k = len(feats_train)  # Number of nearest neighbors to retrieve
#     # knn_model = NearestNeighbors(n_neighbors=k, algorithm='auto')
#     # knn_model.fit([sublist[0] for sublist in feats_train])

#     # Aggregate Precision and Recall values across all queries

#     total_ap = []
#     total_precision = []
#     # total_recall = {}

#     for feat, query in zip(feats_test, feats_query):
#         distances, indices = index.search(np.array([query]), k=len(feats_train))
#         # distances, indices = index.search(np.array([feat[0]]), k=len(feats_train))
#         # distances_knn, indices_knn = knn_model.kneighbors(np.array(feat[0]).reshape(1, -1))
#         extracted_elements = [feats_train[i][1] for i in indices[0]]
#         # extracted_elements_knn = [feats_train[i][1] for i in indices_knn[0]]

#         # if feat[1] not in total_ap:
#         #     total_ap[feat[1]] = []
#         #     total_precision[feat[1]] = []
#         #     total_recall[feat[1]] = []

#         ap, precision = compute_ap_COCO(extracted_elements, feat[1])

#         total_ap.append(ap)
#         total_precision.append(precision)
#         # total_recall[feat[1]].append(recall)

#         # print(f'Query: {feat[1]} | Retrieved 1: {extracted_elements[0]}')

#     plt.figure(figsize=(8, 6))
#     map_total = 0
#     p_1 = 0
#     p_5 = 0


#     map_total = np.mean(total_ap, axis=0)
#     precision_total = np.mean(total_precision, axis=0)

#     p_1 = precision_total[0]
#     p_5 = precision_total[4]
#     print(f'Total: mAP->{map_total} | P@1->{p_1} | P@5->{p_5}')

#     # Perform PCA for dimensionality reduction
#     # pca = PCA(n_components=50)  # Reduce to 50 components
#     # X_pca = pca.fit_transform([sublist[0] for sublist in feats_train])

#     for split in ['train', 'test']:
#         if split == 'train':
#             feats = feats_train
#             embed = feats_embedd
#         else:
#             feats = feats_test
#             embed = feats_query

#         tsne_visualization(feats, embed, name, split, int_to_class)
#         umap_visualization(feats, embed, name, split, int_to_class)
#         pca_visualization(feats, embed, name, split, int_to_class)

#     return map_total, p_1, p_5