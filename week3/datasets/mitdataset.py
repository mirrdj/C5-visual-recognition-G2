from torch.utils.data import Dataset
import os 
from PIL import Image
import torch
import numpy as np
from utils import get_imgs_lbls_dict


class MiTDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform

        self.images = []
        self.labels = []
        for i, cls in enumerate(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, cls)
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
    
class SiameseMITDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

        self.images = []
        self.labels = []

        for i, cls in enumerate(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, cls)
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(i)
        
        self.labels = np.array(self.labels)
        self.labels_set = set(self.labels)
            
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                            for label in self.labels_set}
        
        # Create fixed pairs for testing 
        if mode == 'test':
            random_state = np.random.RandomState(29)

            # Create fixed positive pairs (pairs of images from the same class)
            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.labels[i].item()]),
                               1]
                              for i in range(0, len(self.images), 2)]
            

            # or negative pairs (pairs of images from the same class) for testing
            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.images), 2)]
            self.test_pairs = positive_pairs + negative_pairs


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.mode == 'train':
            target = np.random.randint(0, 2)
            img1, label1 = self.images[idx], self.labels[idx].item()

            # Find a pair for img1 with the same label1 - create positive pair
            if target == 1:
                siamese_index = np.random.choice(
                    [i for i in self.label_to_indices[label1] if i != idx]
                )
            else: # Find a pair for img1 with label different from label1 - create negative pair
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])

            img2 = self.images[siamese_index]

        else: # select from the predefined pairs in testing
            img1 = self.images[self.test_pairs[idx][0]]
            img2 = self.images[self.test_pairs[idx][1]]
            target = self.test_pairs[idx][2]

        img1 = Image.open(img1).convert("RGB")
        img2 = Image.open(img2).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        

        return img1, img2, target


class TripletMITDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

        self.images = []
        self.labels = []

        for i, cls in enumerate(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, cls)
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(i)
        
        self.labels = np.array(self.labels)
        self.labels_set = set(self.labels)
            
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                            for label in self.labels_set}
        
        # Create fixed triplets for testing 
        if mode == 'test':
            random_state = np.random.RandomState(29)

            triplets = [[i,
                random_state.choice(self.label_to_indices[self.labels[i].item()]),
                random_state.choice(self.label_to_indices[
                                        np.random.choice(
                                            list(self.labels_set - set([self.labels[i].item()]))
                                        )
                                    ])
                ]
            for i in range(len(self.images))]
        
            self.triplests = triplets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.mode == 'train':
            img1, label1 = self.images[idx], self.labels[idx].item()
            positive_idx = np.random.choice(
                    [i for i in self.label_to_indices[label1] if i != idx]
                )
            
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_idx = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.images[positive_idx]
            img3 = self.images[negative_idx]

        else: # Retrieve from the predefined triplets
            img1 = self.images[self.triplests[idx][0]]
            img2 = self.images[self.triplests[idx][1]]
            img3 = self.images[self.triplests[idx][2]]

        img1 = Image.open(img1).convert("RGB")
        img2 = Image.open(img2).convert("RGB")
        img3 = Image.open(img3).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        # Return anchor, pos, neg
        return img1, img2, img3

class TripletCOCO(Dataset):
    def __init__(self, data_dir, annotations, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

        self.images = []
        self.labels = []

        # imgs_labels = get_imgs_lbls_dict(annotations)

        if mode == 'train':
            for object, img_ids in annotations.items():
                for img_id in img_ids:
                    self.images.append(os.path.join(data_dir, "COCO_train2014_" + str(img_id).zfill(12) + ".jpg"))
                    self.labels.append(int(object))
        if mode == 'test':
            for object, img_ids in annotations.items():
                for img_id in img_ids:
                    self.images.append(os.path.join(data_dir, "COCO_val2014_" + str(img_id).zfill(12) + ".jpg"))
                    self.labels.append(int(object))

        # for i, cls in enumerate(os.listdir(data_dir)):
        #     class_dir = os.path.join(data_dir, cls)
        #     for img_name in os.listdir(class_dir):
        #         self.images.append(os.path.join(class_dir, img_name))
        #         self.labels.append(i)
        
        self.labels = np.array(self.labels)
        self.labels_set = set(self.labels)
            
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                            for label in self.labels_set}
        
        # Create fixed triplets for testing 
        if mode == 'test':
            random_state = np.random.RandomState(29)

            triplets = [[i,
                random_state.choice(self.label_to_indices[self.labels[i].item()]),
                random_state.choice(self.label_to_indices[
                                        np.random.choice(
                                            list(self.labels_set - set([self.labels[i].item()]))
                                        )
                                    ])
                ]
            for i in range(len(self.images))]
        
            self.triplests = triplets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.mode == 'train':
            img1, label1 = self.images[idx], self.labels[idx].item()
            positive_idx = np.random.choice(
                    [i for i in self.label_to_indices[label1] if i != idx]
                )
            
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_idx = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.images[positive_idx]
            img3 = self.images[negative_idx]

        else: # Retrieve from the predefined triplets
            img1 = self.images[self.triplests[idx][0]]
            img2 = self.images[self.triplests[idx][1]]
            img3 = self.images[self.triplests[idx][2]]

        img1 = Image.open(img1).convert("RGB")
        img2 = Image.open(img2).convert("RGB")
        img3 = Image.open(img3).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        # Return anchor, pos, neg
        return img1, img2, img3




