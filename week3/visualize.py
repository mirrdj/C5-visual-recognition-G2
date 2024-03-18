from utils import *

import cv2
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import seaborn as sns
sns.set()

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)

DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
DATASET_TEST = '/ghome/mcv/datasets/C3/MIT_large_train'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def confusion(actual, predicted, classes):
    conf = np.zeros((len(classes), len(classes)), dtype=np.float64)

    # mp = {}
    # for i, cls in enumerate(classes):
    #     mp[cls] = i
    real_classes = {0:5, 1:7,2:2,3:1,4:0,5:4,6:3,7:6}

    for a, p in zip(actual, predicted):
        conf[real_classes[a], real_classes[p]] += 1

    print(conf)
    #conf = conf / conf.sum(axis=1, keepdims=True)
    ax = sns.heatmap(conf, cmap='YlGnBu', xticklabels=classes, yticklabels=classes, square=True, annot=True)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
    ax.set_title('Confusion Matrix')
    plt.savefig('visualizations/conf_map.png')
    return 


def visualizate_save(model, params, IMG_SIZE, target_layer):
    directory_test = DATASET_TEST+'/test'
    class_nodict = ['Opencountry','coast','forest','highway','inside_city','mountain','street','tallbuilding']
    classes = {'Opencountry':4, 'coast':3,'forest':2,'highway':6,'inside_city':5,'mountain':0,'street':7,'tallbuilding':1}
    # classes = {'Opencountry':0, 'coast':1,'forest':2,'highway':3,'inside_city':4,'mountain':5,'street':6,'tallbuilding':7}
    classes_inv = list(classes.keys())
    
    transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.43239, 0.45149, 0.44213],
                std=[0.255, 0.244, 0.270]
                )])

    test_labels = []
    preds = []

    for class_dir in os.listdir(directory_test):
        print(class_dir)
        cls = classes[class_dir]
        for imname in os.listdir(os.path.join(directory_test,class_dir)):
            img = Image.open(os.path.join(directory_test,class_dir,imname)).convert("RGB")

            im = transform(img)
            im = im.unsqueeze(0)
            pred = model(im.to(device))
            _, predicted = torch.max(pred, 1)
            predicted = predicted.cpu().numpy()[0]

            rgb_img = cv2.imread(os.path.join(directory_test,class_dir,imname), 1)[:, :, ::-1]
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img,
                                    mean=[0.43239, 0.45149, 0.44213],
                                    std=[0.255, 0.244, 0.270]).to(device)
            
            cam = GradCAM(model=model, target_layers=target_layer)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # icam = GradCAM(model, out, 'conv2d_1') 
            # heatmap = icam.compute_heatmap(im)
            # heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

            # image = cv2.imread(os.path.join(directory_test,class_dir,imname))
            # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            # (heatmap, output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)


            if predicted==cls:
                #cv2.imwrite(f'visualizations/correct/{class_dir}/heatmap_{imname}', heatmap) 
                if not os.path.exists(f'visualizations/correct/{class_dir}'):
                    # If it doesn't exist, create it
                    os.makedirs(f'visualizations/correct/{class_dir}')
                cv2.imwrite(f'visualizations/correct/{class_dir}/output_{imname}', heatmap) 

            else: 
                #cv2.imwrite(f'visualizations/incorrect/{class_dir}/heatmap_{classes_inv[out]}_{imname}', heatmap) 
                if not os.path.exists(f'visualizations/incorrect/{class_dir}'):
                    # If it doesn't exist, create it
                    os.makedirs(f'visualizations/incorrect/{class_dir}')

                cv2.imwrite(f'visualizations/incorrect/{class_dir}/output_{classes_inv[predicted]}_{imname}', heatmap) 
            
            test_labels.append(cls)
            preds.append(predicted)
    
    confusion(test_labels, preds, classes)

    return

def model(params):
    """
    CNN model configuration
    """
    
    # Define the test data generator for data augmentation and preprocessing
    IMG_WIDTH = int(params['img_size'])

    BEST_MODEL_FNAME = '/ghome/group02/C5-G2/Week1/weights'

    # create the base pre-trained model
    # base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(params)
    model.to(device)


    # optimizer = get_optimizer(params, model)
    # criterion = nn.CrossEntropyLoss()
    #Test
    model.load_state_dict(torch.load(f'{BEST_MODEL_FNAME}/best_model.pth'))
    model.eval()
    target_layer=model.conv_block2.conv_block
    print(target_layer)
    
    visualizate_save(model, params, IMG_WIDTH, target_layer)
    
    

params = {
    #'unfrozen_layers': trial.suggest_categorical('unfrozen_layers', ["1"]),  # 1,2,3,4,5
    
    'substract_mean': 'True',
    'batch_size': '16',  # 8,16,32,64
    'img_size': '224',  # 8,16,32,64,128,224,256
    'lr': 1,  # 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3
    'optimizer': 'adadelta',  # adadelta, adam, sgd, RMSprop


    'activation': 'relu',
    'n_filters_1': '64',
    'n_filters_2': '16',
    'n_filters_3': '16',
    'n_filters_4': '16',

    'kernel_size_1': '3',
    'kernel_size_2': '3',
    'kernel_size_3': '3',
    'kernel_size_4': '3',

    'stride': 1,

    'pool': 'max',

    'padding': 'same',
    'neurons': '256',

    'data_aug': 'sr',
    'momentum': 0.95,
    'dropout': '0',
    'bn': 'True',
    'L2': 'False',
    'epochs': 100,
    'depth': 2, 
    'pruning_thr': 0.1,
    'output': 8,
}

#Execute the 'main'
model(params)