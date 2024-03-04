import numpy as np
from pycocotools import mask as coco_mask
import cv2
import os
from utils import *
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import random
import wandb

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import DatasetEvaluators
from detectron2.config import get_cfg
from sklearn.model_selection import train_test_split
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import DatasetEvaluator
import time as t

wandb.login(key='4127c8a2b851657f629b6f8f83ddc2e3415493f2')  # IKER
path = '/ghome/group02/mcv/datasets/C5/KITTI-MOTS/'
pathOutput = '/ghome/group02/C5-G2/Week2/outputs/'
# model = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x"
model = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x"
threshold= 0.5
task = model.split('-')[1].split('/')[0]
method = model.split('/')[1]
training_ids = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
testing_ids  = [2, 6, 7, 8, 10, 13, 14, 16, 18]

# sweep_config = {
#     'method': 'random',
#     'metric': {'goal': 'minimize', 'name': 'total_loss'},
#     'parameters': {
#         'lr' : {
#             'distribution': 'uniform',
#             'max': 0.01,
#             'min': 0.00001
#         },

#         'batch_size': {
#             'values': [2, 4, 8]
#         },

#         'optimizer': {
#             'values': ["SGD", "Adadelta", "AdamW"]
#         },
#     }
#  }

sweep_config = {
    'method': 'random',
    'metric': {'goal': 'minimize', 'name': 'total_loss'},
    'parameters': {
        'lr' : {
            'values': [0.006700322675853369]
        },

        'batch_size': {
            'values': [8]
        },

        'optimizer': {
            'values': ["AdamW"]
        },
    }
 }

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval_det", exist_ok=True)
            output_folder = "coco_eval_det"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks


def create_coco_format(data_pairs):
    file_dicts = []

    for i, pair in enumerate(data_pairs):
        
        filename = pair[0]

        _, _, _, height, width, _ = pair[1][0].split()
        #print(i)
        img_item = {}
        img_item['file_name'] = filename
        img_item['image_id'] = i
        img_item['height']= int(height)
        img_item['width']= int(width)

        objs = []
        for line in pair[1]:
            # print('Line: ', line)
            _, object_id, _, height, width, rle = line.split()

            #Obtain instance_id
            class_id = int(object_id) // 1000
            instance_id = int(object_id) % 1000

            if class_id == 10:
                continue
            # elif class_id == 1:
            #     class_id = 2
            # elif class_id == 2:
            #     class_id = 0

            # Create LRE to match COCO’s compressed RLE format 
            seg = {"size": [int(height), int(width)], "counts": rle}
            
            #Extract BB
            bbox = coco_mask.toBbox(seg)
            
            obj = {
                "bbox": list(map(float,bbox)),
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": seg,
                "category_id": class_id-1,
            }
            objs.append(obj)
        img_item['annotations'] = objs
        file_dicts.append(img_item)

    return file_dicts   


train_whole = create_data_pairs(path, training_ids)
train, validation = train_test_split(train_whole, test_size=0.2, random_state=42)

test = create_data_pairs(path, testing_ids)
print(len(train), len(validation), len(test))

train_list = create_coco_format(train)
val_list = create_coco_format(validation)
test_list = create_coco_format(test)

# for d in ["train", "validation", "test"]:
for d in ["train", "validation"]:
    if d == 'train':
        split = train_list
    elif d == 'validation':
        split = val_list
    elif d == 'test':
        split = test_list    

    DatasetCatalog.register("Kitti-mots_" + d, lambda d=d: split)
    MetadataCatalog.get("Kitti-mots_" + d).set(thing_classes=['car', 'person'])

metadata = MetadataCatalog.get("Kitti-mots_train")

def train():
    with wandb.init(config=wandb.config,sync_tensorboard=True ):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(f"{model}.yaml"))
        cfg.DATASETS.TRAIN = ("Kitti-mots_train",)
        cfg.DATASETS.TEST = ("Kitti-mots_validation",)
        cfg.MODEL.DEVICE = 'cuda' # cpu
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"{model}.yaml")
        cfg.SOLVER.IMS_PER_BATCH = wandb.config['batch_size']
        cfg.SOLVER.CHECKPOINT_PERIOD = 1000
        cfg.SOLVER.OPTIMIZER_NAME = wandb.config['optimizer']
        cfg.SOLVER.BASE_LR = wandb.config['lr']
        cfg.SOLVER.MAX_ITER = 3000  
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.SOLVER.STEPS = []
        cfg.OUTPUT_DIR = 'output_train_test_DET'
        cfg.INPUT.MASK_FORMAT = 'bitmask'
        cfg.TEST.EVAL_PERIOD =  200
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        #trainer = DefaultTrainer(cfg)
        trainer = CocoTrainer(cfg)
        trainer.resume_or_load(resume=False)


        s1 = t.time()
        trainer.train()
        s2 = t.time()
        print('Elapsed training time: ', s2-s1)

        # Post training evaluation on test set
        # evaluator = COCOEvaluator("Kitti-mots_test", output_dir="evaluate_results")
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50

        trainer = CocoTrainer(cfg) 
        trainer.test(cfg, trainer.model)


if __name__ == '__main__':

    sweep_id = wandb.sweep(sweep_config, project='C5-Week2_DET_final', entity='c3_mcv')
    wandb.agent(sweep_id, function=train, count=2)