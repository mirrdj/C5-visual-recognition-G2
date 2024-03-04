import numpy as np
from pycocotools import mask as coco_mask
import cv2
import os
from utils import create_data_pairs
from decode_I import create_coco_format
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import random
from detectron2.utils.visualizer import Visualizer

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode

path = '/ghome/group02/mcv/datasets/C5/KITTI-MOTS/'
pathOutput = '/ghome/group02/C5-G2/Week2/outputs/'
model = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x"
threshold= 0.5
task = model.split('-')[1].split('/')[0]
method = model.split('/')[1]
training_ids = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
testing_ids  = [2, 6, 7, 8, 10, 13, 14, 16, 18]

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(f"{model}.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"{model}.yaml")
predictor = DefaultPredictor(cfg)

for d in ["train", "test"]:
    if d == 'train':
        idxs = training_ids
    elif d == 'test':
        idxs = testing_ids

    data_pairs = create_data_pairs(path, idxs)

    DatasetCatalog.register("Kitti-mots_" + d, lambda d=d: create_coco_format(data_pairs))
    MetadataCatalog.get("Kitti-mots_" + d).set(thing_classes=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)

print(f"Model name: {model}")

evaluator = COCOEvaluator("Kitti-mots_test", output_dir="evaluate_results")
val_loader = build_detection_test_loader(cfg, "Kitti-mots_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))

print("-------------------")

