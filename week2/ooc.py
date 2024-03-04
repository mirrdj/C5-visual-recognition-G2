from cocoapi.PythonAPI.pycocotools.mask import decode
import torch, detectron2
# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import time
import shutil
from tqdm import tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


path = '/ghome/group02/mcv/datasets/C5/out_of_context/'
path_output = '/ghome/group02/C5-G2/Week2/outputs/OOC/'
model = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x'
task = model.split('-')[1].split('/')[0]
method = model.split('/')[1]
threshold = 0.5
output_txt_file = 'times.txt'


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(f"{model}.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"{model}.yaml")
predictor = DefaultPredictor(cfg)

if not os.path.exists(os.path.join(path_output)):
    os.mkdir(path_output)

# if not os.path.exists(os.path.join(path_output, task)):
#     os.mkdir(os.path.join(path_output, task))


# if not os.path.exists(os.path.join(path_output, task, method)):
#     os.mkdir(os.path.join(path_output, task, method))

# if not os.path.exists(os.path.join(path_output, task, method, str(threshold))):
#     os.mkdir(os.path.join(path_output, task, method, str(threshold)))

# path_output = os.path.join(path_output, task, method, str(threshold))

count_total_images = 0
total_time = 0
info_list = []

for i, image in tqdm(enumerate(os.listdir(path))):
    im_path = os.path.join(path, image)
    im = cv2.imread(im_path)

    outputs = predictor(im)

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out = out.get_image()[:, :, ::-1]
    cv2.imwrite(path_output + str(i).zfill(4) + '.jpg', out)
