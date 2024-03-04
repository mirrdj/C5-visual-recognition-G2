import re
import os
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import logging
import numpy as np

def get_file_number(str):
    pattern = r'0*(\d+)\.'
    
    match = re.search(pattern, str)

    # Check if a match is found
    if match:
        extracted_numbers = match.group(1)
        return int(extracted_numbers)
    else:
        return None

def create_data_pairs(dataset_path, idxs):
    img_paths_dict = get_split_paths(idxs, dataset_path)

    pairs = []
    for idx in idxs:
        
        folder_formatted = "{:04d}".format(idx)
        txt_path = f'{dataset_path}/instances_txt/{folder_formatted}.txt'
        group = group_lines(txt_path)

        for frame_id, lines in group.items():
            pairs.append([img_paths_dict[idx][int(frame_id)], lines])

    return pairs

def group_lines(filename):
    """
    Group lines according to timeframe so that different instances are passed
    as part of the same image.
    """
    grouped_lines = dict()

    with open(filename, 'r') as file:
        for line in file:
            time_frame = line.split()[0]
            class_id = line.split()[2]

            if class_id == 10:
                continue

            if time_frame not in grouped_lines:
                grouped_lines[time_frame] = []

            grouped_lines[time_frame].append(line)

    return grouped_lines


def get_split_paths(ids_array, path):
    imgs_paths = dict()
    
    for folder_id in ids_array:
        folder_formatted = "{:04d}".format(folder_id)
        folder_path = os.path.join(path, 'training', 'image_02', folder_formatted)

        if folder_id not in imgs_paths:
            imgs_paths[folder_id] = []

        for img_path in os.listdir(folder_path):
            imgs_paths[folder_id].append(os.path.join(folder_path, img_path))
        
        imgs_paths[folder_id].sort()
    
    sorted_dict = dict(sorted(imgs_paths.items()))
    return sorted_dict



class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


