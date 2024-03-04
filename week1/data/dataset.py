from torch.utils.data import DataLoader, Dataset
from utils import *
import numpy as np

DATASET_DIR = 'MIT_small_train'
DATASET_TEST = 'MIT_large_train'

batch_size = 64

train_dl = DataLoader(cropped_dataset, batch_size, shuffle=True, num_workers=3, pin_memory=True)
show_batch(train_dl)


class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data
            self.transforms = T.Compose(
                T.ToPILImage(),
                T.CenterCrop(0.75 * 64),
                T.RandomHorizontalFlip(),
                T.ToTensor()
            )

        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            img = self.data[idx]
            img = self.transform(img)
            return img

def create_dataset(path, substr_mean=False):
    X_train = np.load(DATASET_DIR)

    mean_r, mean_g, mean_b = (0, 0, 0)

    if substr_mean:
        # Compute the mean values of the dataset
        mean_r, mean_g, mean_b = utils.compute_dataset_mean(DATASET_DIR + '/train/')
        std = utils.compute_dataset_std(DATASET_DIR + '/train/', (mean_r, mean_g, mean_b))
        print(mean_r, mean_g, mean_b)
        print(std)
