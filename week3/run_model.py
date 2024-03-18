from utils import data_augmentation, get_optimizer
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np
import wandb
import yaml
from retrieval_umap import retrieval


# Load YAML file
with open('config.yml', 'r') as file:
    data = yaml.safe_load(file)

# Accessing variables from the YAML data
dataset_dir = data['DATASET_DIR']
dataset_train = data['DATASET_TRAIN']
dataset_test = data['DATASET_TEST']

BEST_MODEL_FNAME = './weights/best_siamese_IKER_miner_EASY_'

# Set the random seed for Python and NumPy
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device selected: {device}")


def run_model(params, model, trainer, validator, dataset, criterion, trial_number, miner, id):
    model_name =  BEST_MODEL_FNAME + str(id) + '_' + str(trial_number) + '.pth'

    model.to(device)

    IMG_WIDTH = params['img_size']
    IMG_HEIGHT = params['img_size']
    NUMBER_OF_EPOCHS = params['epochs']
    BATCH_SIZE = params['batch_size']

    validation_ratio = 0.2
    dataset_train = dataset(data_dir=dataset_dir + '/train/', mode='train', transform=data_augmentation(False, params, IMG_WIDTH, IMG_HEIGHT))
    dataset_size = len(dataset_train)
    validation_size = int(dataset_size * validation_ratio)
    train_size = dataset_size - validation_size

    dataset_train, dataset_val = random_split(dataset_train, [train_size, validation_size])
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    # Validation data
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    # Test Data
    dataset_test = dataset(data_dir=dataset_dir + '/test/', mode='test', transform=data_augmentation(False, params, IMG_WIDTH, IMG_HEIGHT))
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = get_optimizer(params, model)

    # Define early stopping parameters
    patience = 200
    min_delta = 0.001
    best_val_loss = np.Inf
    current_patience = 0

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

    for epoch in range(NUMBER_OF_EPOCHS):
        train_loss, train_accuracy = trainer.train(model, dataloader_train, criterion, optimizer, params, device, miner)
        val_loss, val_accuracy = validator.validation(model, dataloader_val, criterion, params, device)
        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            current_patience = 0

            # Save the best model
            print("Best model. Saving weights")
            torch.save(model.state_dict(), model_name)
        else:
            current_patience += 1
            if current_patience > patience:
                print("Early stopping.")
                break

        print(f'Epoch [{epoch+1}/{NUMBER_OF_EPOCHS}], Train Loss/Accuracy: {train_loss:.4f}/{train_accuracy:.4f} Val Loss/Accuracy: {val_loss:.4f}/{val_accuracy:.4f}')

        #Add info to wandb
        wandb.log({
            'Train Loss': train_loss,
            'Validation Loss': val_loss,
            'Train Accuracy': train_accuracy,
            'Validation Accuracy': val_accuracy,
        })


    #Test
    model.load_state_dict(torch.load(model_name))
    test_loss, test_accuracy = validator.validation(model, dataloader_test, criterion, params, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    wandb.log(data={"Test Accuracy": test_accuracy})
    wandb.log(data={"Test Loss": test_loss})
    name = f'{id}_{trial_number}_TRIALS'

    map, p1, p5 = retrieval(params['output'], name, model_name)
    wandb.log(data={"mAP": map, 'p1': p1, 'p5': p5})

    return map

def run_model_COCO(params, model, trainer, validator, dataset, criterion, trial_number, miner, id, annotations):
    model_name =  BEST_MODEL_FNAME + str(id) + '_' + str(trial_number) + '.pth'

    model.to(device)

    IMG_WIDTH = params['img_size']
    IMG_HEIGHT = params['img_size']
    NUMBER_OF_EPOCHS = params['epochs']
    BATCH_SIZE = params['batch_size']

    # validation_ratio = 0.2
    dataset_train = dataset(data_dir=dataset_dir + '/train2014/', annotations= annotations['train'], mode='train', transform=data_augmentation(True, params, IMG_WIDTH, IMG_HEIGHT))
    # dataset_size = len(dataset_train)
    # validation_size = int(dataset_size * validation_ratio)
    # train_size = dataset_size - validation_size

    # dataset_train, dataset_val = random_split(dataset_train, [train_size, validation_size])
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    # Validation data
    # dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    # Test Data
    # dataset_test = dataset(data_dir=dataset_dir + '/val2014/', mode='test', transform=data_augmentation(False, params, IMG_WIDTH, IMG_HEIGHT))
    # dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = get_optimizer(params, model)

    # Define early stopping parameters
    patience = 200
    min_delta = 0.001
    best_val_loss = np.Inf
    current_patience = 0

    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

    for epoch in range(NUMBER_OF_EPOCHS):
        train_loss, train_accuracy = trainer.train(model, dataloader_train, criterion, optimizer, params, device, miner)
        # val_loss, val_accuracy = validator.validation(model, dataloader_val, criterion, params, device)
        # Adjust learning rate based on validation loss
        # scheduler.step(val_loss)
        
        # Early stopping
        if train_loss < best_val_loss - min_delta:
            best_val_loss = train_loss
            current_patience = 0

            # Save the best model
            print("Best model. Saving weights")
            torch.save(model.state_dict(), model_name)
        else:
            current_patience += 1
            if current_patience > patience:
                print("Early stopping.")
                break

        print(f'Epoch [{epoch+1}/{NUMBER_OF_EPOCHS}], Train Loss/Accuracy: {train_loss:.4f}/{train_accuracy:.4f}')

        #Add info to wandb
        wandb.log({
            'Train Loss': train_loss,
            # 'Validation Loss': val_loss,
            'Train Accuracy': train_accuracy,
            # 'Validation Accuracy': val_accuracy,
        })


    #Test
    # model.load_state_dict(torch.load(model_name))
    # test_loss, test_accuracy = validator.validation(model, dataloader_test, criterion, params, device)
    # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # wandb.log(data={"Test Accuracy": test_accuracy})
    # wandb.log(data={"Test Loss": test_loss})
    name = f'{id}_{trial_number}_TRIALS'

    map, p1, p5 = retrieval(params['output'], name, model_name)
    wandb.log(data={"mAP": map, 'p1': p1, 'p5': p5})

    return map