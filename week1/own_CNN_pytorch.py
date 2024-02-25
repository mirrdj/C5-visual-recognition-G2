from utils import *
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import optuna
import wandb

#DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_large_train'
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
DATASET_TEST = '/ghome/mcv/datasets/C3/MIT_large_train'

# Put your key
# wandb.login(key='4c0b25a1f87331e99edadbaa2cf9568a452224ff') #GOIO
wandb.login(key='4127c8a2b851657f629b6f8f83ddc2e3415493f2')  # IKER
# wandb.login(key='50315889c64d6cfeba1b57dc714112418a50e134') #Xavi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def model(params, number):
    """
    CNN model configuration
    """
    # Define the test data generator for data augmentation and preprocessing
    IMG_WIDTH = int(params['img_size'])
    IMG_HEIGHT = int(params['img_size'])
    BATCH_SIZE = int(params['batch_size'])
    NUMBER_OF_EPOCHS = params['epochs']
    BEST_MODEL_FNAME = '/ghome/group02/C5-G2/Week1/weights'

    # if params['substract_mean'] == 'True':
    #     # Compute the mean values of the dataset
    #     mean_r, mean_g, mean_b = compute_dataset_mean(DATASET_DIR+'/train/')
    #     std = compute_dataset_std(DATASET_DIR+'/train/', (mean_r, mean_g, mean_b))
    #     print(mean_r, mean_g, mean_b)
    #     print(std)


    # transform = transforms.Compose([
    #     transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    #     transforms.ToTensor(),
    # ])

    # Train data
    #train_data_generator = data_augmentation(True, params, (mean_r, mean_g, mean_b), std)
    #train_dataset = load_data(train_data_generator, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, directory=DATASET_DIR + '/train/')

    dataset_train = MiTDataset(data_dir=DATASET_DIR + '/train/', transform=data_augmentation(True, params, IMG_WIDTH, IMG_HEIGHT))
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    # Validation data
    #validation_data_generator = data_augmentation(False, params, (mean_r, mean_g, mean_b), std)
    #validation_dataset = load_data(validation_data_generator, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, directory=DATASET_DIR + '/test/')
    dataset_val = MiTDataset(data_dir=DATASET_DIR + '/test/', transform=data_augmentation(False, params, IMG_WIDTH, IMG_HEIGHT))
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    # Test Data
    #test_data_generator = data_augmentation(False, params, (mean_r, mean_g, mean_b), std)
    #test_dataset = load_data(test_data_generator, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, directory=DATASET_TEST + '/test/')
    dataset_test = MiTDataset(data_dir=DATASET_TEST + '/test/', transform=data_augmentation(False, params, IMG_WIDTH, IMG_HEIGHT))
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)


    model = Model(params)
    print(model)
    model.to(device)

    #Plot model
    #plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    # Get the total number of parameters
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    #Get optimizer
    optimizer = get_optimizer(params, model)

    # Create the ReduceLROnPlateau callback
    
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # Monitor validation loss
    #                             factor=0.1,          # Reduce the learning rate by a factor of 0.1
    #                             patience=5,          # Number of epochs with no improvement after which learning rate will be reduced
    #                             min_lr=1e-6)         # Minimum learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

    # earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1,
    #                           baseline=None, restore_best_weights=True)

    # Define early stopping parameters
    patience = 200
    min_delta = 0.001
    best_val_loss = np.Inf
    current_patience = 0

    # checkpoint = ModelCheckpoint(BEST_MODEL_FNAME,  # Filepath to save the model weights
    #                              monitor='val_accuracy',  # Metric to monitor for saving the best model
    #                              save_best_only=True,  # Save only the best model (based on the monitored metric)
    #                              mode='max', # 'max' or 'min' depending on whether the monitored metric should be maximized or minimized
    #                              verbose=1)  # Verbosity mode, 1 for progress updates

    # callbacks = [reduce_lr, earlystop, checkpoint]

    
    
    # compile the model (should be done *after* setting layers to non-trainable)
    #model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUMBER_OF_EPOCHS):
        train_loss, train_accuracy = train(model, dataloader_train, criterion, optimizer, params, device)
        val_loss, val_accuracy = validation(model, dataloader_val, criterion, params, device)
        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            current_patience = 0
            # Save the best model
            print("Best model. Saving weights")
            torch.save(model.state_dict(), f'{BEST_MODEL_FNAME}/best_model.pth')
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
    model.load_state_dict(torch.load(f'{BEST_MODEL_FNAME}/best_model.pth'))
    test_loss, test_accuracy = validation(model, dataloader_test, criterion, params, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    
    # train the model on the new data for a few epochs
    # history = model.fit(train_dataset,
    #                     epochs=NUMBER_OF_EPOCHS,
    #                     validation_data=validation_dataset,
    #                     verbose=2,
    #                     callbacks=callbacks)
    
    # Load the best weights before calling model.evaluate
    # model.load_weights(BEST_MODEL_FNAME)
    # visualize_weights(model)

    #result = model.evaluate(test_dataset, verbose=0)
    ratio = test_accuracy/(total_parameters/100000)

    wandb.log(data={"Test Accuracy": test_accuracy})
    wandb.log(data={"Test Loss": test_loss})
    wandb.log(data={"Total Parameters": total_parameters})
    wandb.log(data={"Ratio": ratio})

    #print(result)
    #print(history.history.keys())
    #print(ratio)

    return ratio


def objective_model_cv(trial):
    
    params = {
        #'unfrozen_layers': trial.suggest_categorical('unfrozen_layers', ["1"]),  # 1,2,3,4,5
        
        'substract_mean': trial.suggest_categorical('substract_mean', ['True']),
        'batch_size': trial.suggest_categorical('batch_size', ['16']),  # 8,16,32,64
        'img_size': trial.suggest_categorical('img_size', ['224']),  # 8,16,32,64,128,224,256
        'lr': trial.suggest_float('lr', 1,1),  # 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3
        'optimizer': trial.suggest_categorical('optimizer', ['adadelta']),  # adadelta, adam, sgd, RMSprop


        'activation': trial.suggest_categorical('activation', ['relu']),
        'n_filters_1': trial.suggest_categorical('n_filters_1', ['64']),
        'n_filters_2': trial.suggest_categorical('n_filters_2', ['16']),
        'n_filters_3': trial.suggest_categorical('n_filters_3', ['16']),
        'n_filters_4': trial.suggest_categorical('n_filters_4', ['16']),

        'kernel_size_1': trial.suggest_categorical('kernel_size_1', ['3']),
        'kernel_size_2': trial.suggest_categorical('kernel_size_2', ['3']),
        'kernel_size_3': trial.suggest_categorical('kernel_size_3', ['3']),
        'kernel_size_4': trial.suggest_categorical('kernel_size_4', ['3']),

        'stride': trial.suggest_int('stride', 1,1),

        'pool': trial.suggest_categorical('pool', ['max']),

        'padding': trial.suggest_categorical('padding', ['same']),
        'neurons': trial.suggest_categorical('neurons', ['256']),

        'data_aug': trial.suggest_categorical('data_aug', ['none']),
        'momentum': trial.suggest_float('momentum', 0.95, 0.95),
        'dropout': trial.suggest_categorical('dropout', ['0']),
        'bn': trial.suggest_categorical('bn', ['True']),
        'L2': trial.suggest_categorical('L2', ['False']),
        'epochs': trial.suggest_int('epochs', 100,100),
        'depth': trial.suggest_int('layer_n', 2,2), 
        'pruning_thr': trial.suggest_float('pruning_thr', 0.1, 0.1),
        'output': trial.suggest_int('output', 8,8),
    }
    
    config = dict(trial.params)
    config['trial.number'] = trial.number

    execution_name = 'C5_test'

    wandb.init(
        project='C5_test',
        entity='c3_mcv',
        name=execution_name,
        config=config,
        reinit=True,
    )
    ratio = model(params, trial.number)

    # report validation accuracy to wandb
    # for epoch in range(len(history.history['accuracy'])):
    #     wandb.log({
    #         'Train Loss': history.history['loss'][epoch],
    #         'Validation Loss': history.history['val_loss'][epoch],
    #         'Train Accuracy': history.history['accuracy'][epoch],
    #         'Validation Accuracy': history.history['val_accuracy'][epoch],
    #     })

    # wandb.log(data={"Test Accuracy": result[1]})
    # wandb.log(data={"Test Loss": result[0]})
    # wandb.log(data={"Total Parameters": total_parameters})
    # wandb.log(data={"Ratio": ratio})
    
    return ratio


study = optuna.create_study(direction="maximize", study_name='C5-Week1')
study.optimize(objective_model_cv, n_trials=1)