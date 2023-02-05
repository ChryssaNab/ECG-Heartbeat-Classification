import os
import shutil

import numpy as np
import torch

from torch import nn, optim
from torch.optim import lr_scheduler

from data_processing import createData, get_dataloader
from training.test import test
from training.train_epoch import train_epoch
from training.val_epoch import val_epoch
from models_generation import generate_model
from utils import WriteLogger


def run(opt, patients):

    # Create datasets for training, validation, and test
    x_train, x_val, y_train, y_val, x_test, y_test = createData(opt, patients)

    # Get dataloaders for train, validation, and test sets
    train_dataloader = get_dataloader(x_train, y_train, batch_size=opt.batch_size, drop_last=False, weightedSampling=opt.weighted_sampling)
    val_dataloader = get_dataloader(x_val, y_val, batch_size=opt.batch_size, drop_last=False, weightedSampling=False)
    test_dataloader = get_dataloader(x_test, y_test, batch_size=opt.batch_size, drop_last=False, weightedSampling=False)

    # Generate model
    model, parameters = generate_model(opt)

    # Define loss function (criterion) and optimizer
    criterion = nn.BCELoss(reduction="mean")

    if opt.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt.learning_rate,
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=opt.dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov
        )

    # Define learning rate scheduler
    if opt.lr_scheduler == 'reducelr':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    elif opt.lr_scheduler == 'cycliclr':
        if opt.optimizer.lower() == 'adam':
            cycle_momentum = False
        else:
            cycle_momentum = True
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.3, step_size_up=20, step_size_down=1000, cycle_momentum=cycle_momentum)
    elif opt.lr_scheduler == 'cosAnnealing':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader), eta_min=0, last_epoch=-1)
    else:
        scheduler = None

    # Create output files
    if opt.state == "pre-training":
        save_file_path = opt.output_path
    else:
        save_file_path = os.path.join(opt.output_path, opt.state, patients[0])

    # Remove folder if already exists
    if os.path.exists(save_file_path):
        shutil.rmtree(save_file_path)
    os.makedirs(save_file_path)

    # Collect evaluation metrics during training, validation, and test processes
    train_logger = WriteLogger(
        os.path.join(save_file_path, 'train.log'), ['epoch', 'loss', 'lr', 'accuracy', 'balanced_accuracy', 'recall', 'precision', 'F1-score'])
    val_logger = WriteLogger(
        os.path.join(save_file_path, 'val.log'),  ['epoch', 'loss', 'accuracy', 'balanced_accuracy', 'recall', 'precision', 'F1-score'])
    test_logger = WriteLogger(
        os.path.join(save_file_path, 'test.log'),  ['loss', 'accuracy', 'balanced_accuracy', 'recall', 'precision', 'F1-score'])

    min_val_loss = np.inf
    n_epochs_stop = 5
    epochs_no_improve = 0
    for epoch in range(opt.n_epochs):
        # Train the model on the training set
        train_state = train_epoch(epoch, train_dataloader, model, criterion, optimizer, train_logger)
        # Evaluate the model on the validation set
        scheduler, val_loss = val_epoch(epoch, val_dataloader, model, criterion, scheduler, val_logger)

        if opt.state == "individuals" or opt.state == "fine_tuning":
            # If the validation loss is at a minimum
            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
            if epoch > 10 and epochs_no_improve == n_epochs_stop:
                # Activate early stopping criterion
                break

        # Save model checkpoints
        save_path = os.path.join(save_file_path, f'save_{epoch}.pth')
        torch.save(train_state, save_path)

    # Evaluate the model on the test set after training is completed
    test_balanced_accuracy = test(test_dataloader, model, criterion, test_logger)

    return test_balanced_accuracy

