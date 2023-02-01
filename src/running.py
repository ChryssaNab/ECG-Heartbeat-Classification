import os
import torch

from torch import nn, optim
from torchsummary import summary

from data_processing import createData, get_dataloader
from model_pretraining.test import test
from model_pretraining.train_epoch import train_epoch
from model_pretraining.val_epoch import val_epoch
from models_generation import generate_model


def run(opt, patients):

    # Create global datasets for training, validation, and test
    x_train, x_val, y_train, y_val, x_test, y_test = createData(opt, patients)

    # Get dataloaders for train, validation, and test sets
    train_dataloader = get_dataloader(x_train, y_train, batch_size=opt.batch_size, shuffle=True, drop_last=False, weightedSampling=False)
    val_dataloader = get_dataloader(x_val, y_val, batch_size=opt.batch_size, shuffle=True, drop_last=False, weightedSampling=False)
    test_dataloader = get_dataloader(x_test, y_test, batch_size=opt.batch_size, shuffle=False, drop_last=False, weightedSampling=False)

    # Generate model for pre-training
    model, parameters = generate_model(opt)
    txt = summary(model=model, input_size=(opt.input_size, opt.block_channels, opt.num_blocks, opt.kernel_size))

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

    if opt.state == "pre-training":
        save_file_path = opt.output_path
    else:
        save_file_path = os.path.join(opt.output_path, opt.state, patients)
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)

    # Start training
    for epoch in range(opt.n_epochs):
        train_state = train_epoch(epoch, train_dataloader, model, criterion, optimizer)
        val_state = val_epoch(epoch, val_dataloader, model, criterion)
        # Save checkpoints
        state = train_state.update(val_state)
        save_path = os.path.join(save_file_path, f'save_{epoch}.pth')
        torch.save(state, save_path)

    test_state = test(test_dataloader, model, criterion)

