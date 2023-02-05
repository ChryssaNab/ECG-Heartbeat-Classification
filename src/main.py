import os

import numpy as np
import random
import torch

from opts import parse_opts
from running import run

# Set seed for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Set optimal parameters for each patient as occurred from GridSearch
optimal_parameters = {'100': {'batch_size': 4, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '102': {'batch_size': 4, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '104': {'batch_size': 4, 'learning_rate': 0.001, 'weighted_sampling': False},
                      '105': {'batch_size': 4, 'learning_rate': 0.001, 'weighted_sampling': False},
                      '106': {'batch_size': 16, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '108': {'batch_size': 32, 'learning_rate': 0.01, 'weighted_sampling': True},
                      '114': {'batch_size': 4, 'learning_rate': 0.001, 'weighted_sampling': False},
                      '116': {'batch_size': 16, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '119': {'batch_size': 4, 'learning_rate': 0.01, 'weighted_sampling': True},
                      '200': {'batch_size': 32, 'learning_rate': 0.01, 'weighted_sampling': True},
                      '202': {'batch_size': 16, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '201': {'batch_size': 32, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '203': {'batch_size': 4, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '205': {'batch_size': 32, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '208': {'batch_size': 4, 'learning_rate': 0.01, 'weighted_sampling': False},
                      '209': {'batch_size': 4, 'learning_rate': 0.001, 'weighted_sampling': False},
                      '212': {'batch_size': 16, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '210': {'batch_size': 16, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '213': {'batch_size': 4, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '215': {'batch_size': 4, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '217': {'batch_size': 4, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '219': {'batch_size': 16, 'learning_rate': 0.01, 'weighted_sampling': True},
                      '220': {'batch_size': 16, 'learning_rate': 0.01, 'weighted_sampling': True},
                      '222': {'batch_size': 32, 'learning_rate': 0.001, 'weighted_sampling': False},
                      '221': {'batch_size': 32, 'learning_rate': 0.001, 'weighted_sampling': False},
                      '228': {'batch_size': 16, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '223': {'batch_size': 16, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '231': {'batch_size': 4, 'learning_rate': 0.001, 'weighted_sampling': True},
                      '233': {'batch_size': 32, 'learning_rate': 0.001, 'weighted_sampling': False}}


def main():
    opt = parse_opts()

    # Collect all patients data
    potential_patients = []
    for file in os.listdir(opt.data_path):
        if file.endswith(".txt"):
            potential_patients.append(os.path.basename(os.path.join(opt.data_path, file))[:3])

    opt.selected_patients = potential_patients
    test_balanced_accuracy = []

    if opt.state == "pre-training":
        # Pre-train the model on all the patients
        test_balanced_accuracy = run(opt,  opt.selected_patients)
    elif opt.state == "individuals" or opt.state == "fine_tuning":
        print("\nThe list of the selected patients is:", opt.selected_patients_fine_tuning)
        # Set early stopping criterion for individuals and fine-tuning mode
        opt.early_stopping = True
        # Run either fine-tuning or simple training for each patient
        for patient in opt.selected_patients_fine_tuning:
            print('\nPatient:', patient)
            if patient in optimal_parameters.keys():
                opt.batch_size = optimal_parameters[patient]['batch_size']
                opt.learning_rate = optimal_parameters[patient]['learning_rate']
                opt.weighted_sampling = optimal_parameters[patient]['weighted_sampling']
            test_bal_acc = run(opt, [patient])
            test_balanced_accuracy.append(test_bal_acc)

    print(f"Test Balanced Accuracy: {test_balanced_accuracy}")


if __name__ == '__main__':
    main()
