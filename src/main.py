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
        for patient in opt.selected_patients_fine_tuning:
            print('\nPatient:', patient)
            test_bal_acc = run(opt, [patient])
            test_balanced_accuracy.append(test_bal_acc)

    print(f"Test Balanced Accuracy: {test_balanced_accuracy}")


if __name__ == '__main__':
    main()
