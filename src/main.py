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
    print("The list of the selected patients is:", opt.selected_patients)

    if opt.state == "pre-training":
        # Pre-train the model on all the patients
        run(opt,  opt.selected_patients)
    elif opt.state == "individuals" or opt.state == "fine_tuning":
        for patient in opt.selected_patients:
            run(opt, [patient])


if __name__ == '__main__':
    main()
