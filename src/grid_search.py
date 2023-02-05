import os
from running import run
from opts import parse_opts
from itertools import product


parameters = {
    'batch_size': [4,16,32],
    'learning_rate': [0.1, 0.01, 0.001],
    'weighted_sampling': [True, False]
}

parameter_lists = [item for item in parameters.values()]
parameter_sets = list(product(*parameter_lists))


def main():
    opt = parse_opts()
    opt.state = 'individuals'

    potential_patients = []
    for file in os.listdir(opt.data_path):
        if file.endswith(".txt"):
            potential_patients.append(os.path.basename(os.path.join(opt.data_path, file))[:3])

    opt.selected_patients = potential_patients
    # Set early stopping criterion for individuals and fine-tuning mode
    opt.early_stopping = True

    for patient in opt.selected_patients_fine_tuning:

        print('\nPatient:', patient)
        test_balanced_accuracy = []
        for set in parameter_sets:
            (opt.batch_size, opt.learning_rate, opt.weighted_sampling) = set
            print(f"Setup: batch_size = {opt.batch_size}, learning_rate = {opt.learning_rate}, weighted_sampling = {opt.weighted_sampling}")
            opt.output_path = os.path.join('..','output', f'{opt.batch_size}_{opt.learning_rate}_{opt.weighted_sampling}')
            test_bal_acc = run(opt, [patient])
            test_balanced_accuracy.append(test_bal_acc)

        best_acc = max(test_balanced_accuracy)
        best_set_idx = test_balanced_accuracy.index(best_acc)
        best_set = parameter_sets[best_set_idx]
        print(f"Best parameter set for patient {patient} with balanced accuracy {best_acc}:\n"
              f"batch_size: {best_set[0]}, learning_rate: {best_set[1]}, weighted_sampling: {best_set[2]}")


if __name__ == '__main__':
    main()
