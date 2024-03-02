import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    # |----------------------------------------- Init settings ------------------------------------------------------|
    parser.add_argument(
        '--data_path',
        default='./dataset/mitbih_database/',
        type=str,
        help='The data directory path under which the dataset lies.')
    parser.add_argument(
        '--output_path',
        default='./output/',
        type=str,
        help='The output directory path where the checkpoints and log files are created.')
    parser.add_argument(
        '--state',
        default='pre-training',
        type=str,
        help='(pre-training | baseline individuals | fine-tuning individuals)')
    parser.add_argument(
        '--selected_patients_fine_tuning',
        default=['100', '102', '104', '105', '106', '108', '114', '116', '119', '200', '202', '201', '203', '205',
                 '208', '209', '212', '210', '213', '215', '217', '219', '220', '222', '221', '228', '223', '231',
                 '233'],
        type=str,
        nargs='+',
        help='The list of the selected patients earmarked for the experiments. '
             'Only applies to the baseline individual models and fine-tuning models that target certain patients.')
    parser.add_argument(
        '--input_size',
        default=128,
        type=int,
        help='The size of each pulse-width window')
    parser.add_argument(
        '--pretrain_path', default='./output/save_16.pth', type=str,
        help='The pre-trained model checkpoint (.pth)')

    # |------------------------------------------ CNN default settings ----------------------------------------------|
    parser.add_argument(
        '--num_blocks',
        default=4,
        type=int,
        help='Number of blocks')
    parser.add_argument(
        '--block_channels',
        default=32,
        type=int,
        help='Block channels')
    parser.add_argument(
        '--kernel_size',
        default=5,
        type=int,
        help='The convolution kernel size of CNN')

    # |--------------------------------------- Training global settings ----------------------------------------------|
    parser.add_argument(
        '--optimizer',
        default='Adam',
        type=str,
        help='(Adam | SGD)')
    parser.add_argument(
        '--lr_scheduler',
        default='reducelr',
        type=str,
        help='(reducelr | cycliclr | cosAnnealing)')
    parser.add_argument(
        '--weight_decay', default=1e-4, type=float, help='Weight decay hyperparameter value of optimizer')
    parser.add_argument(
        '--early_stopping',
        action='store_false',
        help='Set to TRUE only for baseline or fine-tuning mode.')
    parser.set_defaults(early_stopping=False)
    parser.add_argument(
        '--n_epochs',
        default=30,
        type=int,
        help='The maximum number of total epochs to run.')

    # |---------------------------------------- Pre-training settings -----------------------------------------------|
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='Batch size used during pre-training.')
    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float,
        help='Initial learning rate')
    parser.add_argument(
        '--weighted_sampling',
        type=bool,
        default=True,
        help='Enable weighted sampling for training.'
    )

    args = parser.parse_args()

    return args
