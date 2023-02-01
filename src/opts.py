import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    # |----------------------------------------- Init settings ------------------------------------------------------|
    parser.add_argument(
        '--data_path',
        default='../dataset/mitbih_database/',
        type=str,
        help='Data directory path')
    parser.add_argument(
        '--output_path',
        default='../output/',
        type=str,
        help='Output directory path')
    parser.add_argument(
        '--selected_patients',
        default=['104', '106', '119', '200', '201', '203', '207', '208', '209', '212', '213', '217',
                 '221', '223', '231', '232', '233'],
        type=list,
        help='List with the selected patients')
    parser.add_argument(
        '--state',
        default='pre-training',
        type=str,
        help='(pre-training | individuals | fine_tuning)')
    parser.add_argument(
        '--input_size',
        default=128,
        type=int,
        help='Input dimension')
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=True)
    parser.add_argument(
        '--only_individuals',
        action='store_true',
        help='If true, no pre-training is performed.')
    parser.set_defaults(only_individuals=False)
    # |---------------------------------------- Pre-training settings -----------------------------------------------|
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='Pre-training Batch size')
    parser.add_argument(
        '--block_channels',
        default=32,
        type=int,
        help='Block channels')
    parser.add_argument(
        '--num_blocks',
        default=4,
        type=int,
        help='Number of blocks')
    parser.add_argument(
        '--kernel_size',
        default=5,
        type=int,
        help='Size of kernel in CNN')
    parser.add_argument(
        '--optimizer',
        default='Adam',
        type=str,
        help='Currently only support adam')
    parser.add_argument(
        '--lr_scheduler',
        default='reducelr',
        type=str,
        help='(reducelr | cycliclr | cosAnnealing)')
    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float,
        help='Initial learning rate')
    parser.add_argument(
        '--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument(
        '--n_epochs',
        default=30,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--no_pre_training',
        action='store_true',
        help='If true, pre-training is not performed; The pre-trained model is directly loaded.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--weighted_sampling',
        type=bool,
        help='Enable weighted sampling for training.'
    )

    args = parser.parse_args()

    return args
