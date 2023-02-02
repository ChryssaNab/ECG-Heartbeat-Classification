import torch
from models import cnn1D


def generate_model(opt):
    assert opt.state in [
        'pre-training', 'individuals', 'fine_tuning'
    ]

    if opt.state == 'pre-training' or opt.state == "individuals":
        model = cnn1D.cnn_1D(
            num_blocks=opt.num_blocks,
            block_channels=opt.block_channels,
            kernel_size=opt.kernel_size)

    elif opt.state == "fine_tuning":

        model = cnn1D.cnn_1D(
            num_blocks=opt.num_blocks,
            block_channels=opt.block_channels,
            kernel_size=opt.kernel_size)

        if opt.pretrain_path:
            print('Loading pretrained model {}'.format(opt.pretrain_path))
            checkpoint = torch.load(opt.pretrain_path)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model = model.get_frozen_cnn()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    return model, model.parameters()
