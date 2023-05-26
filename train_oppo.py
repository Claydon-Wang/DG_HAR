# encoding=utf-8
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import os
import os.path as osp
# from train import train
# from utils import set_name
# import network_ucihar as net
# import data_preprocess_ucihar
import torch
import argparse
import yaml
from data_processing import data_process_oppo
from models.cnn import CNN_choose
from models.deepconvlstm import DeepConvLSTM_choose
from trainer import Trainer
from utils import get_dataset, get_model

def main():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(type(DEVICE))
    parser = argparse.ArgumentParser(description='argument setting of network')
    #---------------------------------- dataset_data -----------------------------------------#
    parser.add_argument('--dataset', type=str, default='oppo', help='name of feature dimension')
    parser.add_argument('--n_feature', type=int, default=77, help='name of feature dimension')
    parser.add_argument('--len_sw', type=int, default=30, help='length of sliding window')
    parser.add_argument('--n_class', type=int, default=17, help='number of class')

    #---------------------------------- dataset_domain ---------------------------------------#
    parser.add_argument('--target_domain', type=str, default='S1', help='the target domain, [S1, S2, S3, S4]')
    parser.add_argument('-n_domains', type=int, default=4, help='number of total domains actually')
    parser.add_argument('-n_target_domains', type=int, default=1, help='number of target domains')

    #------------------------------------ training -------------------------------------------#
    parser.add_argument('--batch_size', type=int, default=512, help='batch size of training')
    parser.add_argument('--n_epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--seed', type=int, default=10, help='seed')
    # parser.add_argument('--device', default=DEVICE, help='seed')
    parser.add_argument('--interval_validate', type=int, default=2, help='interval epoch number to valide the model')
    parser.add_argument('--out_path', type=str, default='./logs', help='log path')
    parser.add_argument('--save_model', type=bool, default=False, help='save model or not')

    #-------------------------------------- model -------------------------------------------#
    parser.add_argument('--model', type=str, default='cnn', help='choose backbone')
    parser.add_argument('--res', type=bool, default=False, help='CNN resnet or not')
    parser.add_argument('--resume', type=bool, default=False, help='load the pretrain model')

    #---------------------------------- optimizer -------------------------------------------#
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate',)
    parser.add_argument('--wd', type=float, default=5e-4, help='weight delay')
    parser.add_argument('--lr_decrease_rate', type=float, default=0.5, help='ratio multiplied to initial lr')
    parser.add_argument('--lr_decrease_interval', type=int, default=20, help='lr decrease interval')

    #---------------------------------- other config ----------------------------------------#
    parser.add_argument('--cfg', type=str, default=None, help='other cfg')

    args = parser.parse_args()

    # load additional config file
    if args.cfg is not None: 
        with open(args.cfg, 'r') as f:
            new_cfg = yaml.full_load(f)
        parser.set_defaults(**new_cfg)
        args = parser.parse_args()


    # load the time and the output dir
    # now = datetime.now()
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    args.out = osp.join(args.out_path, args.dataset, args.model,  args.target_domain, current_time)
    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    # load the seed and device
    torch.manual_seed(10)
    args.device = DEVICE



    # dataset
    source_loader, target_loader = get_dataset(args=args)

    # model
    model = get_model(args=args)
    model = model.cuda()

    # load weights
    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)


    # 3. optimizer
    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.wd
    )


    # build trainer
    trainer = Trainer(
        args = args,
        model = model,
        train_loader = source_loader,
        val_loader = target_loader,
        optim = optim,
    )

    # train model
    start_epoch = 0
    start_iteration = 0
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
