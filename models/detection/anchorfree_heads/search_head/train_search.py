import os
import sys
sys.path.append('/home/haida_sunxin/lqx/code/llseg')
import tqdm
import torch
import logging
import torch.nn as nn
from models.detection.anchorfree_heads.search_head.search_datasets import COCOValSearch, get_dataloader

import numpy as np
import torch.backends.cudnn as cudnn

from engine.launch import launch
from engine.defaults import default_argument_setup

from torch.nn.parallel import DistributedDataParallel

import utils.comm as comm
from models.detection.anchorfree_heads.search_head.model_search import SearchHead
from models.detection.anchorfree_heads.search_head.arch_search import SearchArch


def get_search_dicts(root_dir):
    dict_list = os.listdir(root_dir)
    dict_list = np.array(dict_list)
    np.random.shuffle(dict_list)

    total_length = len(dict_list)
    split = total_length // 2
    train_dict, val_dict = dict_list[:split], dict_list[split:]

    return train_dict, val_dict

def get_train_val_dataset(root_dir, batch_size):
    train_list, val_list = get_search_dicts(root_dir)
    train_dataset = COCOValSearch(root_dir, train_list)
    val_dataset = COCOValSearch(root_dir, val_list)
    train_loader = get_dataloader(train_dataset, batch_size)
    val_loader = get_dataloader(val_dataset, batch_size)
    return train_loader, val_loader

def main(args):
    # set random seed
    seed = 100
    np.random.seed(seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # dataset part
    root_dir = '/home/haida_sunxin/lqx/data/search'
    batch_size = 1
    out_dir = './out_noskip'
    os.makedirs(out_dir, exist_ok=True)

    # logging config
    if comm.is_main_process():
        log_format = '%(asctime)s ----- %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
        if os.path.exists(os.path.join(os.path.join(out_dir, 'log.txt'))):
            os.remove(os.path.join(out_dir, 'log.txt'))
        fh = logging.FileHandler(os.path.join(out_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

    # Model part
    model_lr = 0.01
    model_lr_min = 0.001
    momentum = 0.9
    weight_decay = 3e-4
    epochs = 15

    model = SearchHead(C_in=256, C_mid=256, C_out=256, num_classes=80, layers=1, multiplier=2).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=model_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=model_lr_min)
    model = DistributedDataParallel(model, device_ids=[comm.get_local_rank()])

    # Search Controller
    architect = SearchArch(model.module)

    # Training part
    for i in range(epochs):

        lr = scheduler.get_lr()[0]
        if comm.is_main_process():
            logging.info("================================ \nEpoch : %d | Lr : %f" % (i, lr))
        train_loader, val_loader = get_train_val_dataset(root_dir, batch_size)

        loss_dict = train(train_loader, val_loader, model, architect, optimizer, lr, i)
        genotype = model.module.genotype()

        if comm.is_main_process():
            logging.info("GenoType : %s", genotype)
            loss_str = 'loss_dict : '
            for k,v in loss_dict.items():
                loss_str += " " + k + " : " + str(v.item())
            logging.info("Loss dict is : %s" % loss_str)
            logging.info("================================ \n")

        scheduler.step()
        # save model
        if (i+1) % 5 == 0:
            torch.save(model.module, os.path.join(out_dir, "Epoch_" + str(i) + ".pth"))


def train(train_loader, val_loader, model, architect, optimizer, lr, epochs):

    val_iter = iter(val_loader)
    train_iter = iter(train_loader)

    print_period = 25
    total_loss = 0

    # val losses
    loss_dict = {
        'loss_fcos_cls': [],
        'loss_fcos_loc': [],
        'loss_fcos_ctr': []
    }

    for i in tqdm.tqdm(range(len(train_loader)), disable=not comm.is_main_process()): #
        data = next(train_iter)
        val_data = next(val_iter)
        if epochs > 6:
            architect.search_step(data, val_data, lr, optimizer, unrolled=True)

        loss = model.module.model_forward(data)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), 5)
        optimizer.step()

        if comm.is_main_process():
            total_loss += loss.item()
            if (i+1) % print_period == 0:
                logging.info("Training iter %d/%d | Avg Loss %f" % (i, len(train_loader), total_loss / print_period))
                total_loss = 0
                with torch.no_grad():
                    losses = model.module.model_forward(val_data, flag=True)
                    for k,v in loss_dict.items():
                        loss_dict[k].append(float(losses[k].item()))
    for k,v in loss_dict.items():
        loss_dict[k] = np.mean(v)
    return loss_dict


if __name__ == '__main__':
    # TODO Early Stopping
    # TODO Eval the Eigenvalue
    # TODO Separately search the box head and reg head

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

    args = default_argument_setup().parse_args()
    args.num_gpus = 4

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

