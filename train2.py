r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.
"""
import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import transforms as T
import utils
from dataset.icdar_dataset import ICDAR
from dataset.marnot_dataset import MarnotDataset
from dataset.tablebank_dataset import TableBank
from engine import train_one_epoch, evaluate
from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from models.faster_rcnn import get_faster_crnn_model, get_mask_crnn_model
from models.segmentation import get_segmentation_model

import copy


def get_transform(train):
    transforms = []
    if train:
        # transforms.append(T.OneOf([
        #     # T.RandomDistortions(0.5),
        #     T.Blur(0.5, 2),
        #     T.BlurThreshold(0.5, 2, 0.5),
        #     T.BinaryBlur(0.5, 2),
        # ]))
        transforms.append(T.ToTensor())
        transforms.append(T.RandomHorizontalFlip(0.5))
    else:
        transforms.append(T.ToTensor())
    return T.Compose(transforms)


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    num_classes = len(MarnotDataset.classes) + 1

    device = torch.device(args.device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # Data loading code
    print("Loading data")
    torch.manual_seed(1)
    if args.model == 'faster_rcnn':
        TASK = 'FASTER_CRNN'
    else:
        TASK = 'MY_SEGMENTATION'
    MarnotDataset.TASK = TASK
    ICDAR.TASK = TASK
    # marmot dataset
    if 'data_marmot' in args.data_path:
        dataset = MarnotDataset(transforms=get_transform(train=False))

        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset, indices[-50:])
    # iCDAR dataset
    elif 'data_icdar' in args.data_path:
        dataset = ICDAR(transforms=get_transform(train=False), mode='all')

        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-40])
        dataset_test = torch.utils.data.Subset(dataset, indices[-40:])
    elif 'TableBank_data' in args.data_path:
        dataset = TableBank(transforms=get_transform(train=False), sources=['Latex'])

        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-40])
        dataset_test = torch.utils.data.Subset(dataset, indices[-40:])

    else:
        dataset_marnot = MarnotDataset(transforms=get_transform(train=False))
        dataset_icdar = ICDAR(transforms=get_transform(train=False), mode='all')
        # dataset_tablebank = TableBank(transforms=get_transform(train=False), sources=['Latex'])
        # dataset = dataset_marnot + dataset_icdar + dataset_tablebank

        dataset = dataset_marnot + dataset_icdar

        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset, indices[-50:])

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)
    # for i, o in range(len(dataset_test)):
    #     print(i)

    print("Creating model")
    if args.model == 'faster_rcnn':
        model = get_faster_crnn_model(num_classes)
        # model = get_mask_crnn_model(num_classes)
    else:
        model = get_segmentation_model('DinkNet34', num_classes)
    print(model)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    args.lr = args.world_size / 8 * args.lr
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch Detection Training')

    parser.add_argument('--data-path', default='data/data_marmot', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='faster_rcnn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=0, type=int)
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
# CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2504 --use_env train2.py --batch-size 3 --world-size 2 --model faster_rcnn --data-path data
