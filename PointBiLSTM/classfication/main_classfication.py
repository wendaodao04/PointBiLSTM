"""
for training with resume functions.
Usage:
python main.py --model PointNet --msg demo
or
CUDA_VISIBLE_DEVICES=0 nohup python main.py --model PointNet --msg demo > nohup/PointNet_demo.out &
"""
import argparse
import os
import logging
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from torchvision import transforms
from ScanObjectNN import ScanObjectNN
# from validatin_scan import ScanObjectNN
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
import numpy as np
from models.pointlstm import PointBilstm
from pc_transform import *
from numpy import random
import warnings
warnings.filterwarnings("ignore")

from ptflops import get_model_complexity_info
from thop import profile
import time

class PointcloudRotateZ(object):
    def __call__(self, points):
        theta = torch.rand(1) * 2 * torch.pi
        cosval = torch.cos(theta)
        sinval = torch.sin(theta)

        rot = torch.tensor([
            [cosval, -sinval, 0],
            [sinval,  cosval, 0],
            [0,       0,      1]
        ], dtype=points.dtype, device=points.device).squeeze()

        return torch.matmul(points, rot)

def shuffle_tensor(x):
    B, D, _ = x.size()
    index = [i for i in range(len(x[0][0]))]
    random.shuffle(index)
    x[:,:]=x[:,:,index]
    return x

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='pointBilstm', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=15, type=int, help='default value for classes of ScanObjectNN')
    parser.add_argument('--epoch', default=500, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--learning_rate', default= 0.01, type=float, help='learning rate in training')

    # 1e-4
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--smoothing', action='store_true', default=True, help='loss smoothing')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=4, type=int, help='workers')
    return parser.parse_args()




def main():
    args = parse_args()
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = 'cuda'
        if args.seed is not None:
            torch.cuda.manual_seed(args.seed)
    else:
        device = 'cpu'
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg is None:
        message = time_str
    else:
        message = "-" + args.msg
    args.checkpoint = 'checkpoints/' + args.model + message
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    # Model
    printf(f"args: {args}")
    printf('==> Building model..')
    # net = models.__dict__[args.model](num_classes=args.num_classes)
    net = PointBilstm()
   

    criterion = cal_loss
    net = net.to(device)
    # Compute FLOPs and Params
    if device == 'cuda':
        net_for_flops = net.module if isinstance(net, torch.nn.DataParallel) else net
    else:
        net_for_flops = net

    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(
            net_for_flops,
            (3, args.num_points),  # 注意是 (C, N)，没有 batch_size
            as_strings=False,      # 修改为 False，返回数字，方便后续处理
            print_per_layer_stat=False,
            verbose=False
        )
        
        if flops is not None:
            flops = flops / 1e9  # 转成 GFLOPs
        else:
            flops = 0.0
        if params is not None:
            params = params / 1e6  # 转成 M
        else:
            params = 0.0
          
        # lops, params = profile(net, inputs=(torch.randn(1, 3, args.num_points).to(device),))
          
        allocated_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
        reserved_memory = torch.cuda.memory_reserved() / (1024**2)    # MB
        # print(f"FLOPs: {flops / 1e9:.3f} G, Params: {params / 1e6:.3f} M")
        printf('==> Model Profile:')
        printf(f'FLOPs: {flops:.2f} GFLOPs')
        printf(f'Params: {params:.2f} M')
        printf(f'Allocated Memory: {allocated_memory:.2f} MB')
        printf(f'Reserved Memory: {reserved_memory:.2f} MB')
    
    
    # total = sum([param.nelement() for param in net.parameters()])
    printf(net)
    #printf("Number of parameter: %.2fM" % (total/1e6))
    
    # criterion = criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    best_test_acc = 0.  # best test accuracy
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    optimizer_dict = None

    if not os.path.isfile(os.path.join(args.checkpoint, "last_checkpoint.pth")):
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model)
        logger.set_names(["Epoch-Num", 'Learning-Rate',
                          'Train-Loss', 'Train-acc-B', 'Train-acc',
                          'Valid-Loss', 'Valid-acc-B', 'Valid-acc'])
    else:
        printf(f"Resuming last checkpoint from {args.checkpoint}")
        checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        best_train_acc = checkpoint['best_train_acc']
        best_test_acc_avg = checkpoint['best_test_acc_avg']
        best_train_acc_avg = checkpoint['best_train_acc_avg']
        best_test_loss = checkpoint['best_test_loss']
        best_train_loss = checkpoint['best_train_loss']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model, resume=True)
        optimizer_dict = checkpoint['optimizer']

    printf('==> Preparing data..')
    train_loader = DataLoader(ScanObjectNN(partition='training', num_points=args.num_points), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=args.workers,
                             batch_size=args.batch_size, shuffle=False, drop_last=False)
    

    # traindataset = ScanObjectNN(1024)
    
    # testdataset = ScanObjectNN(1024, partition='test')
    # fulldataset = torch.utils.data.ConcatDataset([traindataset, testdataset])
    # train_size = int(0.8 * len(fulldataset))
    # test_size = len(fulldataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(fulldataset, [train_size, test_size],generator=torch.Generator().manual_seed(0))
    
    # train_loader = DataLoader(train_dataset, num_workers=args.workers,
    #                           batch_size=args.batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(test_dataset, num_workers=args.workers,
    #                          batch_size=args.batch_size, shuffle=True, drop_last=False)


    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate / 100, last_epoch=start_epoch - 1)

    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(net, train_loader, optimizer, criterion, device)
        test_out = validate(net, test_loader, criterion, device)
        scheduler.step()
    
        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
        best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss

        save_model(
            net, epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc,
            best_train_acc=best_train_acc,
            best_test_acc_avg=best_test_acc_avg,
            best_train_acc_avg=best_train_acc_avg,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict()
        )
        logger.append([epoch, optimizer.param_groups[0]['lr'],
                       train_out["loss"], train_out["acc_avg"], train_out["acc"],
                       test_out["loss"], test_out["acc_avg"], test_out["acc"]])
        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")
        printf(
            f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}% time:{test_out['time']}s "
            f"inference_time_per_batch:{test_out['inference_time_per_batch']}s "
            f"inference_time_per_sample:{test_out['inference_time_per_sample']}ms "
            f"[best test acc: {best_test_acc}%] \n\n")
        
    logger.close()

    printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
    printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
    printf(f"++  Last Test inference time per batch: {test_out['inference_time_per_batch']}s | per sample: {test_out['inference_time_per_sample']}ms  ++")
    printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
    printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++")
    printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
    printf(f"++++++++" * 5)
    printf('dropout,rotate')

def train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    train_transforms = transforms.Compose(
    [
        PointcloudRandomInputDropout(),
        PointcloudRotate(),
        # PointcloudRotateZ(),
        # PointcloudScale(),
        # PointcloudTranslate(),
    ]
    )
    time_cost = datetime.datetime.now()
    
    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device).squeeze()
        data = train_transforms(data)
        data = data.permute(0, 2, 1)  # so, the input data shape is [batch, 3, 1024]
        # data_shuffle = shuffle_tensor(data) 
        optimizer.zero_grad()
        logits = net(data)
        # logits_shuffle = net(data_shuffle)
        # loss = 0.25*criterion(logits, label) + 0.75*criterion(logits_shuffle, label)
        loss = criterion(logits, label) 
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        total += label.size(0)
        correct += preds.eq(label).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }


def validate(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    inference_time_total = 0.0  # 记录总推理时间

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)

            # 测量推理时间
            torch.cuda.synchronize()  # 确保 GPU 计算完成
            start_time = time.time()
            logits = net(data)
            torch.cuda.synchronize()  # 确保推理完成
            inference_time = time.time() - start_time
            inference_time_total += inference_time

            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    
    # 计算平均推理时间（每批次和每样本）
    num_batches = len(testloader)
    avg_inference_time_per_batch = inference_time_total / num_batches  # 秒/批次
    avg_inference_time_per_sample = (inference_time_total / total) * 1000  # 毫秒/样本

    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost,
        "inference_time_per_batch": float("%.4f" % avg_inference_time_per_batch),  # 新增
        "inference_time_per_sample": float("%.4f" % avg_inference_time_per_sample)  # 新增
    }


if __name__ == '__main__':
    main()
