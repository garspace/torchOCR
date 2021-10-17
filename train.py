import os
import sys
import time
import math
import yaml
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP

import val
from losses import build_loss
from data import build_dataloader
from modeling.architectures import build_model

from utils import build_optimizer, build_scheduler
from utils.loggers import Loggers
from utils.callbacks import Callbacks
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_git_status, check_img_size, check_requirements, \
    check_file, check_yaml, check_suffix, print_args, print_mutation, set_logging, one_cycle, colorstr, methods
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def train(cfg,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          callbacks
          ):

    # ===============================================================================
    # Init
    save_dir, epochs, batch_size, weights, evolve, cfg, resume, noval, val_period, nosave, workers, freeze, fp16 = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.evolve, opt.cfg, \
        opt.resume, opt.noval, opt.val_period, opt.nosave, opt.workers, opt.freeze, opt.fp16

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pth', w / 'best.pth'

    # Config
    if isinstance(cfg, str):
        with open(cfg) as f:
            cfg = yaml.safe_load(f)  # load cfg dict
    hyp = cfg['Hyp']
    # LOGGER.info(colorstr('config: ') + ', '.join(f'{k}={v}' for k, v in cfg.items()))

    # Save run settings
    with open(save_dir / 'cfg.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None

    # Loggers
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, cfg, LOGGER)  # loggers instance
        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)

    # ===============================================================================
    # Build model
    check_suffix(weights, '.pth')  # check weights
    pretrained = weights.endswith('.pth')
    if pretrained:
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = build_model(cfg['Architecture']).to(device)
        csd = intersect_dicts(ckpt['state_dict'], model.state_dict(), exclude=[])  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = build_model(cfg['Architecture']).to(device)

    # ===============================================================================
    # Build optimizer and scheduler and ema
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")
    optimizer = build_optimizer(model, hyp, opt.adam, LOGGER)
    scheduler, lf = build_scheduler(optimizer, epochs, hyp, linear_lr=opt.linear_lr)
    ema = ModelEMA(model) if RANK in [-1, 0] else None
    
    # ===============================================================================
    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # ===============================================================================
    # Build dataloader
    train_loader, trainset = build_dataloader(cfg['Data'], 'train', batch_size // WORLD_SIZE, rank=RANK)
    nb = len(train_loader)  # number of batches

    # Process 0
    if RANK in [-1, 0]:
        val_loader, valset = build_dataloader(cfg['Data'], 'val', 1, rank=-1)
        if not resume:
            if plots:
                pass
                # need to do
                # plot_labels(labels, names, save_dir)
            # model.half().float()  # pre-reduce precision, destroy hmean 
        callbacks.run('on_pretrain_routine_end')
    
    # DDP mode
    print(f'RANK:{RANK}')
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # ===============================================================================
    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    last_opt_step = -1
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda and fp16)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = build_loss(cfg['Loss'])  # init loss class
    LOGGER.info(f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    # return train_loader, trainset, val_loader, valset

    for epoch in range(start_epoch, epochs):
        # start epoch ------------------------------------------------------------------
        model.train()

        mloss = None
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)

        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()

        for i, data in pbar:
            # start batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            data = [d.to(device) for d in data]
            imgs = data[0]

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Forward
            with amp.autocast(enabled=cuda and fp16):
                pred = model(imgs)  # forward
                loss = compute_loss(pred, data)  # loss scaled by batch_size
                loss_total = loss.pop('loss')
                loss_items = loss
                if RANK != -1:
                    loss_total *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                    
            # Backward
            scaler.scale(loss_total).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni
            # Log
            if RANK in [-1, 0]:
                if mloss is None:
                    loss_num = len(loss_items)
                    loss_keys = sorted(list(loss_items.keys()))
                    mloss = [0 for _ in range(loss_num)]
                    LOGGER.info(('\n' + '%10s' * (3+loss_num)) % ('Epoch', 'gpu_mem', *loss_keys, 'img_size'))
                loss_values = [loss_items[k].item() for k in loss_keys]
                mloss = [(mloss[l] * i + loss_values[l]) / (i + 1) for l in range(loss_num)]
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * (loss_num+1)) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, imgs.shape[-1]))
                # callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, opt.sync_bn)
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = {f'x/lr{j}':x['lr'] for j,x in enumerate(optimizer.param_groups)} # for loggers
        scheduler.step()

        # ===============================================================================
        # Do val and save model
        if RANK in [-1, 0]:
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            metric = {}
            if (not noval and epoch%val_period==0) or final_epoch:
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).state_dict(),
                        'ema': deepcopy(ema.ema).state_dict(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict()}
                torch.save(ckpt, last)
                metric = val.run(cfg, model=ema.ema, dataloader=val_loader, half=cuda and fp16)
                print(f'Val results: {metric}')
                if metric['main'] > best_fitness:
                    best_fitness = metric['main']
                    torch.save(ckpt, best)
                del ckpt                
                
            # tensorboard log
            metric = {f'metric/{k}':metric[k] for k in metric}
            log_vals = {f'train/{l}':mloss[j] for j,l in enumerate(loss_keys)}
            log_vals.update(lr)
            log_vals.update(metric)
            callbacks.run('on_fit_epoch_end', log_vals, epoch)
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        if not evolve:
            # Strip optimizers
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
        # callbacks.run('on_train_end', last, best, plots, epoch)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='pretrained/ch_ptocr_v2_det_infer.pth', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='configs/ch_det_mv3_db_v2.0.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='configs/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--val_period', type=int, default=1, help='Do val after every "val_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--fp16', action='store_true', help='fp16 training')
    opt = parser.parse_known_args()[0]
    return opt

def main():
    opt = parse_opt()
    callbacks = Callbacks()

    # Check
    set_logging(RANK)
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)
        check_requirements(exclude=['thop'])

    # Resume
    if opt.resume and not opt.evolve:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.cfg = check_file(opt.data), check_yaml(opt.cfg)  # check YAMLs
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = 'runs/evolve'
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        from datetime import timedelta
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.cfg, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            _ = LOGGER.info('Destroying process group... ', end=''), dist.destroy_process_group(), LOGGER.info('Done.')

if __name__ == '__main__':
    main()