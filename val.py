import os
import sys
import yaml
import time
import torch
import argparse
import platform
from tqdm import tqdm
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from data import build_dataloader
from modeling.architectures import build_model
from postprocess import build_post_process
from metrics import build_metric
from utils.general import increment_path, check_suffix, colorstr
from utils.torch_utils import select_device

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ch_det_mv3_db_v2.0.yaml', help='model.yaml path')
    parser.add_argument('--weights', type=str, default='pretrained/ch_ptocr_v2_det_infer.pth', help='initial weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-vis', action='store_true', help='save results to *.vis')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_known_args()[0]
    return opt

@torch.no_grad()
def run(
    cfg,
    weights=None,
    batch_size=1,
    device='',
    save_txt=False,
    save_vis=False,
    project=ROOT / 'runs/val',
    name='exp',
    half=True,
    model=None,
    dataloader=None
    ):
    assert weights is not None or model is not None
    training = model is not None
    if training:
        device = next(model.parameters()).device
        assert dataloader.batch_size == 1, "only support batchsize 1 now!"
    else:
        assert batch_size==1, "only support batchsize 1 now!"
        device = select_device(device, batch_size=batch_size)

        # =============================================================
        # load model
        check_suffix(weights, '.pth')  # check weights
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = build_model(cfg['Architecture']).to(device)
        model.load_state_dict(ckpt['state_dict'], strict=True)  # load
        print("Load mode sucessfully!")

        # =============================================================
        # build dataloader
        dataloader, dataset = build_dataloader(cfg['Data'], 'val', batch_size, rank=-1)

    # =============================================================
    # build post process and metric
    post_process_class = build_post_process(cfg['PostProcess'])
    eval_class = build_metric(cfg['Metric'])

    # =============================================================
    # do eval
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()
    model.eval()
    total_frame = 0.0
    total_time = 0.0
    for batch in tqdm(dataloader):
        images = batch[0].to(device)
        start = time.perf_counter()
        preds = model(images)
        batch = [item.numpy() for item in batch]
        total_time += time.perf_counter() - start
        post_result = post_process_class(preds, batch[1])
        eval_class(post_result, batch)
        total_frame += len(images)
    metric = eval_class.get_metric()
    metric['fps'] = total_frame / total_time
    model.float()

    return metric

def main():
    opt = parse_opt()
    opt.project = str(opt.project)
    cfg, weights, batch_size, device, save_txt, save_vis, project, name, half = \
        opt.cfg, opt.weights, opt.batch_size, opt.device, opt.save_txt, opt.save_vis, opt.project, opt.name, opt.half
    if isinstance(cfg, str):
        with open(cfg) as f:
            cfg = yaml.safe_load(f)  # load cfg dict

    # =============================================================
    # make results dir
    save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir_txt = None
    save_dir_vis = None
    if save_txt:
        save_dir_txt = (save_dir / 'labels')
        save_dir_txt.mkdir(parents=True, exist_ok=True)
    if save_vis:
        save_dir_vis = (save_dir / 'vis')
        save_dir_vis.mkdir(parents=True, exist_ok=True)

    # =============================================================
    # run
    metric = run(cfg, weights, batch_size, device, save_dir_txt, save_dir_vis, project, name, half)

    # =============================================================
    # save run settings        
    with open(save_dir / 'cfg.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    with open(save_dir / 'metric.yaml', 'w') as f:
        yaml.safe_dump(metric, f, sort_keys=False)
    print(f"Results saved to {colorstr('bold', save_dir)}")
    
    for k, v in metric.items():
        print('{}:{}'.format(k, v))

if __name__ == '__main__':
    main()