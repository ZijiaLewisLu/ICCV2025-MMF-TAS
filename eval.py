from src.model.model import PGNet
from src.utils.dataset import create_dataset, DataLoader
from src.utils import utils
from pathlib import Path
import torch 
import numpy as np
from src.utils.analysis import Checkpoint, Video
from tqdm import tqdm
from yacs.config import CfgNode
import json





def load_model(path, i, size=512):
    exp = Path(path) 
    with open(exp / 'args.json') as fp:
        cfg = json.load(fp)
    cfg = CfgNode(cfg)

    net = PGNet(cfg, size)
    net = net.cuda()

    ckpt = exp / f'network.iter-{i}.net'
    ckpt = torch.load(ckpt, map_location='cpu')
    net.load_state_dict(ckpt)
    net = net.cuda().eval()
    return cfg, net


def add_results(ckpt, vnames, video_saves, i=1):
    scalars, vpred = video_saves
    for b, (p, l) in enumerate(vpred):
        vname = f"{vnames[b]}_{i}"
        video = Video(vname)
        video.pred = utils.to_numpy(p)
        video.gt_label = utils.to_numpy(l)
        ckpt.add_videos([video])

model_dict = {
    'coin_3t3v': (
        'model_weights/coin_3t3v',
        117000,
    ),
    'coin_5t3v': (
        'model_weights/coin_5t3v',
        180000,
    ),
    'crosstask_3t3v': (
        'model_weights/crosstask_3t3v',
        170000,
    ),
    'crosstask_5t3v': (
        'model_weights/crosstask_5t3v',
        130000,
    ),
}
N = 4 # average performance over N runs
mode = 'fewshot_full_label' # fewshot - have support videos and full labels
# mode = 'zeroshot' # zeroshot - no support videos, only textual action names
# mode = 'fewshot_no_label' # fewshot - have support videos but no labels
# mode = 'fewshot_weak_label' # fewshot - have support videos. only one frame label per segment.

# Note: We corrected a error in zeroshot evaluation, 
#       so the results of zeroshot, fewshot_no_label, fewshot_weak_label are slightly different from those reported in the paper.
#       The results of fewshot_full_label remain unchanged.
#       The results of fewshot_weak_label can vary due to random sampling of weak labels.

# We use 800 evaluation samples for Crosstask and 1000 evaluation samples for COIN.

np.set_printoptions(precision=1)
_DEFAULT_METRICS = ['edit', 'F1@0.10#', 'F1@0.25#', "F1@0.50#", 'Mof#']
def simple_print(d):
    for key in _DEFAULT_METRICS:
        s = d[key] * 100
        print(f"{s:.1f}", end='; ')
    print()

for model_name, (base_path, iteration) in model_dict.items():
    print(f"Evaluating model: {model_name}")
    all_metrics = []
    for i in range(1, N+1):
        exp_path = f'{base_path}/{i}/'
        cfg, net = load_model(exp_path, iteration)
        dataset, unseen_test = create_dataset(cfg)

        loader = DataLoader(unseen_test, cfg.nt, cfg.nv, device='cuda', preset=cfg.test_set) 
        ckpt = Checkpoint(-1, [dataset.bg_class])
        for bidx, (vnames, tasks, seqs, transcript, class_label, seg_label) in enumerate(tqdm(loader)):
            loss, loss_dict, video_saves = net.inference(tasks, seqs.clone(), transcript, class_label, seg_label, 
                                                        mode=mode)
            add_results(ckpt, vnames, video_saves, i=bidx)
        ckpt.compute_metrics()
        all_metrics.append(ckpt.metrics)

    all_metrics = utils.easy_reduce(all_metrics)
    simple_print(all_metrics)

