#!/usr/bin/python3

import numpy as np
import argparse
import os
from torch import optim
import torch
import wandb
from src.utils.dataset import Dataset, DataLoader, create_dataset
from src.utils.analysis import Video, Checkpoint
from src.home import get_project_base
from src.configs.utils import cfg2flatdict, setup_cfg
from src.utils.train_tools import resume_ckpt, resume_wandb_runid, to_numpy
from src.utils.utils import count_parameters, easy_reduce, to_numpy, save_grad
from src.model.model import PGNet
import json

def add_results(ckpt, vnames, video_saves, loss_dict=None, i=1):
    scalars, vpred = video_saves
    ckpt.scalar_dict_list.append(scalars)
    for b, (p, l) in enumerate(vpred):
        vname = f"{vnames[b]}_{i}"
        video = Video(vname)
        video.pred = to_numpy(p)
        video.gt_label = to_numpy(l)
        if loss_dict is not None:
            video.loss = loss_dict
        ckpt.add_videos([video])


def evaluate(global_step, net, tloader, savedir):
    print("TESTING", "~"*10)

    ckpt = Checkpoint(global_step+1, bg_class=tloader.dataset.bg_class, eval_edit=False)
    net.eval()
    with torch.no_grad():
        for batch_idx, (vnames, tasks, seqs, transcript, class_label, seg_label) in enumerate(tloader):
            loss, loss_dict, video_saves = net.forward_and_loss(tasks, seqs, transcript, class_label, seg_label)
            add_results(ckpt, vnames, video_saves, i=batch_idx)
    ckpt.compute_metrics()
    net.train()

    log_dict = {}
    string = ""
    for k, v in ckpt.metrics.items():
        log_dict[f'test-unseen-metric/{k}'] = v
        string += "%s:%.1f, " % (k, v*100) if k != 'num_segs' else "%s:%.1f, " % (k, v)
    print(string)
    wandb_log(log_dict, step=global_step+1)

    fname = "test_unseen_%d.gz" %  (global_step+1) 
    ckpt.save(os.path.join(savedir, fname))

    return ckpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="cfg_file", nargs="*",
                            help="optional config file", default=[])
    parser.add_argument("--set", dest="set_cfgs",
            help="set config keys", default=None, nargs=argparse.REMAINDER,)

    args = parser.parse_args()
    BASE = get_project_base()

    ### initialize experiment #########################################################
    if len(args.cfg_file) == 1:
        args.cfg_file = args.cfg_file[0].split(' ')
    if len(args.set_cfgs) == 1:
        args.set_cfgs = args.set_cfgs[0].split(' ')
    cfg = setup_cfg(args.cfg_file, args.set_cfgs)
    try:
        torch.cuda.set_device('cuda:%d'%cfg.aux.gpu)
    except Exception as e:
        print(e)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.aux.gpu)

    print('============')
    print(cfg)
    print('============')

    if cfg.aux.debug:
        seed = 1 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.set_printoptions(precision=2, linewidth=160)

    logdir = cfg.aux.logdir
    ckptdir = os.path.join(logdir, 'ckpts')
    savedir = os.path.join(logdir, 'saves')
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    print('Saving log at', logdir)

    
    if cfg.aux.use_wandb:
        wandb_runid = resume_wandb_runid(logdir)
        run = wandb.init(
                    project=cfg.aux.wandb_project, entity="<your-account>",
                    dir=cfg.aux.logdir,
                    group=cfg.aux.exp, id=wandb_runid, resume="allow",
                    config=cfg2flatdict(cfg),
                    reinit=True, save_code=False,
                    mode="offline" if (cfg.aux.wandb_offline or cfg.aux.debug) else "online",
                    notes="log_dir: " + logdir,
                    job_type=cfg.aux.wandb_job,
                    )
        cfg.aux.wandb_id = run.id
        wandb_log = run.log
        print("WANDB ID", run.id)
    else:
        wandb_log = lambda log_dict, step=None: None


    argSaveFile = os.path.join(logdir, 'args.json')
    with open(argSaveFile, 'w') as f:
        json.dump(cfg, f, indent=True)

    ### load dataset #########################################################
    dataset, unseen_test_dataset = create_dataset(cfg)
    device = 'cuda'
    unseen_testloader = DataLoader(unseen_test_dataset, cfg.nt, cfg.nv, device=device, preset=cfg.test_set)
    trainloader = DataLoader(dataset, cfg.nt, cfg.nv, device=device)
    print('Train dataset', dataset)
    print('Test dataset ', unseen_test_dataset)
    
    num_save = len(trainloader) * cfg.epoch // cfg.aux.eval_every
    print(f">>>>>>>> Iteration Per Epoch {len(trainloader)}, Total Iteration {len(trainloader)*cfg.epoch}")
    print(f">>>>>>>> Num Saves {num_save}")

    ### create network #########################################################
    net = PGNet(cfg, dataset.input_dimension) 
    global_step, ckpt_file = resume_ckpt(cfg, logdir)
    if ckpt_file is not None:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        net.load_state_dict(ckpt, strict=False)
    net.cuda()

    print(net)
    print(f'Total Number of Parameters -- {count_parameters(net)/1e6} M')

    ### create optimizer #########################################################
    if cfg.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(),
                            lr=cfg.lr,
                            momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)

    elif cfg.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(),
                               lr=cfg.lr,
                               weight_decay=cfg.weight_decay)

    ### start training #########################################################
    start_epoch = global_step // len(trainloader)
    ckpt = Checkpoint(-1, bg_class=dataset.bg_class, eval_edit=False)
    total_iterations = cfg.epoch * len(trainloader)
    print(f'Start Training from Epoch {start_epoch}...')

    for eidx in range(start_epoch, cfg.epoch):

        for batch_idx, (vnames, tasks, seqs, transcript, class_label, seg_label) in enumerate(trainloader):

            optimizer.zero_grad()
            loss, loss_dict, video_saves = net.forward_and_loss(tasks, seqs, transcript, class_label, seg_label)
            loss.backward()
            if cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_grad_norm)
            optimizer.step()

            add_results(ckpt, vnames, video_saves, loss_dict=loss_dict, i=global_step)

            # print some progress information
            if (global_step+1) % cfg.aux.print_every == 0 or global_step == 0:

                ckpt.compute_metrics()
                ckpt.average_losses()

                log_dict = {}
                string = "Iter%d[GPU%d]: " % (global_step+1, cfg.aux.gpu)
                _L = len(string)
                for k, v in ckpt.loss.items():
                    log_dict[f"train-loss/{k}"] = v
                    string += f"{k}:{v:.3f}, "
                print(string)

                string = " " * _L 
                for k, v in ckpt.metrics.items():
                    log_dict[f'train-metric/{k}'] = v
                    string += "%s:%.1f, " % (k, v*100) if k != 'num_segs' else "%s:%.1f, " % (k, v)
                print(string)

                wandb_log(log_dict, step=global_step+1)
                ckpt = Checkpoint(-1, bg_class=dataset.bg_class, eval_edit=False)

            # test and save model every x iterations
            if global_step != 0 and (global_step+1) % cfg.aux.eval_every == 0:
                evaluate(global_step, net, unseen_testloader, savedir)

                print('save snapshot ' + str(global_step+1))
                network_file = ckptdir + '/network.iter-' + str(global_step+1) + '.net'
                net.save_model(network_file)

                print(logdir)
                print(f"Progress: {eidx}/{cfg.epoch} - {100*float(global_step)/total_iterations:.2f}%")
                print()

            global_step += 1

        if cfg.lr_decay > 0 and ( eidx + 1 ) % cfg.lr_decay == 0:
            for g in optimizer.param_groups:
                g['lr'] = cfg.lr * 0.1
            print('------------------------------------Update Learning rate--------------------------------')

    print("Finish Stamp")
    open(os.path.join(logdir, "FINISH_PROOF"), "w").close()
    if cfg.aux.use_wandb:
        run.finish()
