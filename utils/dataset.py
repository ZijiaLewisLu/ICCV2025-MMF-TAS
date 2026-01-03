#!/usr/bin/python3
import numpy as np
import os
import torch
from ..home import get_project_base
from collections import namedtuple
from yacs.config import CfgNode
from .utils import shrink_frame_label, parse_label, load_file
from ..model.basic import CombinedSequence
import json

BASE = get_project_base()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def load_feature(feature_dir, video, feature_type, transpose=False):
    file_name = os.path.join(feature_dir, video + feature_type)
    if feature_type == '.npy':
        feature = np.load(file_name)
    elif feature_type == '.npz':
        feature = np.load(file_name)
        feature = feature['data']

    if transpose:
        feature = feature.T
    if feature.dtype != np.float32:
        feature = feature.astype(np.float32)
    
    return feature

def load_action_mapping(map_fname, sep=" "):
    label2index = dict()
    index2label = dict()
    with open(map_fname, 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            tokens = line.split(sep)
            l = sep.join(tokens[1:])
            i = int(tokens[0])
            label2index[l] = i
            index2label[i] = l

    return label2index, index2label

def class_label_to_segment_label(label, bg=0):
    segs = parse_label(label)
    
    trans = [s.action for s in segs]
    has_bg = (bg in trans)
    if has_bg:
        trans = [s for s in trans if s != bg]
        trans.insert(0, bg)
    
    labels = []
    ct = 1 if has_bg else 0
    for seg in segs:
        if seg.action == bg:
            labels.extend( [bg] * seg.len )
        else:
            labels.extend( [ct] * seg.len )
            ct += 1

    return trans, labels

# -----------------------------------------------------------------------------
# Dataset Configuration
# -----------------------------------------------------------------------------

def get_dataset_config(cfg):
    """
    Centralized place for dataset-specific parameters.
    """
    name = cfg.dataset.lower()
    
    if name == 'crosstask':
        base_dir = 'dataset/crosstask'
        config = {
            'name': name,
            'map_file': os.path.join(base_dir, 'mapping.txt'),
            'task_map_file': os.path.join(base_dir, 'task_mapping.txt'),
            'train_split': os.path.join(base_dir, 'splits', f'{cfg.split}.train'),
            'test_split': os.path.join(base_dir, 'splits', f'{cfg.split}.unseen_test'),
            'bg_class': [0],
        }
        
        if cfg.feature == "vrl":
            config['gt_path'] = os.path.join(base_dir, 'groundTruth')
            config['feat_path'] = os.path.join(base_dir, 'vrl_feature')
            config['feat_ext'] = '.npy'

        def load_task_map(fname):
            mapping = {}
            with open(fname) as f:
                for line in f.read().strip().split('\n'):
                    if not line: continue
                    i, tid = line.split(" ")
                    mapping[tid] = int(i)
            return mapping
            
        def get_task_id(vname, mapping):
            # crosstask vname format: taskid_videoid
            prefix = vname.split('_')[0]
            return mapping[prefix]

        config['load_task_map'] = load_task_map
        config['get_task_id'] = get_task_id
        
        return config

    elif name == 'coin':
        base_dir = 'dataset/coin'
        config = {
            'name': name,
            'map_file': os.path.join(base_dir, 'mapping.txt'),
            'task_map_file': os.path.join(base_dir, 'task_mapping.txt'),
            'train_split': os.path.join(base_dir, 'splits', f'{cfg.split}.train'),
            'test_split': os.path.join(base_dir, 'splits', f'{cfg.split}.unseen_test'),
            'bg_class': [0],
        }

        if cfg.feature == "vrl":
            config['gt_path'] = os.path.join(base_dir, 'groundTruth')
            config['feat_path'] = os.path.join(base_dir, 'vrl_feature')
            config['feat_ext'] = '.npy'

        def load_task_map(fname):
            mapping = {}
            with open(fname) as f:
                for line in f.read().strip().split('\n'):
                    if not line: continue
                    v, i = line.split(" ")
                    mapping[v] = int(i)
            return mapping

        def get_task_id(vname, mapping):
            return mapping[vname]

        config['load_task_map'] = load_task_map
        config['get_task_id'] = get_task_id
        
        return config
        
    else:
        raise ValueError(f"Unknown dataset: {name}")

# -----------------------------------------------------------------------------
# Dataset & DataLoader
# -----------------------------------------------------------------------------

class Dataset(object):
    """
    Generic Dataset class.
    """
    def __init__(self, video_list, config, label2index, task_map, 
                 sr=2 # we downsample feature to 1 fps by default.
                 ):
        self.video_list = video_list
        self.config = config
        self.label2index = label2index
        self.task_map = task_map
        self.sr = sr
        
        self.nclasses = len(label2index)
        self.bg_class = config['bg_class']

        # Load first video to get dimension
        self.data = {}
        if video_list:
            self.data[video_list[0]] = self.load_video(video_list[0])
            self.input_dimension = self.data[video_list[0]][0].shape[1] 
        else:
            self.input_dimension = 0

        # Build task mapping
        self.task2video = {}
        self.vname2task = {}
        
        get_task_id = config['get_task_id']
        for vname in self.video_list:
            task = get_task_id(vname, task_map)
            if task not in self.task2video:
                self.task2video[task] = []

            self.task2video[task].append(vname)
            self.vname2task[vname] = task

    def load_video(self, vname):
        feature = load_feature(self.config['feat_path'], vname, self.config['feat_ext'])
        
        with open(os.path.join(self.config['gt_path'], vname + '.txt')) as f:
            gt_label = [ self.label2index[line] for line in f.read().split('\n')[:-1] ]

        if feature.shape[0] != len(gt_label):
            l = min(feature.shape[0], len(gt_label))
            feature = feature[:l]
            gt_label = gt_label[:l]

        if self.sr > 1:
            feature = feature[::self.sr]
            gt_label_sampled = shrink_frame_label(gt_label, self.sr)
        else:
            gt_label_sampled = gt_label

        bg_class = self.bg_class
        assert len(bg_class) == 1, bg_class
        trans, gt_segment_label = class_label_to_segment_label(gt_label_sampled, bg=bg_class[0])

        return feature, trans, gt_label_sampled, gt_segment_label, gt_label

    def __str__(self):
        string = "< Dataset %d videos, %d tasks, %d feat-size, %d classes >"
        string = string % (len(self.video_list), len(self.task2video), self.input_dimension, self.nclasses)
        return string
    
    def __repr__(self):
        return str(self)

    def get_vnames(self):
        return self.video_list[:]

    def __getitem__(self, video):
        if video not in self.video_list:
            raise ValueError(video)

        if video not in self.data:
            self.data[video] = self.load_video(video)

        return self.data[video]

    def __len__(self):
        return len(self.video_list)


class DataLoader():

    def __init__(self, dataset: Dataset, ntask, nvideo_per_task, shuffle=False, device='cpu', preset=None):

        self.num_video = len(dataset)
        self.dataset = dataset
        self.videos = list(dataset.get_vnames())
        self.tasks = list(dataset.task2video.keys())
        self.shuffle = shuffle
        self.device = device
        self.preset = preset

        # assert shuffle

        self.ntask = ntask
        self.nvideo_per_task = nvideo_per_task
        self.batch_size = ntask * nvideo_per_task
        if self.preset:
            with open(self.preset) as fp:
                self.preset_batch = json.load(fp)
            self.num_batch = len(self.preset_batch)
            assert self.ntask == len(self.preset_batch[0]) - 1
            assert self.nvideo_per_task == len(self.preset_batch[0][1]) + 1
        else:
            self.num_batch = int(np.ceil(self.num_video/self.batch_size))

        # self.selector = list(range(self.num_video))
        self.index = 0
        # if self.shuffle:
            # np.random.shuffle(self.selector)
            # self.selector = self.selector.tolist()

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def sample(self, _list, size):
        if size >= len(_list):
            return _list
        else:
            return np.random.choice(_list, size, replace=False)

    def sample_task_video(self):
        tasks = self.sample(self.tasks, self.ntask)

        videos = []
        video_tasks = []

        for task in tasks: 

            # choose video
            n = self.nvideo_per_task
            vnames = self.sample(self.dataset.task2video[task], n)
            assert len(vnames) >= 1, (task, vnames)

            for vname in vnames:
                videos.append(vname)
                video_tasks.append(task)
        
        return video_tasks, videos


    def __next__(self):
        # if self.index >= self.num_video:
        if self.index >= self.num_batch:
            self.index = 0
            raise StopIteration

        else:
            if self.preset: 
                epi = self.preset_batch[self.index]
                tasks, vnames = [], []
                for i in range(self.ntask):
                    vnames.extend(epi[i+1])
                    t = self.dataset.vname2task[epi[i+1][0]]
                    tasks.extend([t] * (self.nvideo_per_task - 1))
                    if i == 0:
                        vnames.insert(0, epi[0]) # hack
                        tasks.insert(0, t)
            else:
                tasks, vnames = self.sample_task_video()
            
            self.index += 1


            batch_data = []
            for vname in vnames:
                batch_data.append(self.dataset[vname])

            batch_sequence = CombinedSequence.create_from_sequences([x[0] for x in batch_data], pad=0)
            batch_transcript = CombinedSequence.create_from_sequences([x[1] for x in batch_data], torch.long)
            batch_train_class_label = CombinedSequence.create_from_sequences([x[2] for x in batch_data], torch.long)
            batch_train_seg_label = CombinedSequence.create_from_sequences([x[3] for x in batch_data], torch.long)
            # batch_eval_label = [x[4] for x in batch_data]

            if self.device != 'cpu':
                batch_sequence.to(self.device)
                # batch_aset.to(self.device)
                batch_train_class_label.to(self.device)
                batch_train_seg_label.to(self.device)
                batch_transcript.to(self.device)

            return vnames, tasks, batch_sequence, batch_transcript, batch_train_class_label, batch_train_seg_label

# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def create_dataset(cfg: CfgNode):
    
    # 1. Get configuration
    config = get_dataset_config(cfg)
    
    # 2. Load mappings
    label2index, index2label = load_action_mapping(config['map_file'])
    task_map = config['load_task_map'](config['task_map_file'])
    
    # 3. Create Datasets
    unseen_videos = load_file(config['test_split'])
    unseen_test_dataset = Dataset(unseen_videos, config, label2index, task_map, sr=2)

    if cfg.aux.debug:
        dataset = unseen_test_dataset
    else:
        video_list = load_file(config['train_split'])
        dataset = Dataset(video_list, config, label2index, task_map, sr=2)
        
    return dataset, unseen_test_dataset
