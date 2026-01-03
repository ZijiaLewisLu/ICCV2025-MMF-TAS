import numpy as np
from collections import Counter
import os
from tqdm import tqdm
import shutil

class Video():
    
    def __init__(self, vname=''):
        self.vname = vname
    
    def __str__(self):
        return "< Video %s >" % self.vname

    def __repr__(self):
        return "< Video %s >" % self.vname


class Segment():
    def __init__(self, action, start, end):
        assert start >= 0
        self.action = action
        self.start = start
        self.end = end
        self.len = end - start + 1
    
    def __repr__(self):
        return "<%r %d-%d>" % (self.action, self.start, self.end)
    
    def intersect(self, s2):
        s = max([self.start, s2.start])
        e = min([self.end, s2.end])
        return max(0, e-s+1)

    def union(self, s2):
        s = min([self.start, s2.start])
        e = max([self.end, s2.end])
        return e-s+1
    
# def parse_label(label_list):
#     """
#     convert framewise labels into segments
#     """
#     current = label_list[0]
#     start = 0
#     anno = []
#     for i, l in enumerate(label_list):
#         if l == current:
#             pass
#         else:
#             anno.append(Segment(current, start, i-1))
#             current = l
#             start = i
#     anno.append(Segment(current,start, len(label_list)-1))
    
#     return anno

def parse_label(label: np.array):
    if not isinstance(label, np.ndarray):
        label = np.array(label)

    loc = label[:-1] != label[1:]
    loc = np.where(loc)[0]
    segs = []
    
    if len(loc) == 0:
        return [ Segment(label[0], 0, len(label)-1) ]
        
    for i, l in enumerate(loc):
        if i == 0:
            start = 0
            end = l
        else:
            start = loc[i-1]+1
            end = l
        
        seg = Segment(label[start], start, end)
        segs.append(seg)
        
    segs.append(Segment(label[loc[-1]+1], loc[-1]+1, len(label)-1))
    return segs

def class_label_to_segment_label(label):
    segs = parse_label(label)
    labels = []
    for i, seg in segs:
        labels.extend( [i] * seg.len )

    labels = np.array(labels)
    return labels


#############################################
def expand_frame_label(label, target_len: int):
    if len(label) == target_len:
        return label

    import torch
    is_numpy = isinstance(label, np.ndarray)
    if is_numpy:
        label = torch.from_numpy(label).float()
    if isinstance(label, list):
        label = torch.FloatTensor(label)

    label = label.view([1, 1, -1])
    resized = torch.nn.functional.interpolate(
        label, size=target_len, mode="nearest"
    ).view(-1)
    resized = resized.long()
    
    if is_numpy:
        resized = resized.detach().numpy()

    return resized

def shrink_frame_label(label: list, clip_len: int) -> list:
    num_clip = ((len(label) - 1) // clip_len) + 1
    num_clip = int(num_clip)
    new_label = []
    for i in range(num_clip):
        s = int(i * clip_len)
        e = int(s + clip_len)
        l = label[s:e]
        ct = Counter(l)
        l = ct.most_common()[0][0]
        new_label.append(l)

    return new_label

def interpolate_matrix(matrix, target_size, axis=-1):
    from scipy.interpolate import interp1d
    s = matrix.shape[axis]
    heights = np.linspace(0, 1, s)
    interp_function = interp1d(heights, matrix, axis=axis)

    new_heights = np.linspace(0, 1, target_size)
    new_matrix = interp_function(new_heights)
    return new_matrix


def easy_reduce(scores, mode="mean", skip_nan=False):
    assert isinstance(scores, list), type(scores)

    if isinstance(scores[0], list):
        average = []
        L = len(scores[0])
        for i in range(L):
            average.append( easy_reduce([s[i] for s in scores ], mode=mode, skip_nan=skip_nan) )

    elif isinstance(scores[0], np.ndarray):
        assert len(scores[0].shape) == 1
        stack = np.stack(scores, axis=0)
        average = stack.mean(0)

    elif isinstance(scores[0], tuple):
        average = []
        L = len(scores[0])
        for i in range(L):
            average.append( easy_reduce([s[i] for s in scores ], mode=mode, skip_nan=skip_nan) )
        average = tuple(average)

    elif isinstance(scores[0], dict):
        average = {}
        for k in scores[0]:
            average[k] = easy_reduce([s[k] for s in scores], mode=mode, skip_nan=skip_nan)

    elif isinstance(scores[0], float) or isinstance(scores[0], int) or isinstance(scores[0], np.float32): # TODO - improve
        if skip_nan:
            scores = [ x for x in scores if not np.isnan(x) ]

        if mode == "mean":
            average = np.mean(scores)
        elif mode == "max":
            average = np.max(scores)
        elif mode == "median":
            average = np.median(scores)
        elif mode == "std":
            average = np.std(scores)
    else:
        raise TypeError("Unsupport Data Type %s" % type(scores[0]) )

    return average


###################################
###################################

def save_grad(var):
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    # return hook


def to_numpy(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    else:
        return x

def load_file(fname):
    with open(fname) as fp:
        content = fp.read().split('\n')[:-1]
    return content

def class_label_to_segment_label(label):
    segment_label = np.zeros_like(label)
    current = label[0]
    aid = 0
    for i, l in enumerate(label):
        if l == current:
            pass
        else:
            current = l
            aid += 1
        segment_label[i] = aid
    
    return segment_label

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def minmax_norm(x):
    m1, m2 = x.min(), x.max()
    return (x - m1) / (m2 - m1 + 1e-5)

