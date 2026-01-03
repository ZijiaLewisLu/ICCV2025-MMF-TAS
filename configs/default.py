from yacs.config import CfgNode as CN

BASE = CN()

# auxiliary setting
BASE.aux = CN()
BASE.aux.gpu = 1
BASE.aux.mark = "" # for adding addtional note
BASE.aux.runid = 0 # the X-th run of this configuration
BASE.aux.debug = False
BASE.aux.wandb_project = "PGNET"
BASE.aux.use_wandb = False 
BASE.aux.wandb_offline = False
BASE.aux.wandb_job = None
BASE.aux.resume = "max" # "", ckpt_path, "max" (resume latest ckpt of the experiment)
BASE.aux.epoch_resume = None # "", ckpt_path, "max" (resume latest ckpt of the experiment)
BASE.aux.skip_finished = True
BASE.aux.eval_every = 1000
BASE.aux.print_every = 200

# dataset
BASE.dataset = "breakfast"
BASE.feature = "vrl"
BASE.split = "split1"
BASE.nt = 3 # number of support tasks
BASE.nv = 4 # number of support videos per task + 1 query video
BASE.test_set = None  
BASE.temb_path = None

# optimizer
BASE.optimizer = "SGD"
BASE.epoch = 2
BASE.lr = 0.1
BASE.lr_decay = -1
BASE.momentum = 0.009
BASE.weight_decay = 0.000
BASE.clip_grad_norm = 10.0

# Prototype Building Block
BASE.PBB = PBB = CN() 
PBB.dropout = 0.0
PBB.attn_dp = None
PBB.a = "sca" 
PBB.a_nhead = 8
PBB.a_ffdim = 2048
PBB.a_layers = 6
PBB.a_dim = 256
PBB.a_inmap = False

PBB.f = 'cnn'
PBB.f_layers = 5
PBB.f_d = 2
PBB.f_ln = True
PBB.f_dim = 256
PBB.f_ngp = 4

# Matching Block -- it share most of the config with PBB
BASE.MB = MB = PBB.clone() # BASE.MB
MB.a = "sa"
MB.f_share = True 

# unique configs for PBB dynamic graph transformer
PBB.dgt_nhead = 8  
PBB.dgt_ffdim = 2048 
PBB.dgt_layers = 4 
PBB.dgt_dim = 256 
PBB.dgt_inmap = False 
PBB.dgt_etype = 'v,a,c' 

# Loss weights
BASE.Loss = Loss = CN()
Loss.task_w = 1.0 
Loss.simi_w = 1.0 
Loss.smooth_w = 1.0 
Loss.align_w = 0.0 

def get_cfg_defaults():
    return BASE.clone()

RENAME_KEYS = {}



