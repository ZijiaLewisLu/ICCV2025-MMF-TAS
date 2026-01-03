import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import copy
import einops
import numpy as np

PAD_ID = 99988

class CombinedSequence():
    """
    a helper class to store different variables of a sequence 
    e.g., feature, mask, task/video/action ID, ...
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        
    @classmethod
    def create_from_sequences(cls, sequences, dtype=torch.float, pad=PAD_ID, L=None):
        for i, seq in enumerate(sequences):
            if isinstance(seq, torch.Tensor):
                continue
            if dtype == torch.float:
                sequences[i] = torch.FloatTensor(seq)
            elif dtype == torch.long:
                sequences[i] = torch.LongTensor(seq)

        B = len(sequences)
        lens = [ s.shape[0] for s in sequences ]
        if L is None:
            L = max(lens)

        if len(sequences[0].shape) == 2:
            feat = torch.zeros([B, L, sequences[0].shape[1]], dtype=dtype) + pad
            mask = torch.zeros([B, L, 1])
        elif len(sequences[0].shape) == 1:
            feat = torch.zeros([B, L], dtype=dtype) + pad
            mask = torch.zeros([B, L])

        feat = feat.to(sequences[0].device)
        mask = mask.to(sequences[0].device)
        lens = torch.LongTensor(lens).to(sequences[0].device)

        for i, seq in enumerate(sequences):
            t = lens[i]
            feat[i, :t] = seq
            mask[i, :t] = 1
        
        return CombinedSequence(sequences=feat, masks=mask, lens=lens)

    def to(self, device):
        self.sequences =self.sequences.to(device)
        self.masks =self.masks.to(device)
        self.lens  =self.lens.to(device)

    def clone(self, deep=False):
        new = CombinedSequence()
        for k, v in vars(self).items():
            if deep:
                v = v.clone()
            setattr(new, k, v)
        return new


def torch_class_label_to_segment_label(label: CombinedSequence, bg_class=0):
    segment_label = label.clone(deep=True)
    segment_label.sequences[:] = PAD_ID

    N = label.sequences.shape[0]
    all_transcripts = []
    for i in range(N):
        T = label.lens[i].item()
        bg_loc = torch.where(label.sequences[i, :T] == bg_class)[0]
        segment_label.sequences[i, bg_loc] = 0

        aid = 1
        trans = [0]
        for t in range(T):
            l = label.sequences[i, t]
            if l == bg_class:
                continue
            if len(trans) == 1:
                trans.append(l)
            if l == trans[-1]:
                pass
            else:
                aid += 1
                trans.append(l)
            segment_label.sequences[i, t] = aid

        trans = torch.LongTensor(trans) #.to(label.device)
        all_transcripts.append(trans)
        
    tlens = [ len(x) for x in all_transcripts ]
    tmatrix = torch.zeros([N, max(tlens)]) + PAD_ID
    tmask = torch.zeros([N, max(tlens)])
    for i in range(N):
        tmatrix[i, :tlens[i]] = all_transcripts[i]
        tmask[i, :tlens[i]] = 1
    
    transcript = CombinedSequence()
    transcript.sequences = tmatrix.long().to(label.sequences.device)
    transcript.masks = tmask.to(label.sequences.device)
    transcript.lens = torch.LongTensor(tlens).to(label.sequences.device)

    return transcript, segment_label

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, nchannels, dropout=0.5, ngroup=1):
        super(DilatedResidualLayer, self).__init__()
        self.dilation = dilation
        self.nchannels = nchannels
        self.dropout_rate = dropout

        self.conv_dilated = nn.Conv1d(nchannels, nchannels, 3, padding=dilation, dilation=dilation, groups=ngroup)
        self.conv_1x1 = nn.Conv1d(nchannels, nchannels, 1)
        self.dropout = nn.Dropout(dropout)

    def __str__(self):
        return f"DilatedResidualLayer(Conv(d={self.dilation},h={self.nchannels}), 1x1(h={self.nchannels}), Dropout={self.dropout_rate}, ln={self.use_layernorm})"

    def __repr__(self):
        return str(self)

    def forward(self, x, mask=None):
        """
        x: B, D, T
        """
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        if mask is not None:
            # x = (x + out) * mask[:, 0:1, :]
            x = (x + out) * mask
        else:
            x = x + out

        return x

class TCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout=0.5, dilation_factor=2, ln=True, ngroup=1, in_map=False):
        super(TCN, self).__init__()
        if in_map:
            self.conv_1x1 = nn.Conv1d(in_dim, hid_dim, 1)
        else:
            assert in_dim == hid_dim

        self.layers = nn.ModuleList([DilatedResidualLayer(dilation_factor ** i, hid_dim, dropout, ngroup=ngroup) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(hid_dim, out_dim, 1)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.in_map = in_map
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.dilation_factor = dilation_factor

        self.ln = None
        if ln:
            self.ln = nn.LayerNorm(out_dim)

        self.string = f"TCN(h:{in_dim}->{hid_dim}x{num_layers}->{out_dim}, d={dilation_factor}, ng={ngroup}, dropout={dropout}, in_map={in_map})"

    def __str__(self):
        return self.string 

    def __repr__(self):
        return str(self)

    def forward(self, x, mask=None):
        """
        x: T, B, H
        mask: T, B, 1
        """
        x = einops.rearrange(x, 't b h -> b h t')
        _mask = einops.rearrange(mask, 't b h -> b h t')

        if self.in_map:
            x = self.conv_1x1(x)

        for layer in self.layers:
            x = layer(x, _mask)

        out = self.conv_out(x) 
        out = einops.rearrange(out, 'b h t -> t b h')

        if self.ln:
            out = self.ln(out)

        if mask is not None:
            out = out * mask

            if self.training and out.requires_grad:
                _mask = (1-mask).bool()
                out.register_hook(lambda grad: grad.masked_fill(_mask, 0))

        self.output = out

        return self.output

class ClassEmbedding(nn.Module):

    @classmethod
    def load_and_create(cls, temb_path, psample='subset'):
        temb_paths = [ x.strip() for x in temb_path.split(';') ]
        tembs = [ torch.load(x).float() for x in temb_paths ]
        cemb = torch.cat(tembs, dim=0) # nprompt, nclass, in_dim
        return cls(cemb, psample=psample)

    def __init__(self, embedding, psample='all'):
        """
        embedding: nprompt, nclass, ndim
        """
        assert psample in ['all', 'subset']
        super().__init__()
        D = embedding.shape[2]
        self.psample = psample
        self.fg_action_embedding = embedding
        self.context_embedding = torch.nn.Parameter(torch.randn(1, 1, D))
        self.bg_embedding = torch.nn.Parameter(torch.randn(1, D))


    def init_embedding(self):
        self.fg_action_embedding = self.fg_action_embedding.to(self.bg_embedding.device) # HACK

        if self.psample == 'all' or ((not self.training) and (self.psample=='subset')):
            # average the embeddings from all prompts
            self.batch_fg_action_embedding = self.fg_action_embedding.mean(0)
        elif (self.training and self.psample == 'subset'):
            # randomly sample a subset of prompts to use during each training step
            P, C, D = self.fg_action_embedding.shape
            fg_action_embedding = []
            num_p = np.random.choice(P, C) + 1
            for i in range(C):
                p = num_p[i]
                idx = np.random.choice(P, p, replace=False)
                emb = self.fg_action_embedding[idx, i].mean(0)
                fg_action_embedding.append(emb)
            self.batch_fg_action_embedding = torch.stack(fg_action_embedding, dim=0)

    def forward(self, transcripts, add_context_emb=False):
        B, T = transcripts.shape
        class_embedding = torch.cat([self.bg_embedding, self.batch_fg_action_embedding], dim=0) # C, E
        class_embedding = class_embedding[transcripts] # B, T, H
        if add_context_emb:
            context_emb = self.context_embedding.expand([B, 1, class_embedding.shape[-1]])
            class_embedding = torch.cat([context_emb, class_embedding], dim=1)
        class_embedding = class_embedding / torch.clamp( torch.norm(class_embedding, dim=-1, keepdim=True), min=1e-5 )
        return class_embedding





def add_positional_encoding(tensor, pos):
    if pos is None:
        return tensor
    else:
        d = pos.size(-1)
        tensor = tensor.clone()
        tensor[:, :, :d] = tensor[:, :, :d] + pos
        return tensor


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, max_len=5000, empty=False):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.empty = empty
        self.__compute_pe__(d_model, max_len)


    def __compute_pe__(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)

        if not self.empty:
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            # pe = pe.unsqueeze(0).transpose(0, 1)

        pe = pe.unsqueeze(1) 
        self.register_buffer('pe', pe)
    
    def __str__(self):
        if self.empty:
            return f"PositionalEncoding(EMPTY)"
        else:
            return f"PositionalEncoding(Dim={self.d_model}, MaxLen={self.max_len})"

    def __repr__(self):
        return str(self)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x.dim0 = sequence length
            output: [sequence length, batch_size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        if x.size(0) > self.pe.shape[0]: 
            self.__compute_pe__(self.d_model, x.size(0)+10)
            self.pe = self.pe.to(x.device)

        return self.pe[:x.size(0), :]


def create_padding_mask(seq_lens: torch.Tensor):
    device = seq_lens.device
    N = len(seq_lens)
    maxT = torch.max(seq_lens)

    padding_mask = torch.BoolTensor([[False] * maxT for i in range(N)]).to(device) # N, Tf
    for i in range(N):
        padding_mask[i, seq_lens[i]:] = True
    return padding_mask

def padding_mask_to_attn_mask(padding_mask, vlen):
    N, klen = padding_mask.shape
    attn_mask = torch.zeros_like(padding_mask, dtype=torch.float)
    attn_mask.masked_fill_(padding_mask, float("-inf"))
    attn_mask = attn_mask.unsqueeze(1)
    attn_mask = attn_mask.expand([N, vlen, klen])
    return attn_mask

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class SALayer(nn.Module):

    def __init__(self, q_dim, nhead, dim_feedforward=2048, kv_dim=None,
                 dropout=0.1, attn_dropout=0.1,
                 activation="relu", vpos=False):
        super().__init__()
        assert not vpos

        kv_dim = q_dim if kv_dim is None else kv_dim
        self.multihead_attn = nn.MultiheadAttention(q_dim, nhead, kdim=kv_dim, vdim=kv_dim, dropout=attn_dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(q_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, q_dim)

        self.norm1 = nn.LayerNorm(q_dim)
        self.norm2 = nn.LayerNorm(q_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.q_dim = q_dim
        self.kv_dim=kv_dim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        self.use_vpos = vpos
        self.dropout_rate = (dropout, attn_dropout)

    def __str__(self) -> str:
        return f"SALayer( q({self.q_dim})xkv({self.kv_dim})->{self.q_dim}, head:{self.nhead}, ffdim:{self.dim_feedforward}, dropout:{self.dropout_rate}, vpos:{self.use_vpos} )"
    
    def __repr__(self):
        return str(self)

    # def with_pos_embed(self, tensor, pos: Optional[Tensor]):
    #     if pos is None:
    #         return tensor
    #     else:
    #         d = pos.size(-1)
    #         tensor = tensor.clone()
    #         tensor[:, :, :d] = tensor[:, :, :d] + pos
    #         return tensor
        # return tensor if pos is None else tensor + pos

    def forward(self, tgt,
            attn_mask: Optional[Tensor] = None):
        """
        tgt : query
        memory: key and value
        """
        # tgt = tgt.sequences

        tgt2, self.attn = self.multihead_attn(tgt, tgt, tgt, attn_mask=attn_mask, need_weights=False) # attn: nhead, batch, q, k

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt 

class SCALayer(nn.Module):

    def __init__(self, action_dim, frame_dim, nhead, dim_feedforward=2048, dropout=0.1, attn_dropout=0.1,
                 activation="relu",
                 sa_value_w_pos=False, ca_value_w_pos=False,):
        """
        Self-Attention + Cross-Attention Module
        """
        super().__init__()

        assert action_dim == frame_dim, (action_dim, frame_dim)

        self.self_attn = nn.MultiheadAttention(action_dim, nhead, dropout=attn_dropout)
        self.multihead_attn = nn.MultiheadAttention(action_dim, nhead, kdim=frame_dim, vdim=frame_dim, dropout=attn_dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(action_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, action_dim)

        self.norm1 = nn.LayerNorm(action_dim)
        self.norm2 = nn.LayerNorm(action_dim)
        self.norm3 = nn.LayerNorm(action_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.sa_value_w_pos = sa_value_w_pos
        self.ca_value_w_pos = ca_value_w_pos

        
        self.string = f"SCALayerSpecial( adim:{action_dim}, fdim:{frame_dim}, head:{nhead}, ffdim:{dim_feedforward}, dropout:{(dropout, attn_dropout)}, svpos:{sa_value_w_pos}, cvpos:{ca_value_w_pos} )"

    def __str__(self) -> str:
        return self.string
    
    def __repr__(self):
        return str(self)

    def forward(self, tgt, memory1,
                tgt_mask, memory_mask, 
                ca_attn_mask = None, pos = None, query_pos = None,
                     ):
        # self attention
        # tgt_mask = (1 - tgt.masks[..., 0]).bool()
        # tgt = tgt.sequences
        q = k = add_positional_encoding(tgt, query_pos)
        v = tgt if not self.sa_value_w_pos else q
        tgt2, _ = self.self_attn(q, k, v, key_padding_mask=tgt_mask, need_weights=False)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        # memory_mask = (1 - memory1.masks[..., 0]).bool()
        # memory1 = memory1.sequences
        query = add_positional_encoding(tgt, query_pos)
        key1 = add_positional_encoding(memory1, pos)
        value = memory1 if not self.ca_value_w_pos else key1

        tgt2, _ = self.multihead_attn(query, key1, value, key_padding_mask=memory_mask, attn_mask=ca_attn_mask, need_weights=False)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class BasicTransformer(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, layer, num_layers, 
                 norm=False, in_map=False, out_map=True):
        super().__init__()
        if in_map:
            self.in_linear = nn.Linear(in_dim, hid_dim)
        else:
            assert in_dim == hid_dim
            self.in_linear = nn.Identity()
            
        self.num_layers = num_layers
        self.layers = _get_clones(layer, num_layers)

        if out_map:
            self.out_linear = nn.Linear(hid_dim, out_dim)
        else:
            assert hid_dim == out_dim
            self.out_linear = nn.Identity()

        if norm:
            self.norm = torch.nn.LayerNorm(out_dim)
        else:
            self.norm = nn.Identity()

        self.string = f"({type(layer)})(h:{in_dim}->{hid_dim}x{num_layers}->{out_dim}, in_map={in_map}\n"

    def __str__(self) -> str:
        return self.string
    
    def __repr__(self):
        return str(self)

    def forward(self, *args, **kwargs):
        args = list(args)

        # if self.in_map:
        args[0] = self.in_linear(args[0])

        # self.intermediate = []
        for layer in self.layers:
            out = layer(*args, **kwargs)
            args[0] = out
            # self.intermediate.append(output)

        # output.sequences = self.out_linear(output.sequences)
        args[0] = self.out_linear(args[0])

        # norm
        args[0] = self.norm(args[0])
        
        return args[0]


################################################################################
################################################################################

class DynamicMultiheadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_k_input_channels: int = None,
        num_v_input_channels: int = None,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels: Optional[int] = None,
        # max_heads_parallel: Optional[int] = None,
        # causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        n_edge_mask = 3,
        # share_weight=False,
    ):
        super().__init__()

        if num_k_input_channels is None:
            num_k_input_channels = num_q_input_channels

        if num_v_input_channels is None:
            num_v_input_channels = num_q_input_channels

        if num_qk_channels is None:
            num_qk_channels = num_q_input_channels

        # self.num_head_per_group = num_head_per_group
        # self.num_head_group = num_head_group
        # num_heads = num_head_group * num_head_per_group
        num_qk_channels_per_head = num_qk_channels // num_heads
        num_qk_channels = num_qk_channels_per_head * num_heads

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        # if num_qk_channels % num_heads != 0:
        #     raise ValueError("num_qk_channels must be divisible by num_heads")

        # if num_v_channels % num_heads != 0:
        #     raise ValueError("num_v_channels must be divisible by num_heads")

        self.dp_scale = num_qk_channels_per_head**(-0.5)
        self.num_heads = num_heads
        self.num_qk_channels = num_qk_channels
        self.num_v_channels = num_v_channels
        # self.causal_attention = causal_attention

        # if max_heads_parallel is None:
        #     self.max_heads_parallel = num_heads
        # else:
        #     self.max_heads_parallel = max_heads_parallel

        self.q_proj = nn.Linear(num_q_input_channels, num_qk_channels, bias=qkv_bias)
        # self.q2_proj = nn.Linear(num_q_input_channels, num_qk_channels, bias=qkv_bias)
        self.k_proj = nn.Linear(num_k_input_channels, num_qk_channels, bias=qkv_bias)
        # self.k2_proj = nn.Linear(num_k_input_channels, num_qk_channels, bias=qkv_bias)
        self.v_proj = nn.Linear(num_v_input_channels, num_v_channels, bias=qkv_bias)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels, bias=out_bias)
        self.dropout = dropout

        # self.mask_linear = nn.Linear(num_qk_channels_per_head, n_edge_mask)
        self.mask_weight = torch.randn(num_heads, n_edge_mask+1) 
        # else:
        #     self.mask_weight = torch.randn(1, n_edge_mask+1) 
        self.mask_weight[:, -1] = (num_qk_channels_per_head**(0.5))
        self.mask_weight = nn.Parameter(self.mask_weight)
        # self.num_real_type = n_edge_mask

        # self.head_combine_weight = nn.Parameter(torch.zeros(num_heads))

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask = None,
        # relative_pos = None, 
        need_weights=False,
        # rot_pos_emb_q: Optional[RotaryPositionEmbedding] = None,
        # rot_pos_emb_k: Optional[RotaryPositionEmbedding] = None,
    ):
        """
        attn_mask: b, r, len(q), len(k) 
        key_padding_mask: b, len(k)
        """
        assert not need_weights 

        q = self.q_proj(x_q)
        k = self.k_proj(x_k)
        v = self.v_proj(x_v)


        q, k, v = (einops.rearrange(x, "n b (h c) -> b h n c", h=self.num_heads) for x in [q, k, v])

        rel_mask = torch.softmax(self.mask_weight[:, :-1], dim=-1)
        attn_mask = (~attn_mask).float()
        rel_mask = torch.einsum('h r, h, b r n m -> b h n m', rel_mask, self.mask_weight[:, -1], attn_mask)
        if key_padding_mask is not None: # b, m
            # attn_max_neg = -torch.finfo(rel_mask.dtype).max 
            rel_mask.masked_fill_(key_padding_mask[:, None, None], float("-inf")) # attn_max_neg)

        o = F.scaled_dot_product_attention(q, k, v, attn_mask=rel_mask, dropout_p=self.dropout if self.training else 0)
        o = einops.rearrange(o, "b h n c -> n b (h c)") # h=self.num_heads)
        o = self.o_proj(o)

        return o, None


class DynamicGraphAttention(nn.Module):

    def __init__(self, action_dim, frame_dim, nhead, dim_feedforward=2048, dropout=0.1, attn_dropout=0.1,
                 activation="relu",
                 ca_value_w_pos=False, n_edge_mask=3):
        """
        Dynamic-Self-Attention + Cross-Attention Module
        """
        super().__init__()


        self.action_token_attn = DynamicMultiheadAttention(nhead, action_dim, dropout=attn_dropout, n_edge_mask=n_edge_mask)
        self.cross_attn = nn.MultiheadAttention(action_dim, nhead, kdim=frame_dim, vdim=frame_dim, dropout=attn_dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(action_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, action_dim)

        self.norm1 = nn.LayerNorm(action_dim)
        self.norm2 = nn.LayerNorm(action_dim)
        self.norm3 = nn.LayerNorm(action_dim)

        self.dropout = nn.Dropout(dropout)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.ca_value_w_pos = ca_value_w_pos
        

    def forward(self, all_token, attn_mask, token_padding_mask, frame, frame_padding_mask, frame_pos):
        
        # dynamic self attention
        all_token2, _ = self.action_token_attn(all_token, all_token, all_token, 
                                                key_padding_mask=token_padding_mask, attn_mask=attn_mask, 
                                                need_weights=False)
        all_token = self.norm1(all_token + self.dropout(all_token2))

        # cross attention
        key = add_positional_encoding(frame, frame_pos) 
        value = key if self.ca_value_w_pos else frame
        all_token2, _ = self.cross_attn(all_token, key, value, 
                                            key_padding_mask=frame_padding_mask, 
                                            need_weights=False)
        all_token = self.norm2(all_token + self.dropout(all_token2))

        # ffn
        tgt = all_token
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(tgt2))

        return tgt


class DynamicGraphTransformer(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, decoder_layer, num_layers, norm=None, in_map=False):
        super().__init__()
        self.in_map = in_map
        if in_map:
            self.in_linear = nn.Linear(in_dim, hid_dim)
        else:
            assert hid_dim == in_dim
        self.layers = _get_clones(decoder_layer, num_layers)
        self.out_linear = nn.Linear(hid_dim, out_dim)
        self.num_layers = num_layers
        if norm is None:
            self.norm = None
        else:
            self.norm = nn.LayerNorm(out_dim)

    def forward(self, all_token, attn_mask, token_padding_mask, frame, frame_padding_mask, frame_pos):
        if self.in_map:
            all_token = self.in_linear(all_token)

        output = all_token

        self.intermediate = []
        for layer in self.layers:
            output = layer(output, attn_mask, token_padding_mask, frame, frame_padding_mask, frame_pos)
            self.intermediate.append(output)

        output = self.out_linear(output)

        if self.norm is not None:
            output = self.norm(output)

        return output


#==================================================
#==================================================
class SHCA(nn.Module):

    def __init__(self, x_dim, y_dim, y_outdim, head_dim, dropout=0.5, drop_on_att=True, kq_pos=False):
        super().__init__()
        self.drop_on_att = drop_on_att
        self.kq_pos = kq_pos

        self.X_K = nn.Linear(x_dim, head_dim)
        self.X_V = nn.Linear(x_dim, head_dim)
        self.Y_Q = nn.Linear(y_dim, head_dim)

        self.Y_W = nn.Linear(y_dim+head_dim, y_outdim)
        self.dropout = nn.Dropout(dropout)

        self.alignment_weight = nn.Parameter(torch.FloatTensor([0]))

    def forward(self, X, Y, alignment=None, X_pos=None, Y_pos=None, X_pad_mask=None):
        """
        X: x, b, h
        Y: y, b, h
        alignment: b y x
        X_pad_mask: b, y, x -> 1 denote masking out
        """
        attn_logit = self.compute_logit(X, Y, X_pos=X_pos, Y_pos=Y_pos)
        Y = self.compute_output(attn_logit, X, Y, alignment=alignment, X_pad_mask=X_pad_mask)
        return Y

    def compute_logit(self, X, Y, X_pos=None, Y_pos=None):

        if (X_pos is not None) and self.kq_pos:
            x = add_positional_encoding(X, X_pos)
            xk = self.X_K(x) 
        else:
            xk = self.X_K(X)


        if (Y_pos is not None) and self.kq_pos:
            y = add_positional_encoding(Y, Y_pos)
            yq = self.Y_Q(y)
        else:
            yq = self.Y_Q(Y)


        attn_logit = torch.einsum('xbd,ybd->byx', xk, yq)
        attn_logit = attn_logit / math.sqrt(xk.shape[-1])
        self.attn_logit_bs_nomask = attn_logit # logit before adding shared alignment and mask
        return attn_logit

    def compute_output(self, attn_logit, X, Y, alignment=None, X_pad_mask=None):
        attn_mask = None
        if X_pad_mask is not None:
            attn_mask = torch.zeros_like(X_pad_mask, dtype=torch.float)
            attn_mask.masked_fill_(X_pad_mask, -torch.finfo(X.dtype).max)

        if alignment is not None:
            alignment = torch.sigmoid(self.alignment_weight) * alignment
            attn_logit = attn_logit + alignment 
            attn_logit = torch.clamp(attn_logit, min=-torch.finfo(attn_logit.dtype).max)

        self.attn_logit_nomask = attn_logit
        if attn_mask is not None:
            attn_logit = attn_logit + attn_mask
            attn_logit = torch.clamp(attn_logit, min=-torch.finfo(attn_logit.dtype).max)
            self.attn_logit_bs = self.attn_logit_bs_nomask + attn_mask
            self.attn_logit_bs = torch.clamp(self.attn_logit_bs, min=-torch.finfo(attn_logit.dtype).max)
        self.attn_logit = attn_logit
        attn = torch.softmax(attn_logit, dim=-1) # B, y, x

        if self.drop_on_att:
            attn = self.dropout(attn)
        
        xv = self.X_V(X)
        attn_feat = torch.einsum('byx,xbh->ybh', attn, xv)

        concat_feature = torch.cat([Y, attn_feat], dim=-1)

        if not self.drop_on_att:
            concat_feature = self.dropout(concat_feature)

        Y = self.Y_W(concat_feature)

        self.attn = attn.unsqueeze(1) # B, nhead=1, X, Y

        return Y