import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
from collections import defaultdict
from . import basic
from ..utils import utils
from ..configs.utils import update_from
from .basic import CombinedSequence, PAD_ID
from copy import deepcopy
from ..utils.dataset import class_label_to_segment_label

class REUSE():
    def __init__(self) -> None:
        pass

    def clone(self, deep=False):
        new = REUSE()
        for k, v in vars(self).items():
            if deep:
                v = v.clone()
            setattr(new, k, v)
        return new


def create_f(cfg, in_dim=None, f_inmap=False):
    if in_dim is None:
        in_dim = cfg.f_dim

    frame_update = basic.TCN(in_dim, cfg.f_dim, cfg.f_dim, cfg.f_layers, 
                        dropout=cfg.dropout, dilation_factor=cfg.f_d, ln=cfg.f_ln, ngroup=cfg.f_ngp,
                        in_map=f_inmap)

    return frame_update

class PrototypeBuildingBlock(nn.Module):

    def __init__(self, cfg, in_dim):
        super().__init__()
        self.cfg = cfg
        PBB = cfg.PBB

        ##############################################
        # parameters for initializing features
        self.action_embed = basic.ClassEmbedding.load_and_create(cfg.temb_path)

        if self.action_embed.fg_action_embedding.shape[-1] != PBB.a_dim:
            self.action_embed_map = nn.Linear(self.action_embed.fg_action_embedding.shape[-1], PBB.a_dim)
        else:
            self.action_embed_map = nn.Identity()
        self.task_node = nn.Parameter(torch.randn(PBB.a_dim))
        self.action_pe = basic.PositionalEncoding(PBB.a_dim, max_len=500)

        self.frame_update = create_f(PBB, in_dim, f_inmap=True)
        self.channel_masking_dropout = nn.Dropout1d(p=0.3)
        self.frame_pe = basic.PositionalEncoding(PBB.f_dim, max_len=10000)

        layer = basic.SCALayer(PBB.a_dim, PBB.f_dim, PBB.a_nhead, PBB.a_ffdim,
                                        dropout=PBB.dropout, 
                                        attn_dropout=PBB.dropout if PBB.attn_dp else 0.0, 
                                        sa_value_w_pos=False, ca_value_w_pos=False
                                        )
        self.instance_node_update = basic.BasicTransformer(PBB.a_dim, PBB.a_dim, PBB.a_dim, layer, PBB.a_layers, norm=True, in_map=PBB.a_inmap)

        ##############################################
        # dynamic graph transformer
        self.edge_mask_list = PBB.dgt_etype.split(',')
        layer = basic.DynamicGraphAttention(PBB.dgt_dim, PBB.f_dim, PBB.dgt_nhead, PBB.dgt_ffdim,
                                            dropout=PBB.dropout,
                                            attn_dropout=PBB.dropout if PBB.attn_dp else 0.0, 
                                            n_edge_mask = len(self.edge_mask_list)
                                            )  
        self.v_dgt = basic.DynamicGraphTransformer(PBB.a_dim, PBB.dgt_dim, PBB.a_dim, 
                                                         layer, PBB.dgt_layers, norm=True, in_map=(PBB.dgt_dim!=PBB.a_dim))
        self.textual_exclude_mask = ['a', 'v', 'c2']
        n = len([x for x in self.edge_mask_list if x not in self.textual_exclude_mask]) 
        layer = basic.DynamicGraphAttention(PBB.dgt_dim, PBB.f_dim, PBB.dgt_nhead, PBB.dgt_ffdim,
                                        dropout=PBB.dropout,
                                        attn_dropout=PBB.dropout if PBB.attn_dp else 0.0, 
                                        n_edge_mask = n
                                        )
        self.t_dgt = basic.DynamicGraphTransformer(PBB.a_dim, PBB.dgt_dim, PBB.a_dim, 
                                                         layer, PBB.dgt_layers // 2, norm=True, in_map=(PBB.dgt_dim!=PBB.a_dim)) 
                                                         # use less layers for textual data to reduce overfitting

    def compute_edge_mask(self, ntask, naction, tid, vid, aid, padding_mask, exclude_list=[]):
        """
        ntask: number of task nodes
        naction: number of action nodes
        tid, vid, aid: b, action + instance nodes
        padding_mask: b, action + instance nodes
        """

        mask_types = self.edge_mask_list
        if len(mask_types) == 0:
            return None

        
        # create variable for later use
        B = tid.shape[0] # batch size
        to_task_mask = (tid[..., None] == torch.arange(ntask).to(tid.device)) # B, naction+ninstance, ntask 
                                                                              # attention from action/instance nodes to the task node of same task
        to_task_mask_null = torch.zeros_like(to_task_mask).bool().to(tid.device) # no attention to task nodes
        from_task_mask_null = torch.zeros([B, ntask, ntask+tid.shape[1]]).bool().to(tid.device) # B, ntask, total_token
                                                                              # attention mask from task nodes to all nodes, but masking out all attentions 

        mask_list = []
        for m in mask_types:
            if m in exclude_list:
                continue

            if m == 't': # Task Edge
                task_node_mask = torch.zeros([B, ntask+tid.shape[1], ntask+tid.shape[1]]).bool().to(tid.device)
                task_node_mask[:, :ntask, :ntask] = True # attentions among task nodes
                task_node_mask[:, :ntask, ntask:] = to_task_mask.transpose(1, 2) # attentions from task nodes to action/instance nodes of same task
                task_node_mask[:, :ntask, ntask+naction:] = False  # masking out attentions to instance nodes
                task_node_mask[:, ntask:, :ntask] = to_task_mask  # attentions from action/instance nodes to task nodes of same task
                task_node_mask[:, ntask+naction:, :ntask] = False # masking out attentions from instance nodes to task nodes 
                mask_list.append(task_node_mask)
            
            elif m == 'p': # Prototype Edge
                # action nodes attend to action nodes of the same task
                proto_mask = (vid.unsqueeze(1) == vid.unsqueeze(-1)) # B, total_token - ntask, total_token - ntask
                proto_mask[:, naction:, naction:] = False # masking out attention to/from instance nodes
                proto_mask = torch.cat([to_task_mask_null, proto_mask], dim=-1) # padding 
                proto_mask = torch.cat([from_task_mask_null, proto_mask], dim=1) # B, total_token, total_token
                mask_list.append(proto_mask)

            elif m == 'a': # Action Edge
                m1 = (tid.unsqueeze(1) == tid.unsqueeze(-1)) # attention among action/instance nodes of same task
                m2 = (aid.unsqueeze(1) == aid.unsqueeze(-1)) # attention among action/instance nodes of same action
                same_action_mask = torch.logical_and(m1, m2)
                same_action_mask = torch.cat([to_task_mask_null, same_action_mask], dim=-1)
                same_action_mask = torch.cat([from_task_mask_null, same_action_mask], dim=1) # B, total_token, total_token
                mask_list.append(same_action_mask)

            elif m == 'v': # Video Edge
                # instance nodes attend to instance nodes in same video
                same_video_mask = (vid.unsqueeze(1) == vid.unsqueeze(-1)) # B, total_token - ntask, total_token - ntask
                same_video_mask[:, :naction, :naction] = False # no attention among action nodes
                same_video_mask = torch.cat([to_task_mask_null, same_video_mask], dim=-1)
                same_video_mask = torch.cat([from_task_mask_null, same_video_mask], dim=1) # B, total_token, total_token
                mask_list.append(same_video_mask)

            elif m == 'c1': # Context Edge 1
                instance_task_mask = torch.zeros([B, ntask+tid.shape[1], ntask+tid.shape[1]]).bool().to(tid.device) # B, total_token, total_token
                instance_task_mask[:, :ntask, ntask:] = to_task_mask.transpose(1, 2) # attentions from task nodes to action/instance nodes of same task
                instance_task_mask[:, :ntask, ntask:ntask+naction] = False # masking out attentions to action nodes
                instance_task_mask[:, ntask:, :ntask] = to_task_mask # attentions from action/instance nodes to task nodes of same task
                instance_task_mask[:, ntask:ntask+naction, :ntask] = False # masking out attentions from action nodes
                mask_list.append(instance_task_mask)

            elif m  == 'c2': # Context Edge 2
                # instance nodes attend to instances nodes of same task, different video, different action
                diff_video_action_mask = (tid.unsqueeze(1) == tid.unsqueeze(-1)) # attention among instance nodes of same task
                diff_video_action_mask = torch.logical_and(diff_video_action_mask, aid.unsqueeze(1) != aid.unsqueeze(-1) ) # remove attention among nodes of same action
                diff_video_action_mask = torch.logical_and(diff_video_action_mask, vid.unsqueeze(1) != vid.unsqueeze(-1) ) # remove attention among nodes of same video; this also removes attention among action nodes and instance nodes
                diff_video_action_mask = torch.cat([to_task_mask_null, diff_video_action_mask], dim=-1)
                diff_video_action_mask = torch.cat([from_task_mask_null, diff_video_action_mask], dim=1) # B, total_token, total_token
                mask_list.append(diff_video_action_mask)

            else:
                raise ValueError(f'unknown mask type {m}')

        attn_mask = torch.stack(mask_list, dim=1) # b, nrel, na, nall
        attn_mask = ~attn_mask

        padding_mask = ~padding_mask.bool() # True for padded and False for non-padded
        padding_mask = torch.cat([padding_mask.new_zeros([B, ntask]), padding_mask], dim=1)
        attn_mask.masked_fill_(padding_mask[:, None, None, :], True)
        attn_mask.masked_fill_(padding_mask[:, None, :, None], False)

        return attn_mask    

    def initalize_visual_feature(self, frame: CombinedSequence, transcript, segment_label):
        """
        initialize frame features and features of instance nodes
        we follow a similar approach as the Input Block in FACT model.
        """

        ###################################
        # use convolution to initialize frame features
        frame_feature = frame.sequences.permute([0, 2, 1]) # B, H, T
        frame_feature = self.channel_masking_dropout(frame_feature)
        frame.sequences = frame_feature.permute([2, 0, 1]) # T, B, H

        frame_pe = self.frame_pe(frame.sequences)

        _masks = einops.rearrange(frame.masks, 'b t h -> t b h')
        frame_feature = self.frame_update(frame.sequences, _masks)
        frame.sequences = frame_feature
        # self.pre_frame = frame_feature


        ###################################
        # initialize features of instance nodes
        N = transcript.sequences.shape[1]
        seg_idx = torch.arange(N).to(transcript.sequences.device)
        onehot = (seg_idx == segment_label.sequences.unsqueeze(-1)) # onehot label denoting which frame belongs to which action segment

        # run average pooling for features of each action segment
        # this gives the first features for instance nodes
        pooled_action_feat = torch.einsum('bts,tbh->sbh', onehot.float(), frame.sequences) / ( onehot.sum(1).T.unsqueeze(-1) + 1e-5 )  
        inode = CombinedSequence(
            sequences = pooled_action_feat, masks = transcript.masks.unsqueeze(-1), aid = transcript.sequences, 
        )
        inode_pe = self.action_pe(inode.sequences) # A, 1, T
        inode_pe = einops.repeat(inode_pe, 'a b t -> a (repeat b) t', repeat=inode.sequences.size(1)).clone()
        _bg_mask = (inode.aid == 0).permute(1, 0).unsqueeze(-1)
        inode_pe.masked_fill_(_bg_mask, 0) # no positional encoding for background token 
        
        # further refine instance node via cross-attention to the frame features of their corresponding frames
        ca_attn_mask = ~onehot.permute(0, 2, 1) # b, n, t
        ca_attn_mask.masked_fill_(~inode.masks.bool(), False) # padding node cannot see any frame, lead to nan. this is to unmask them.
        B, N, T = ca_attn_mask.shape
        ca_attn_mask = ca_attn_mask.unsqueeze(1).expand([B, self.cfg.PBB.a_nhead, N, T])
        ca_attn_mask = ca_attn_mask.reshape(-1, N, T) 

        imask = (1 - inode.masks[..., 0]).bool()
        fmask = (1- frame.masks[..., 0]).bool()

        inode.sequences = self.instance_node_update(inode.sequences, frame.sequences, imask, fmask, ca_attn_mask, 
                                                       pos=frame_pe, query_pos=inode_pe)
        masks = einops.rearrange(inode.masks, 'b n h -> n b h')
        inode.sequences = inode.sequences * masks
        return frame, frame_pe, inode

    def gather_action_node_atv_id(self, transcript: CombinedSequence, qs_vidx):
        device = transcript.sequences.device

        support_video_idx = [ x[1] for x in qs_vidx ]
        support_video_idx = torch.LongTensor(support_video_idx).to(device)
        aid = transcript.sequences[support_video_idx] # action class id of instance nodes
        B, T, V, N = aid.shape # batch, number task, number video, number action per video
        aid_np = aid.view(B, T, -1).detach().cpu().numpy() 

        ##########################################
        # find the unique actions in each task, and gather action nodes' information
        data = []
        for b in range(B):
            d = []
            for t in range(T):
                aid=set(aid_np[b, t].tolist()) # IDs of all unique actions in this task
                if PAD_ID in aid: aid.remove(PAD_ID)  # remove padding action class id

                if 0 in aid: aid.remove(0) # remove background action class id
                aid = list(aid)
                aid.insert(0, 0)  # add back background action class id 

                aid = torch.LongTensor(aid)
                n = len(aid)
                tid=torch.zeros([n]).long() + t
                vid=torch.zeros([n]).long() + T*V + t # for convinience, we give a fake video id to action nodes, which is different from all video ids of instance nodes
                d.append([aid, tid, vid])
        
            aid = torch.concat([x[0] for x in d]).to(device)
            tid = torch.concat([x[1] for x in d]).to(device)
            vid = torch.concat([x[2] for x in d]).to(device)
            data.append([aid, tid, vid])
        
        lens = [ len(data[b][0]) for b in range(B) ]
        L = max(lens) # max number of action nodes per task


        ###############################################################
        # create a flatten sequence of all action nodes in the graph
        anode_aid, anode_tid, anode_vid = [ torch.zeros([B, L], device=device) + PAD_ID for _ in range(3) ]
        anode_mask = torch.zeros([B, L], device=device).bool()
        for b in range(B):
            l1 = lens[b]
            # t_anode[b, :l1] = t_action_embedding[data[b][0]]
            anode_aid[b, :l1] = data[b][0]
            anode_tid[b, :l1] = data[b][1]
            anode_vid[b, :l1] = data[b][2]
            anode_mask[b, :l1] = True

        return anode_aid, anode_tid, anode_vid, anode_mask, lens

    def initialize_textual_graph(self, anode_data, ntask):

        B = anode_data[0].shape[0]
        H = self.task_node.shape[0]
        device = anode_data[0].device

        ########################################
        # task Token
        t_task_node = self.task_node.view([1, 1, H]).expand([ntask, B, H]) # the initial feature of task tokens, shared for visual and textual modality

        #########################################
        # action nodes
        self.action_embed.init_embedding()
        t_action_embedding = torch.cat([self.action_embed.bg_embedding, self.action_embed.batch_fg_action_embedding], dim=0) # C, E
        t_action_embedding = t_action_embedding / torch.clamp( torch.norm(t_action_embedding, dim=-1, keepdim=True), min=1e-5 )
        t_action_embedding = self.action_embed_map(t_action_embedding)

        anode_aid, anode_tid, anode_vid, anode_mask, lens = anode_data
        L = max(lens) # max number of action nodes per task
        t_anode = torch.zeros([B, L, H], device=device)
        for b in range(B):
            l1 = lens[b]
            t_anode[b, :l1] = t_action_embedding[anode_aid[b, :l1].long()]

        t_anode = t_anode.permute(1, 0, 2)
        t_anode = CombinedSequence(sequences=t_anode, masks=anode_mask.unsqueeze(-1).float(), tid=anode_tid, vid=anode_vid, aid=anode_aid)
        t_attn_mask = self.compute_edge_mask(ntask, L, anode_tid, anode_vid, anode_aid, anode_mask.squeeze(-1), 
                                             exclude_list=self.textual_exclude_mask)
        t_attn_mask = t_attn_mask[:, :, :ntask+L, :ntask+L]

        return t_task_node, t_anode, t_attn_mask #, [ntask, L]


    def initalize_visual_graph(self, initial_inode: CombinedSequence, anode_data, qs_vidx, transcript):
        N, TOTAL_V, H = initial_inode.sequences.shape
        ntask = len(qs_vidx[0][1])
        inode_sequences = initial_inode.sequences.transpose(0, 1) # n b h -> b n h
        device = initial_inode.sequences.device

        #########################################
        # instance nodes
        support_video_idx = [ x[1] for x in qs_vidx ]
        ei = torch.LongTensor(support_video_idx).to(device)

        inode = inode_sequences[ei] # initial features of visual instance nodes (no instance node for textual modality)
        inode_aid = transcript.sequences[ei] # action class id of instance nodes
        B, T, V, N = inode_aid.shape # batch, number task, number video, number action per video
        inode_tid = torch.arange(T, device=device) # task id of instance nodes
        inode_tid = inode_tid[None, :, None, None].expand_as(inode_aid)
        inode_vid = torch.arange(T * V, device=device) # video id of instance nodes
        inode_vid = inode_vid[None, :, None].expand(B, -1, N)

        inode = inode.view(B, -1, H)
        inode_tid = inode_tid.reshape(B, -1)
        inode_vid = inode_vid.reshape(B, -1)
        inode_aid = inode_aid.view(B, -1)
        inode_mask = initial_inode.masks[ei].view(B, -1).bool()
        
        #########################################
        # action nodes
        anode_aid, anode_tid, anode_vid, anode_mask, lens = anode_data
        L = max(lens) # max number of action nodes per task

        zero_emb = torch.zeros_like(self.action_embed.batch_fg_action_embedding) # for simplicity, we use zeros as the initial embedding for action nodes
        v_action_embedding = torch.cat([self.action_embed.bg_embedding, zero_emb], dim=0) # C, E
        v_action_embedding = v_action_embedding / torch.clamp( torch.norm(v_action_embedding, dim=-1, keepdim=True), min=1e-5 )
        v_action_embedding = self.action_embed_map(v_action_embedding)

        v_anode = torch.zeros([B, L, H], device=device)
        for b in range(B):
            l1 = lens[b]
            v_anode[b, :l1] = v_action_embedding[anode_aid[b, :l1].long()]

        ########################################
        # task Token
        v_task_node = self.task_node.view([1, 1, H]).expand([ntask, B, H]) # the initial feature of task tokens, shared for visual and textual modality

        ########################################
        # assemble nodes -- create a flatten sequence of all action and instance nodes in the graph
        ai_aid = torch.concat([anode_aid, inode_aid], dim=1) # B, naction + ninstance
        ai_tid = torch.concat([anode_tid, inode_tid], dim=1) 
        ai_vid = torch.concat([anode_vid, inode_vid], dim=1)
        ai_mask = torch.concat([anode_mask, inode_mask], dim=1).float().unsqueeze(-1)

        v_ai_node = torch.concat([v_anode, inode], dim=1).permute(1, 0, 2)
        v_ai_node = CombinedSequence(sequences=v_ai_node, masks=ai_mask, tid=ai_tid, vid=ai_vid, aid=ai_aid)
        v_attn_mask = self.compute_edge_mask(ntask, L, ai_tid, ai_vid, ai_aid, ai_mask.squeeze(-1))

        return v_task_node, v_ai_node, v_attn_mask 

    def visual_graph_learning(self, frame: CombinedSequence, frame_pe, graph_data, ntask, naction):

        task_node, action_instance_data, attn_mask = graph_data
        action_instance_node = action_instance_data.sequences
        all_node = torch.cat([task_node, action_instance_node], dim=0)
        frame_mask = (1 - frame.masks[..., 0]).bool()
        node_padding_mask = None # No need, as we have masked out padding nodes in all_attn_mask

        all_node = self.v_dgt(all_node, attn_mask, node_padding_mask,
                                                frame.sequences, frame_mask, frame_pe)

        task_node = all_node[:ntask]
        action_instance_node = all_node[ntask:]
        action_node = action_instance_node[:naction]

        mask = action_instance_data.masks[:, :naction]
        action_node.masked_fill_(~mask.bool().permute(1, 0, 2), 0)

        action_proto = CombinedSequence(
            sequences = action_node, 
            masks = mask,
            aid = action_instance_data.aid[:, :naction].long(),
            tid = action_instance_data.tid[:, :naction].long(),
            vid = action_instance_data.vid[:, :naction].long(),
        )

        return task_node, action_proto

    def textual_graph_learning(self, frame: CombinedSequence, frame_pe, graph_data, ntask, naction):

        task_node, action_instance_data, all_attn_mask = graph_data
        action_instance_node = action_instance_data.sequences
        all_node = torch.cat([task_node, action_instance_node], dim=0)
        frame_mask = (1 - frame.masks[..., 0]).bool()
        node_padding_mask = None # No need, as we have masked out padding nodes in all_attn_mask

        all_node = self.t_dgt(all_node, all_attn_mask, node_padding_mask,
                                                frame.sequences, frame_mask, frame_pe)
        task_node = all_node[:ntask]
        action_instance_node = all_node[ntask:]
        action_node = action_instance_node[:naction]

        masks = action_instance_data.masks[:, :naction]
        action_node.masked_fill_(~masks.bool().permute(1, 0, 2), 0)

        action_proto = CombinedSequence(
            sequences = action_node, 
            masks = masks,
            aid = action_instance_data.aid[:, :naction].long(),
            tid = action_instance_data.tid[:, :naction].long(),
            vid = action_instance_data.vid[:, :naction].long(),
        )

        return task_node, action_proto

    def forward(self, frame: CombinedSequence, reuse_vars):
        transcript = reuse_vars.transcript
        segment_label = reuse_vars.segment_label
        qs_vidx = reuse_vars.qs_vidx
        ntask = len(qs_vidx[0][1])

        # prepare graph
        anode_data = self.gather_action_node_atv_id(transcript, qs_vidx)
        textual_graph_data = self.initialize_textual_graph(anode_data, ntask)

        frame, frame_pe, inode = self.initalize_visual_feature(frame, transcript, segment_label)
        visual_graph_data = self.initalize_visual_graph(inode, anode_data, qs_vidx, transcript)

        # run dynamic graph transformer
        qindices = [ e[0] for e in qs_vidx ]
        query_frame = frame.clone()
        query_frame.sequences = query_frame.sequences[:, qindices]
        query_frame.masks = query_frame.masks[qindices]
        query_frame.lens = query_frame.lens[qindices]
        naction = max(anode_data[-1])
        reuse_vars.frame_pe = frame_pe
        
        v_task_node, v_action_proto = self.visual_graph_learning(query_frame, frame_pe, visual_graph_data, ntask, naction)
        t_task_node, t_action_proto = self.textual_graph_learning(query_frame, frame_pe, textual_graph_data, ntask, naction)
        # print(v_task_node.mean())
        # print(t_task_node.mean())
        # import ipdb; ipdb.set_trace()

        return [v_task_node, v_action_proto, query_frame], [t_task_node, t_action_proto, query_frame]

    def textual_only_forward(self, query_frame: CombinedSequence, reuse_vars):
        transcript = reuse_vars.transcript
        qs_vidx = reuse_vars.qs_vidx
        ntask = len(qs_vidx[0][1])

        # prepare graph
        anode_data = self.gather_action_node_atv_id(transcript, qs_vidx)
        textual_graph_data = self.initialize_textual_graph(anode_data, ntask)

        # initialize frame features of query video
        frame_feature = query_frame.sequences.permute([0, 2, 1]) # B, H, T
        frame_feature = self.channel_masking_dropout(frame_feature)
        query_frame.sequences = frame_feature.permute([2, 0, 1]) # T, B, H

        frame_pe = self.frame_pe(query_frame.sequences)

        _masks = einops.rearrange(query_frame.masks, 'b t h -> t b h')
        frame_feature = self.frame_update(query_frame.sequences, _masks)
        query_frame.sequences = frame_feature

        # run dynamic graph transformer
        naction = max(anode_data[-1])
        reuse_vars.frame_pe = frame_pe
        t_task_node, t_action_proto = self.textual_graph_learning(query_frame, frame_pe, textual_graph_data, ntask, naction)

        return t_task_node, t_action_proto, query_frame


class MatchingBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.node_type_emb = nn.Parameter(torch.rand([2, cfg.PBB.a_dim]))
        self.task_match_emb = nn.Parameter(torch.rand([2, cfg.PBB.a_dim]))

        self.vt_block = Matching_SubBlock1_Shared(cfg)
        self.v_block = Matching_SubBlock2_Seperate(cfg)
        self.t_block = Matching_SubBlock2_Seperate(cfg)

        if self.cfg.MB.f_share: # share the weights of frame convolution layers between two modalities
            del self.t_block.frame_update
            self.t_block.frame_update = self.v_block.frame_update

    def compute_action_attn_mask(self, action_proto: CombinedSequence, task_node: Tensor):
        """
        To run attention over action nodes and task nodes, we compute 
        (1) joint embedding to denote node types and if two nodes belong to the same task
        (2) attention mask to mask out padding nodes
        """
        na, b, h = action_proto.sequences.shape
        nt = task_node.shape[0]

        # embedding to denote a node is a task node or action node
        type_idx = torch.LongTensor([ 0 ] * nt + [ 1 ] * na).to(task_node.device)
        type_emb = self.node_type_emb[type_idx][None, None, ...] # 1, 1, nt+na, h

        # embedding to denote if two nodes belong to the same task
        tid = torch.arange(nt).view(1, nt).expand(b, nt).to(task_node.device)
        tid = torch.cat([tid, action_proto.tid], dim=1) # b, nt+na
        tid = (tid[:, None, :] == tid[:, :, None]).long() # b, nt+na, nt+na
        task_match_emb = self.task_match_emb[tid] # b, nt+na, nt+na, h

        joint_emb = task_match_emb + type_emb

        mask = action_proto.masks  # b, na, 1
        mask = torch.concat([ torch.ones(task_node.shape[1], task_node.shape[0], 1).to(mask.device), mask ], dim=1).permute(0, 2, 1) # b, na, 1
        mask = ~ mask.bool()

        return joint_emb, mask

        

    def forward(self, visual_data, textual_data, reuse):

        visual_data2, textual_data2 = self.vt_block(visual_data, textual_data)

        v_reuse = reuse.clone()
        v_reuse.node_emb, v_reuse.node_mask = self.compute_action_attn_mask(visual_data[1], visual_data[0])
        # visual_data2.append(v_reuse)

        t_reuse = reuse.clone()
        t_reuse.node_emb, t_reuse.node_mask = self.compute_action_attn_mask(textual_data[1], textual_data[0])
        # textual_data2.append(t_reuse)

        self.v_output = self.v_block(*visual_data2, v_reuse)
        self.t_output = self.t_block(*textual_data2, t_reuse)

        self.v_action_frame_similarity = self.v_output[3]
        self.t_action_frame_similarity = self.t_output[3]

        return self.v_output, self.t_output

    def textual_only_forward(self, textual_data, reuse):
        _, t_output = self.vt_block(None, textual_data)
        t_reuse = reuse.clone()
        t_reuse.node_emb, t_reuse.node_mask = self.compute_action_attn_mask(textual_data[1], textual_data[0])
        # t_output.append(t_reuse)
        t_output = self.t_block(*t_output, t_reuse)
        return t_output

    
    def compute_loss(self, *args, **kwargs):

        loss1 = self.vt_block.compute_loss(*args, **kwargs)

        v_loss = self.v_block.compute_loss(*args, **kwargs)
        t_loss = self.t_block.compute_loss(*args, **kwargs)
        loss2 = ( v_loss + t_loss ) / 2
        self.loss_dict = utils.easy_reduce([self.v_block.loss_dict, self.t_block.loss_dict]) 

        loss = ( loss1 + loss2 ) / 2

        return loss

def compute_simi_loss(action_proto, action_frame_similarity, task_activation, train_label, qs_vidx):

    simi_loss = []
    log_similarity = []

    for bidx, (query, supports) in enumerate(qs_vidx):
        T = train_label.lens[query] # get the length of the query video

        m1 = action_proto.masks[bidx, :, 0].bool() # padding mask
        tid = action_proto.tid[bidx, m1]  # task id for each action node
        support_actions = action_proto.aid[bidx, m1]
        l = action_frame_similarity.new_zeros(T, m1.sum()) # T, A
        m2 = tid ==0
        support_actions = support_actions[m2]
        l[:, m2] = (support_actions == train_label.sequences[query, :T, None]).float() # T, A_task1
        new_bg = l.sum(1) == 0
        assert support_actions[0] == 0, support_actions
        l[new_bg, 0] = 1 
        simi = action_frame_similarity[bidx, m1, :T].T
        task_prob = torch.softmax(task_activation[:, bidx, 0], dim=0) # ntask
        task_prob = task_prob[tid]
        simi = simi * task_prob

        logprob = torch.log_softmax(simi, dim=1)
        log_similarity.append(logprob)

        loss = - (l * logprob) # t, u
        focal_weight = (1 - torch.exp(-loss)) ** 2 # gamma==2
        loss = loss * focal_weight
        loss = loss.sum(0) / ( l.sum(0) + 1e-5 )
        loss = loss.sum() / ( (l.sum(0) > 0).sum() + 1e-5  )

        simi_loss.append(loss)

    # similarity_loss = cfg.falw * ( sum(simi_loss) / len(simi_loss) )
    similarity_loss = ( sum(simi_loss) / len(simi_loss) )
    return similarity_loss, log_similarity


class Matching_SubBlock1_Shared(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        PBB = cfg.PBB 

        self.v_task_linear = nn.Linear(PBB.a_dim, 1)
        self.v_first_SHCA = basic.SHCA(PBB.a_dim, PBB.f_dim, PBB.f_dim, head_dim=PBB.a_dim,  
                                                    dropout=PBB.dropout, drop_on_att=PBB.attn_dp, 
                                                    kq_pos=True, 
                                                    )

        self.t_task_linear = nn.Linear(PBB.a_dim, 1)
        self.t_first_SHCA = basic.SHCA(PBB.a_dim, PBB.f_dim, PBB.f_dim, head_dim=PBB.a_dim,  
                                                dropout=PBB.dropout, drop_on_att=PBB.attn_dp, 
                                                kq_pos=True, 
                                                )

    def forward(self, v_data, t_data):

        ##################################
        # compute attention logits
        if v_data is not None:
            v_task_node, v_action_proto, frame = v_data
            v_attn_logit = v_attn_logit_share = self.v_first_SHCA.compute_logit(v_action_proto.sequences, frame.sequences) 
            v_attn_logit_share = v_attn_logit_share.detach()
        else: # v_data is None if evaluating in zero-shot setting
            v_attn_logit_share = None

        t_task_node, t_action_proto, frame = t_data
        t_attn_logit = t_attn_logit_share = self.t_first_SHCA.compute_logit(t_action_proto.sequences, frame.sequences) 
        t_attn_logit_share = t_attn_logit_share.detach()

        ##################################
        # compute cross-attention with shared attention logits
        if v_data is not None:
            self.v_frame = frame.clone()
            self.v_frame.sequences = self.v_first_SHCA.compute_output(v_attn_logit, 
                                                                v_action_proto.sequences, frame.sequences, alignment = t_attn_logit_share,
                                                                X_pad_mask = ~v_action_proto.masks.permute(0, 2, 1).bool())
            self.v_task_activation = self.v_task_linear(v_task_node) # ntask, b, 1

            self.v_action_proto = v_action_proto
            self.v_action_frame_similarity = self.v_first_SHCA.attn_logit.permute(0, 2, 1)
            self.v_task_node_output = v_task_node
            self.v_output = [self.v_frame.clone(), self.v_action_proto.clone(), v_task_node, 
                          self.v_first_SHCA.attn_logit_nomask.permute(0, 2, 1)] 
        else:
            self.v_output = None

        self.t_frame = frame.clone()
        self.t_frame.sequences = self.t_first_SHCA.compute_output(t_attn_logit,
                                                            t_action_proto.sequences, frame.sequences, alignment = v_attn_logit_share,
                                                            X_pad_mask = ~t_action_proto.masks.permute(0, 2, 1).bool())
        self.t_task_activation = self.t_task_linear(t_task_node) # ntask, b, 1

        self.t_action_proto = t_action_proto
        self.t_action_frame_similarity = self.t_first_SHCA.attn_logit.permute(0, 2, 1)
        self.t_task_node_output = t_task_node
        self.t_output = [self.t_frame.clone(), self.t_action_proto.clone(), t_task_node, 
                          self.t_first_SHCA.attn_logit_nomask.permute(0, 2, 1)] 

        return self.v_output, self.t_output

    def compute_loss(self, train_label: CombinedSequence, qs_vidx: list):

        cfg = self.cfg.Loss

        # task prediction
        task_label = self.v_task_activation.new_zeros(self.v_task_activation.shape[1]).long() # ntask, b, 1
        task_prediction_loss = F.cross_entropy(self.v_task_activation[..., 0].T, task_label) + \
                                F.cross_entropy(self.t_task_activation[..., 0].T, task_label)
        task_prediction_loss = task_prediction_loss * cfg.task_w / 2

        # frame action similarity
        v_similarity_loss, v_log_similarity = compute_simi_loss(self.v_action_proto, self.v_action_frame_similarity, self.v_task_activation, train_label, qs_vidx)
        t_similarity_loss, t_log_similarity = compute_simi_loss(self.t_action_proto, self.t_action_frame_similarity, self.t_task_activation, train_label, qs_vidx)
        similarity_loss = cfg.simi_w * (v_similarity_loss + t_similarity_loss) / 2

        # smooth loss
        smooth_loss = []
        for logp in v_log_similarity:
            l = torch.clamp((logp[1:] - logp[:-1])**2, min=0, max=16).mean()
            smooth_loss.append(l)
        v_smooth_loss = sum(smooth_loss) / len(smooth_loss)

        smooth_loss = []
        for logp in t_log_similarity:
            l = torch.clamp((logp[1:] - logp[:-1])**2, min=0, max=16).mean()
            smooth_loss.append(l)
        t_smooth_loss = sum(smooth_loss) / len(smooth_loss)

        smooth_loss = cfg.smooth_w * (v_smooth_loss + t_smooth_loss) / 2


        # alignment weight loss to avoid model relying too much on one modality
        v_w = torch.sigmoid(self.v_first_SHCA.alignment_weight)
        t_w = torch.sigmoid(self.t_first_SHCA.alignment_weight)
        alignment_weight_loss = (v_w ** 2 + t_w ** 2) / 2 * cfg.align_w

        loss = task_prediction_loss + similarity_loss + smooth_loss + alignment_weight_loss
        
        self.loss_dict = {
            'task': task_prediction_loss.item(),
            'simi': similarity_loss.item(),
            'smooth': smooth_loss.item(),
            'align': alignment_weight_loss.item(),
        }

        return loss


class Matching_SubBlock2_Seperate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        cfg = cfg.MB 

        # fupdate
        self.frame_update = create_f(cfg, f_inmap=False)

        # second SHCA: query is action, key/value is frame
        self.second_SHCA = basic.SHCA(cfg.f_dim, cfg.a_dim, cfg.a_dim, head_dim=cfg.f_dim, kq_pos=True, 
                                                dropout=cfg.dropout, drop_on_att=0)
                                          
        # aupdate
        l = basic.SALayer(cfg.a_dim, cfg.a_nhead, dim_feedforward=cfg.a_ffdim, dropout=cfg.dropout, attn_dropout=cfg.dropout, vpos=False)
        self.action_update = basic.BasicTransformer(cfg.a_dim, cfg.a_dim, cfg.a_dim, l, cfg.a_layers, in_map=cfg.a_inmap, norm=True)
        self.embedding_linear = nn.Linear(cfg.a_dim, 1)

        # third SHCA: query is frame, key/value is action
        self.third_SHCA = basic.SHCA(cfg.a_dim, cfg.f_dim, cfg.f_dim, head_dim=cfg.a_dim,  kq_pos=True,
                                                dropout=cfg.dropout, drop_on_att=0)
        
        # task classification
        self.task_linear = nn.Linear(cfg.a_dim, 1)


    def forward(self, frame: CombinedSequence, action_proto: CombinedSequence, task_node: Tensor, alignment: Tensor, reuse_vars):
        """
        frame_feature: T, B, H
        action_feature: N, B, H
        task_node: ntask, B, H
        alignment: B, N, T
        """

        frame = frame.clone()
        action_proto = action_proto.clone()

        # f update
        _masks = einops.rearrange(frame.masks, 'b t h -> t b h')
        frame.sequences = self.frame_update(frame.sequences, _masks)

        # second SHCA
        m = ~frame.masks.permute(0, 2, 1).bool()
        action_proto.sequences = self.second_SHCA(frame.sequences, action_proto.sequences, alignment, 
                                X_pos=reuse_vars.frame_pe, Y_pos=None, X_pad_mask = m)


        # a update
        feat = torch.concat([task_node, action_proto.sequences], dim=0)
        node_emb = reuse_vars.node_emb
        relative_pos = self.embedding_linear(node_emb).squeeze(-1) # b, a, a
        relative_pos.masked_fill_(reuse_vars.node_mask, float("-inf"))
        _b, _a, _ = relative_pos.shape
        relative_pos = relative_pos.unsqueeze(1).expand(_b, self.cfg.MB.a_nhead, _a, _a).reshape(-1, _a, _a)
        feat = self.action_update(feat, attn_mask=relative_pos)
        task_node = feat[:len(task_node)]
        action_proto.sequences = feat[len(task_node):]

        # third SHCA
        m = ~action_proto.masks.permute(0, 2, 1).bool()
        alignment = self.second_SHCA.attn_logit_nomask.permute(0, 2, 1)
        frame.sequences = self.third_SHCA(action_proto.sequences, frame.sequences, alignment,
                                    X_pos=None, Y_pos=None, X_pad_mask = m)


        self.task_activation = self.task_linear(task_node) # ntask, b, 1

        self.action_frame_similarity = self.third_SHCA.attn_logit.permute(0, 2, 1)
        self.frame = frame
        self.action_proto = action_proto
        self.task_node_output = task_node

        return frame.clone(), action_proto.clone(), task_node, self.third_SHCA.attn_logit_nomask.permute(0, 2, 1) #self.action_frame_similarity
    
    def compute_loss(self, train_label: CombinedSequence, qs_vidx: list):

        cfg = self.cfg.Loss

        # task prediction
        task_label = self.task_activation.new_zeros(self.task_activation.shape[1]).long() # ntask, b, 1
        task_prediction_loss = F.cross_entropy(self.task_activation[..., 0].T, task_label) * cfg.task_w

        # frame action similarity
        l1, log_similarity = compute_simi_loss(self.action_proto, self.action_frame_similarity, self.task_activation, train_label, qs_vidx)
        l2, _ = compute_simi_loss(self.action_proto, self.second_SHCA.attn_logit, self.task_activation, train_label, qs_vidx)
        similarity_loss = cfg.simi_w * (l1 + l2) / 2

        # smooth loss
        smooth_loss = []
        for logp in log_similarity:
            l = torch.clamp((logp[1:] - logp[:-1])**2, min=0, max=16).mean()
            smooth_loss.append(l)
        smooth_loss = cfg.smooth_w * sum(smooth_loss) / len(smooth_loss)

        loss = task_prediction_loss + similarity_loss + smooth_loss
        
        self.loss_dict = {
            'task': task_prediction_loss.item(),
            'simi': similarity_loss.item(),
            'smooth': smooth_loss.item(),
        }

        return loss

    def evaluate(self, train_label: CombinedSequence, qs_vidx, use_pred_task=True):
        task_activation = self.task_activation
        action_proto = self.action_proto
        action_frame_similarity = self.action_frame_similarity

        task_activation = torch.softmax(task_activation, dim=0)[..., 0] # T, B
        task_acc = (task_activation.argmax(0) == 0).float().mean().item() # qs_vidx are sampled such that the first support task is always the correct one
        scalars = {
            'task_pred_acc': task_acc,
        }
        
        predictions = []
        for bidx, (query, supports) in enumerate(qs_vidx):
            T = train_label.lens[query]

            if use_pred_task:
                i = task_activation[:, bidx].argmax().item()
            else:
                i = 0 # this is only used during fewshot (no-label/weak-label) setting to estimate the support video label
            m = (action_proto.tid[bidx] == i)
            support_actions = action_proto.aid[bidx, m]

            simi = action_frame_similarity[bidx, m, :T]
            p = support_actions[simi.argmax(0)]

            # uniques = action_proto.aid[bidx, action_proto.tid[bidx] == 0]
            # import ipdb; ipdb.set_trace()
            # support_vidx_flatten = np.array(supports).flatten().tolist()
            l = train_label.sequences[query, :T].clone()
            uniques = train_label.sequences[supports[0]].unique()
            uniques = uniques[uniques != PAD_ID]
            no_match = (l[:, None] != uniques).all(1) 
            l[no_match] = 0 # if an action in the query video does not appear in the support set, set it to background class
            
            predictions.append([p, l])

        return scalars, predictions


class PGNet(nn.Module):

    def __init__(self, cfg, in_dim):
        super().__init__()
        self.cfg = cfg
        block1 = PrototypeBuildingBlock(cfg, in_dim)

        update_from(cfg.MB, cfg.PBB, inplace=True)
        block2 = MatchingBlock(cfg)

        self.block_list = nn.ModuleList([block1, block2])
 
    def generate_qs_vidx(self, tasks):
        # pick the query video and support videos for each task
        if self.training: 
            return self._sample_qs_vidx(tasks) 
        else:
            return self._deterministic_qs_vidx(tasks)


    def _sample_qs_vidx(self, tasks):
        task_group = defaultdict(list)
        for i, t in enumerate(tasks):
            task_group[t].append(i)

        # n = len(task_group[t]) - 1
        qs_vidx = []
        for i, t in enumerate(tasks):
            supports = [[x for x in task_group[t] if x != i]]
            for task, vids in task_group.items():
                if task == t:
                    continue
                if len(vids) <= len(supports[0]):
                    supports.append(vids[:]) 
                else:
                    supports.append(np.random.choice(vids, len(supports[0]), replace=False).tolist())
            qs_vidx.append([i, supports])
 
        return qs_vidx

    def _deterministic_qs_vidx(self, tasks):
        task_group = defaultdict(list)
        for i, t in enumerate(tasks):
            task_group[t].append(i)

        assert len(task_group) == self.cfg.nt, f"task number not match: {len(task_group)} vs {self.cfg.nt}"
        # if len(task_group) != self.cfg.tnt:
        #     print('task number not match')
        #     import ipdb; ipdb.set_trace()

        n = self.cfg.nv - 1
        qs_vidx = []
        for query_task in task_group:
            vid = task_group[query_task][0]
            supports = []
            for task, vids in task_group.items():
                if task == query_task: 
                    supports.append(vids[1:n+1])
                else:
                    supports.append(vids[:n])
            qs_vidx.append([vid, supports])
            break
        
        return qs_vidx

    def set_subset(self, subset):
        for b in self.block_list:
            b._subset = subset

    def forward_and_loss(self, tasks, frame: CombinedSequence, 
                         transcript: CombinedSequence, class_label: CombinedSequence, segment_label: CombinedSequence):

        ############## prepare input
        reuse = REUSE() # create a object to contain reusable variables and pass them through blocks
        reuse.transcript = transcript
        reuse.segment_label = segment_label
        reuse.cls_label = class_label
        reuse.tasks = tasks
        qs_vidx = self.generate_qs_vidx(tasks)
        reuse.qs_vidx = qs_vidx
        self.reuse = reuse


        ############# forward
        pblock, mblock = self.block_list 
        vout, tout = pblock(frame, reuse)
        vout, tout = mblock(vout, tout, reuse)

        ######## loss
        loss = mblock.compute_loss(class_label, qs_vidx)
        loss_dict = mblock.loss_dict

        ######## evaluation
        pred_data = mblock.v_block.evaluate(class_label, qs_vidx)
        
        return loss, loss_dict, pred_data

    def save_model(self, fname):
        torch.save(self.state_dict(), fname)

    def refine_pseudo_label_with_timestamp_labels(self, mblock: MatchingBlock, pseudo_label_prediction):
        for i, (p, l) in enumerate(pseudo_label_prediction):
            if i == 0: # first is query video, skip
                continue   

            # randomly sample a timestamp label per segment
            l = l.detach().cpu().numpy()
            segs = utils.parse_label(l)
            keysteps = [ (s.action, min(s.start + np.random.randint(0, s.len), s.end)) for s in segs ]

            # refine pseudo label based on timestamp labels
            feature = mblock.t_block.frame.sequences[:, i] # T, C
            T = mblock.t_block.frame.lens[i]
            feature = feature[:T]
            simi = (feature[:-1] * feature[1:]).sum(-1)

            new_pred_seg = []
            new_pred_label = []
            for j, (a, m) in enumerate(keysteps):
                if j == 0:
                    start = 0
                else:
                    start = new_pred_seg[-1][2]
                
                if j == len(keysteps)-1:
                    end = T.item()
                else:
                    m1, m2 = m, keysteps[j+1][1]
                    simi_seg = simi[m1:m2]
                    end = m1 + simi_seg.argmin().item()

                new_pred_seg.append((a, start, end))
                new_pred_label.extend([a] * (end-start))

            new_p = p.new_tensor(new_pred_label)
            pseudo_label_prediction[i] = (new_p, l)

        return pseudo_label_prediction

    @torch.no_grad()
    def inference(self, tasks, frame: CombinedSequence, 
                 transcript: CombinedSequence, class_label: CombinedSequence, segment_label: CombinedSequence, 
                 qs_vidx=None,
                 mode='fewshot_full_label',
                 compute_loss=False,
                 ):

        ############## prepare input
        reuse = REUSE() # create a object to contain reusable variables and pass them through blocks
        reuse.transcript = transcript
        reuse.segment_label = segment_label
        reuse.cls_label = class_label
        reuse.tasks = tasks
        if qs_vidx is None:
            qs_vidx = self.generate_qs_vidx(tasks)
        reuse.qs_vidx = qs_vidx
        self.reuse = reuse

        ############# forward prototype building block
        pblock, mblock = self.block_list 

        if mode == 'fewshot_full_label':
            vout, tout = pblock(frame, reuse) 
            vout, tout = mblock(vout, tout, reuse)
            pred_data = mblock.v_block.evaluate(class_label, qs_vidx)

        elif mode == 'zeroshot':
            query_vidx = [ e[0] for e in qs_vidx ]
            query_frame = frame.clone()
            query_frame.sequences = query_frame.sequences[query_vidx]
            query_frame.masks = query_frame.masks[query_vidx]
            query_frame.lens = query_frame.lens[query_vidx]
            tout = pblock.textual_only_forward(query_frame, reuse) 
            tout = mblock.textual_only_forward(tout, reuse)
            pred_data = mblock.t_block.evaluate(class_label, qs_vidx)

        elif mode in ['fewshot_no_label', 'fewshot_weak_label']:
            # estimate pseudo labels for support videos first using textual modality only
            ## for each support video, use it as query while the other videos (include itself) as supports 
            ## and predict label in zero-shot mode
            new_qs_vidx = []
            for t in range(self.cfg.nt):
                for i in range(self.cfg.nv-1):
                    e = deepcopy(reuse.qs_vidx[0])
                    e[0] = e[1][t][i]
                    if t != 0:
                        e[1].insert(0, e[1].pop(t))
                    new_qs_vidx.append(e)
            new_qs_vidx.insert(0, reuse.qs_vidx[0])
            reuse.qs_vidx = new_qs_vidx

            tout = pblock.textual_only_forward(frame.clone(), reuse) 
            tout = mblock.textual_only_forward(tout, reuse)
            pseudo_label_prediction = mblock.t_block.evaluate(class_label, new_qs_vidx, use_pred_task=False)
            pseudo_label_prediction = pseudo_label_prediction[1] 

            # construct new inputs with pseudo labels
            if mode == 'fewshot_weak_label':
                pseudo_label_prediction = self.refine_pseudo_label_with_timestamp_labels(mblock, pseudo_label_prediction)

            clist, tlist, slist = [], [], []
            for i, (p, l) in enumerate(pseudo_label_prediction):
                clist.append(p)
                trans, seg = class_label_to_segment_label(p.detach().cpu().numpy())
                trans = p.new_tensor(trans)
                seg = p.new_tensor(seg)
                tlist.append(trans)
                slist.append(seg)
            reuse_pseudo_label = REUSE()
            reuse_pseudo_label.cls_label = CombinedSequence.create_from_sequences(clist, torch.long)
            reuse_pseudo_label.transcript = CombinedSequence.create_from_sequences(tlist, torch.long)
            reuse_pseudo_label.segment_label = CombinedSequence.create_from_sequences(slist, torch.long)
            reuse_pseudo_label.tasks = reuse.tasks
            reuse_pseudo_label.qs_vidx = qs_vidx

            # inference again with pseudo labels
            vout, tout = pblock(frame.clone(), reuse_pseudo_label)
            vout, tout = mblock(vout, tout, reuse_pseudo_label)
            pred_data = mblock.v_block.evaluate(class_label, qs_vidx)

        else:
            raise NotImplementedError(f'Unknown inference mode: {mode}')

        if compute_loss:
            loss = mblock.compute_loss(class_label, qs_vidx)
            loss_dict = mblock.loss_dict
        else:
            loss = 0
            loss_dict = {}
        
        return loss, loss_dict, pred_data
