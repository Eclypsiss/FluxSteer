from typing import Dict
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.model.utils import timestep_embedding
from models.model.utils import ResBlock, SpatialTransformer
from models.model.scene_model import create_scene_model
from models.base import MODEL
import numpy as np
import matplotlib.pyplot as plt
@MODEL.register()
class UNetModel(nn.Module):
    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        super(UNetModel, self).__init__()
        self.collected_points = []
        self.d_x = cfg.d_x
        self.d_model = cfg.d_model
        self.nblocks = cfg.nblocks
        self.resblock_dropout = cfg.resblock_dropout
        self.transformer_num_heads = cfg.transformer_num_heads
        self.transformer_dim_head = cfg.transformer_dim_head
        self.transformer_dropout = cfg.transformer_dropout
        self.transformer_depth = cfg.transformer_depth
        self.transformer_mult_ff = cfg.transformer_mult_ff
        self.context_dim = cfg.context_dim
        self.use_position_embedding = cfg.use_position_embedding # for input sequence x
        self.use_llm = cfg.use_llm
        self.cfg_dropout_prob = cfg.get('cfg_dropout_prob', 0.1) # Default 10% dropout
        ## create scene model from config
        self.scene_model_name = cfg.scene_model.name
        scene_model_in_dim = 3 + int(cfg.scene_model.use_color) * 3 + int(cfg.scene_model.use_normal) * 3
        if cfg.scene_model.name == 'PointNet':
            scene_model_args = {'c': scene_model_in_dim, 'num_points': cfg.scene_model.num_points,
                                'num_tokens': cfg.scene_model.num_tokens}
        else:
            scene_model_args = {'c': scene_model_in_dim, 'num_points': cfg.scene_model.num_points}
        self.scene_model = create_scene_model(cfg.scene_model.name, **scene_model_args)
        if self.use_llm:
            from models.model.text_model import BERTEmbedder
            self.text_model = BERTEmbedder(n_embed=512, n_layer=32)
        ## load pretrained weights
        weight_path = cfg.scene_model.pretrained_weights_slurm if slurm else cfg.scene_model.pretrained_weights
        if weight_path is not None:
            self.scene_model.load_pretrained_weight(weigth_path=weight_path)
        if cfg.freeze_scene_model:
            for p in self.scene_model.parameters():
                p.requires_grad_(False)

        time_embed_dim = self.d_model * cfg.time_embed_mult
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.in_layers = nn.Sequential(
            nn.Conv1d(self.d_x, self.d_model, 1)
        )

        self.layers = nn.ModuleList()
        for i in range(self.nblocks):
            self.layers.append(
                ResBlock(
                    self.d_model,
                    time_embed_dim,
                    self.resblock_dropout,
                    self.d_model,
                )
            )
            self.layers.append(
                SpatialTransformer(
                    self.d_model, 
                    self.transformer_num_heads, 
                    self.transformer_dim_head, 
                    depth=self.transformer_depth,
                    dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff,
                    context_dim=self.context_dim,
                )
            )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, 1),
        )
        
        # Add a learnable embedding for unconditional generation
        self.uncond_embedding = nn.Parameter(torch.randn(1, 1, self.context_dim))
        
    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """ 应用模型到输入批次

        Args:
            x_t: 输入数据, <B, C> or <B, L, C>
            ts: 时间步, (B,) or (B, 2) 的一维或二维张量。
                对于 MeanFlow, ts 是 (B, 2)，其中 ts[:, 0] 是 t, ts[:, 1] 是 r.
                对于 Diffusion, ts 是 (B,)，代表当前时间点.
            cond: 条件特征
        
        Return:
            去噪后的目标数据, i.e., $x_{t-1}$
        """
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1)
        assert len(x_t.shape) == 3

        ## 时间编码
        if len(ts.shape) == 1: # 兼容原始的 Diffusion 调用
            t_emb = timestep_embedding(ts, self.d_model)
        else: # 为 MeanFlow 处理 (t, r)
            t_emb = timestep_embedding(ts[:, 0], self.d_model) # 对 t 进行编码
            r_emb = timestep_embedding(ts[:, 1], self.d_model) # 对 r 进行编码
            t_emb = t_emb + r_emb # 将 r 和 t 的编码信息相加

        t_emb = self.time_embed(t_emb)

        h = rearrange(x_t, 'b l c -> b c l')
        h = self.in_layers(h) # <B, d_model, L>
        # print(h.shape, cond.shape) # <B, d_model, L>, <B, T , c_dim>

        ## prepare position embedding for input x
        if self.use_position_embedding:
            B, DX, TX = h.shape
            pos_Q = torch.arange(TX, dtype=h.dtype, device=h.device)
            pos_embedding_Q = timestep_embedding(pos_Q, DX) # <L, d_model>
            h = h + pos_embedding_Q.permute(1, 0) # <B, d_model, L>

        for i in range(self.nblocks):
            h = self.layers[i * 2 + 0](h, t_emb)
            h = self.layers[i * 2 + 1](h, context=cond)
        h = self.out_layers(h)
        h = rearrange(h, 'b c l -> b l c')

        ## reverse to original shape
        if in_shape == 2:
            h = h.squeeze(1)

        return h

    def condition(self, data: Dict, force_uncond: bool = False) -> torch.Tensor:
        """ Obtain scene feature with scene model

        Args:
            data: dataloader-provided data
            force_uncond: If True, force return unconditional embedding.
        
        Return:
            Condition feature
        """
        b = data.get('offset', data.get('pos')).shape[0] # Handle different data shapes
        
        # For CFG, force unconditional or randomly drop condition during training
        if force_uncond or (self.training and torch.rand(1).item() < self.cfg_dropout_prob):
            return self.uncond_embedding.repeat(b, 1, 1)

        if self.scene_model_name == 'PointTransformer':
            pos, feat, offset = data['pos'], data['feat'], data['offset']
            p5, x5, o5 = self.scene_model((pos, feat, offset))
            scene_feat = rearrange(x5, '(b n) c -> b n c', b=b, n=self.scene_model.num_groups)
            if self.use_llm:
                text_embedding = self.text_model(data['text'])[:,0,:]   # Get [CLS] embedding (all x 77 x 512) -> (all x 512)
                batch_embedding = torch.split(text_embedding, data['sentence_cnt']) # tuple of splitted embedding (? x 512)
                text_feat = torch.cat([a.max(dim=0,keepdim=True)[0] for a in batch_embedding], dim=0).reshape(b,1,-1) # (b,1,512) 
                scene_feat = torch.cat([text_feat, scene_feat], dim=1)

        elif self.scene_model_name == 'PointNet':
            pos = data['pos'].to(torch.float32)
            scene_feat = self.scene_model(pos).reshape(b, self.scene_model.num_groups, -1)
        elif self.scene_model_name == 'PointNet2':
            pos = data['pos'].to(torch.float32)
            _, scene_feat_list = self.scene_model(pos)
            scene_feat = scene_feat_list[-1].transpose(1, 2)
        else:
            raise Exception('Unexcepted scene model.')

        return scene_feat
