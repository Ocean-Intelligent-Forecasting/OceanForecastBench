#Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import lru_cache
import time
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, trunc_normal_
from climax.swin_model import swin_tiny_patch4_window7_224,PatchEmbed
from climax.resnet import ResNet
from climax.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

class ClimaX(nn.Module):
    """Implements the ClimaX model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension 记得在yaml文件里改成128
        depth (int): number of transformer layers
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
    """

    def __init__(
        self,
        default_vars,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024, 
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.2,
        drop_rate=0.2,
        attn_drop=0.2,
        use_checkpoint=False
    ):
        super().__init__()
 
        # TODO: remove time_history parameter
        self.img_size = img_size #32*64
        self.patch_size = patch_size #2*2
        self.default_vars = default_vars
        self.use_checkpoint = use_checkpoint

        #self.patch_embed = PatchEmbed(patch_size, len(self.default_vars), embed_dim,norm_layer=nn.LayerNorm)
        # variable tokenization: separate embedding layer for each input variable
        # self.token_embeds = nn.ModuleList(
        #     [PatchEmbed(patch_size, 1, embed_dim) for i in range(len(default_vars))]
        # )
        #self.num_patches = self.token_embeds[0].num_patches #512

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.var_map = self.create_var_embedding(embed_dim)

        self.swin_model = ResNet(in_channels=96, out_channels=94,n_blocks=18,hidden_channels=192, dropout=0.1)
                            
        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim): # 为每个变量生成一个embedding，表示每个embedding属于哪个变量
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]


    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
        if not torch.jit.is_scripting() and self.use_checkpoint:
            x = checkpoint.checkpoint(self.swin_model, x, use_reentrant=False)
        else:
            x = self.swin_model(x) 
        return x

    def forward(self, x, y, lead_times, variables, out_variables, metric, lat,mask):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        preds = self.forward_encoder(x, lead_times, variables)  # B, L, D 
        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat, mask) for m in metric]
        #end = time.time()
        #print("计算时间为(ms)：",(end-start)*1000)
        return loss, preds

    def evaluate(self, x, y, lead_times, variables, out_variables, transform, metrics, lat, log_postfix,mask):
        _, preds = self.forward(x, y, lead_times, variables, out_variables, metric=None, lat=lat,mask=mask)
        return [m(preds, y, transform, out_variables, lat, log_postfix,mask) for m in metrics]
    
