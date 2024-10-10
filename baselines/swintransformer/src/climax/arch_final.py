# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import lru_cache
import time
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, trunc_normal_
from climax.swin_model import swin_tiny_patch4_window7_224,PatchEmbed
from climax.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)


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
        attn_drop=0.2
    ):
        super().__init__()
 
        # TODO: remove time_history parameter
        self.img_size = img_size #32*64
        self.patch_size = patch_size #2*2
        self.default_vars = default_vars
        #self.patch_embed = PatchEmbed(patch_size, len(self.default_vars), embed_dim,norm_layer=nn.LayerNorm)
        # variable tokenization: separate embedding layer for each input variable
        # self.token_embeds = nn.ModuleList(
        #     [PatchEmbed(patch_size, 1, embed_dim) for i in range(len(default_vars))]
        # )
        #self.num_patches = self.token_embeds[0].num_patches #512

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.var_map = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        # 可学习的query
        # self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        # 输出为batch*seq*feature
        # self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding and lead time embedding
        # 随机生成位置变量
        #self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        # self.lead_time_embed = nn.Linear(1, embed_dim)
        self.swin_model = swin_tiny_patch4_window7_224(in_chans = len(self.default_vars),
                                                       embed_dim = embed_dim, 
                                                       img_size=self.img_size,
                                                       patch_first = self.patch_size,
                                                       drop_rate = drop_rate,
                                 					   drop_path = drop_path,
                                                       attn_drop_rate = attn_drop)
                            
        # --------------------------------------------------------------------------

        # ViT backbone
        #self.pos_drop = nn.Dropout(p=drop_rate)
        #dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        # self.blocks = nn.ModuleList(
        #     [
        #         Block(
        #             embed_dim,
        #             num_heads,
        #             mlp_ratio,
        #             qkv_bias=True,
        #             drop_path=dpr[i],
        #             norm_layer=nn.LayerNorm,
        #             drop=drop_rate,
        #         )
        #         for i in range(depth)
        #     ]
        # )
        #self.norm = nn.LayerNorm(embed_dim)

        # --------------------------------------------------------------------------

        # prediction head
        # self.head = nn.ModuleList()
        # for _ in range(decoder_depth):
        #     self.head.append(nn.Linear(embed_dim, embed_dim))
        #     self.head.append(nn.GELU())
        # self.head.append(nn.Linear(embed_dim, len(self.default_vars) * patch_size**2))
        # self.head = nn.Sequential(*self.head)
		
        if len(default_vars)>50:
            self.recovery = nn.ConvTranspose2d(embed_dim*2, len(default_vars)-2,kernel_size=patch_size,stride=patch_size)
        else:
            self.recovery = nn.ConvTranspose2d(embed_dim*2, len(default_vars)-1,kernel_size=patch_size,stride=patch_size)
        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        #pos_embed = get_2d_sincos_pos_embed(
       #     self.pos_embed.shape[-1],
       #     int(self.img_size[0] / self.patch_size),
       #     int(self.img_size[1] / self.patch_size),
       #     cls_token=False,
       # )
     #   self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        #var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
        #self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        # token embedding layer
        # for i in range(len(self.token_embeds)):
        #     w = self.token_embeds[i].proj.weight.data
        #     trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
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

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = len(self.default_vars)
        #pad_input = (self.img_size[0] % p != 0) or (self.img_size[1] % p != 0)
       
        h = int(np.ceil(self.img_size[0] / p)) 
        w = int(np.ceil(self.img_size[1] / p))

        #h = self.img_size[0] // p if h is None else h // p
        #w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        #print("final_shape:",x.shape)
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        #print("imgs_shape1:",imgs.shape)
        imgs = imgs[:,:,:self.img_size[0],:self.img_size[1]]
        #print("imgs_shape2:",imgs.shape)
        return imgs

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D     batch*V个变量*patch数目*维度
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x) #B*L*V*D
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0) #BxL, 1, D
        x, _ = self.var_agg(var_query, x, x)  # 输入query key value   BxL,1,D 
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward_encoder(self, x: torch.Tensor):
        # x: `[B, V, H, W]` shape.
        #x = self.patch_embed(x)
        #x = self.pos_drop(x) # dropout embedded patches B,L,D
        # print(x.shape)
        x = self.swin_model(x) 
        # apply Transformer blocks
        # for blk in self.blocks:
        #     x = blk(x)
        # x = self.norm(x)

        return x

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        #start = time.time()
        
        out_transformers = self.forward_encoder(x)  # B, L, D 
        #preds = self.head(out_transformers)  # B, L, V*p*p
		
        #preds = self.unpatchify(preds) # B, V, H, W
        #print("preds_shape:",preds.shape)
        #out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        #preds = preds[:, out_var_ids]
        
        #preds = self.head(out_transformers)  # B, L, V*p*p
        #preds = self.unpatchify(preds) # B, V, H, W
        B, _, C = out_transformers.shape
        H = int(np.ceil(self.img_size[0] / self.patch_size)) 
        W = int(np.ceil(self.img_size[1] / self.patch_size))
        preds = out_transformers.reshape(B, H, W, C).permute(0,3,1,2)
        preds = self.recovery(preds)
        preds = preds[:,:,:self.img_size[0],:self.img_size[1]]
        
		
        # @yinjun 20230606
        # skip loss on land
        #mask = (y != -32767)
        return preds

    def evaluate(self, x, y, lead_times, variables, out_variables, transform, metrics, lat, log_postfix,mask):
        _, preds = self.forward(x, y, lead_times, variables, out_variables, metric=None, lat=lat,mask=mask)
        return [m(preds, y, transform, out_variables, lat, log_postfix,mask) for m in metrics]
    
