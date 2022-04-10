import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            if i < len(self.layers)-1:
                x = attn(x) + x
                x = ff(x) + x
            else:
                x1, attn_map = attn(x, return_attn=True)
                #print(f'Attention Map size: {attn_map.shape}')
                x = x1 + x
                x = ff(x) + x
        return self.norm(x), attn_map

# dim features per token, depths = layer, heads is muti headed attention
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_frames, num_classes=100, dim=192, depth=4, heads=3, pool='cls',
                 in_channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0., scale_dim=4, ):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.to_3d_patches = nn.Conv3d(3, dim, (2, 16, 16), (2, 16, 16))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames // 2, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        print(x.shape)
        # x = self.to_patch_embedding(x)
        x = self.to_3d_patches(x)
        x = x.view(x.shape[0], x.shape[2], -1, x.shape[1])
        print(x.shape)

        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        print(f'Shape after adding cls token {x.shape}')
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        print(f'Shape before space trans: {x.shape}')
        x, sp_attn = self.space_transformer(x)
        print(f'Shape after space trans: {x.shape}')
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        print(x.shape)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x, tm_attn = self.temporal_transformer(x)
        print(f'Shape after time trans: {x.shape}')
        # batch , num frames + cls , features
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        print(x.shape)

        return x  # self.mlp_head(x)