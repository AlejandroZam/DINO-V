'''Code obtained from @ rishikksh20 on github: https://github.com/rishikksh20/ViViT-pytorch
@misc{arnab2021vivit,
      title={ViViT: A Video Vision Transformer}, 
      author={Anurag Arnab and Mostafa Dehghani and Georg Heigold and Chen Sun and Mario Lučić and Cordelia Schmid},
      year={2021},
      eprint={2103.15691},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}'''

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


  
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_frames, num_classes = 100, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, classifier=False, patch_type='2d', type='student'):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        if patch_type.lower() == '3d':
            num_frames = num_frames // 2
            self.to_patch_embedding = nn.Conv3d(3, dim, (2,16,16), (2,16,16))
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
                nn.Linear(patch_dim, dim),
            )
        self.patch_type = patch_type.lower()
        self.type = type

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        if classifier:
            self.linear_classifier = nn.Linear(dim, num_classes)
        self.classifier = classifier

    def forward(self, x):
        if self.patch_type == '3d':
            x = x.permute(0,2,1,3,4)
        #print(x.shape)
        x = self.to_patch_embedding(x)
        if self.patch_type == '3d':
            x = x.view(x.shape[0], x.shape[2], -1, x.shape[1])
        #print(x.shape)

        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        #print(f'Shape after adding cls token {x.shape}')
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        #print(f'Shape before space trans: {x.shape}')
        x, sp_attn = self.space_transformer(x)
        #print(f'Shape after space trans: {x.shape}')
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        #print(x.shape)

        sp_attn = rearrange(sp_attn[:, 0], '(b t) ... -> b t ...',b=b)
        #print(f'{self.type} attn {sp_attn.shape}')
        if self.type == 'student':
            cls_sp_attn = sp_attn[:,::2,0,1:]
        elif self.type == 'teacher':
            cls_sp_attn = sp_attn[:,:,0,1:]

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x, tm_attn = self.temporal_transformer(x)
        #print(f'Shape after time trans: {x.shape}')
        

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #print(x.shape)

        if self.classifier:
            x = self.linear_classifier(x)

        return x, cls_sp_attn #self.mlp_head(x)
    
    
    

if __name__ == "__main__":
    
    img = torch.ones([6, 16, 3, 224, 224])#.cuda()
    
    teacher = ViViT(224, 16, 16, 100, patch_type='2D', type='teacher')#.cuda()
    student = ViViT(224, 16, 16, 100, patch_type='2D', type='student')#.cuda()
    #parameters = filter(lambda p: p.requires_grad, model.parameters())
    #parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    #print('Trainable Parameters: %.3fM' % parameters)
    
    t_out, t_attn = teacher(img)
    st_out, st_attn = student(img)
    print(f't_attn: {t_attn.shape}')
    print(f'st_attn: {st_attn.shape}')
    
    te_attn_la = t_attn[:,:8,:]
    te_attn_lb = t_attn[:,8:,:]
    st_attn = st_attn[1*2:]

    te_attn = torch.stack([te_attn_la[0], te_attn_la[1], te_attn_lb[0], te_attn_lb[1]], dim=0).view(4, -1)
    st_attn = torch.stack([st_attn[0], st_attn[2], st_attn[1], st_attn[3]], dim=0).view(4, -1)

    print(f'final te attn {te_attn.shape}')
    print(f'final st attn {st_attn.shape}')
    
    print("Shape of out :", out.shape)      # [B, num_classes]

    
    
