import torch
import torch.nn as nn 
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import Attention, Block, PatchEmbed
import argparse


class Encoder(nn.Module):
    def __init__(self,*, image_size = 224,embed_dim = 192,patch_size = 16,num_layers = 12,num_heads = 4):
        super(Encoder,self).__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.token_num = (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbed(
            img_size = image_size,patch_size = patch_size,in_chans = 3, embed_dim = embed_dim
            )
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        self.blocks = nn.Sequential(
            *[Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=0.0,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU,
            ) for _ in range(num_layers)]
        )
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.token_num + 1, embed_dim))     
    def forward(self,x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self.pos_embed.expand(batch_size,-1,-1)
        pos_embed = nn.Parameter(pos_embed)
        x = x + pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        #x = self.norm(x)
        return x