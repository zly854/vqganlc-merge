import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
class Decoder(nn.Module):
    def __init__(self, embed_dim=192, num_tokens=196, img_size=224, num_heads=4, num_layers=4, decoder_embed_dim = 128,patch_size = 16 ):
        super(Decoder, self).__init__()
        self.patch_size = patch_size
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_num = self.img_size // self.patch_size
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, self.decoder_embed_dim))
        self.conv_out = torch.nn.Conv2d(3,
                                        3,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.decoder_blocks = nn.ModuleList([
            Block(self.decoder_embed_dim, num_heads, 4, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(num_layers)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3,bias = True)
    def forward(self, x):
        x = self.decoder_embed(x)
        x = x + self.pos_embedding
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        bs, _ , _ = x.shape
        x = x.view(bs,self.num_tokens,3,self.patch_size,self.patch_size)
        x = x.permute(0,2,1,3,4)
        x = x.contiguous().view(bs,3,self.patch_num,self.patch_num,self.patch_size,self.patch_size)
        img = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(bs, 3, self.img_size, self.img_size)
        img = self.conv_out(img)
        return img