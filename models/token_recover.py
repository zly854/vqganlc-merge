import torch
import torch.nn as nn


class Cross_Attention(nn.Module):
    def __init__(self, dim = 192, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., length=27):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2, x_3):
        B, N, C = x_1.shape # 64,196,192
        B, N_1, C = x_3.shape #64,192,10

        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x_2).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(x_3).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class k2l_recovery(nn.Module):
    def __init__(self, num_merge_tokens=77, num_tokens=256, embed_dim=192, num_heads=8):
        super().__init__()
        self.cross_att = Cross_Attention(dim=embed_dim, num_heads=num_heads,
                                         qkv_bias=True, attn_drop=0., proj_drop=0.)
        self.x_token = nn.Parameter(torch.zeros(num_tokens, embed_dim))

    def forward(self,x):
        bs, _, _ = x.shape
        self.x_token_expand = self.x_token.unsqueeze(0)
        self.x_token_expand = self.x_token_expand.repeat(bs, 1, 1)
        output = self.x_token_expand + self.cross_att(self.x_token_expand, x, x)
        return output
