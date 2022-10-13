import torch
import torch.nn as nn


# faizan's linear attention
from fast_transformers.attention import LinearAttention
from fast_transformers.masking import LengthMask, TriangularCausalMask, FullMask
from fast_transformers.attention import AttentionLayer
from fast_transformers.attention import FullAttention


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.Linear_attention = LinearAttention(dim)
        self.linear_attention = AttentionLayer(self.Linear_attention, dim, num_heads)
        # self.linear_attention = AttentionLayer(LinearAttention, dim, num_heads)
        print(self.linear_attention)

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x).reshape(8, 50, 3, 768).permute(2,0,1,3)
        # print(self.qkv(x).reshape(B, N, 3).shape)
        # exit()
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        print(q.shape, k.shape, v.shape, self.num_heads)
        attn_mask = FullMask(C, device=x.device)
        key_lengths = LengthMask(x.new_full((B,), N, dtype=torch.int64), device=x.device)
        print(attn_mask.shape, key_lengths.shape)
        linear_attn = self.linear_attention(q, k, v, attn_mask, key_lengths, key_lengths)
        print(linear_attn.shape)





        exit()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn@v
        if self.rpe_v is not None:
            out += self.rpe_v(attn)
        print(out.shape)
        exit()


        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        print('Init Done')
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


print(torch.cuda.current_device())

bloc = Block(dim=768, num_heads=12)
x = torch.rand(
    8,  # batch size 
    50, # sequence length  
    768  # feature dimensions
    )
cuda = torch.device('cuda')
bloc = bloc.to(cuda)
x=x.to(cuda)
out = bloc(x)
