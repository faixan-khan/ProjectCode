# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # LOMAR
        self.img_size = img_size
        self.patch_size =  patch_size

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def sample_patch_index(self, x, patch_index, keep_ratio):
        N, H, W, D = x.shape
        M,P = patch_index.shape
        patch_index = patch_index.unsqueeze(0).expand(N,M,P)

        noise = torch.rand(N,M,P, device=patch_index.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise,dim=-1)  # ascend: small is keep, large is remove

        ids_keep = ids_shuffle[:,:,:keep_ratio]
        ids_unmasked = ids_shuffle[:,:,keep_ratio:]

        patch_keeps = torch.gather(patch_index, -1, ids_keep)
        patch_unmasked = torch.gather(patch_index, -1, ids_unmasked)

        return patch_keeps, patch_unmasked, ids_shuffle

    def generate_window_patches(self, x, left, top, window_size, mask_ratio, neigh_ratio):
        
        N, H, W, D = x.shape
        window_number = left.shape[0]
        
        #  extract the windows based on the coordinates
        left = left.unsqueeze(-1).expand(window_number,window_size)
        top  = top.unsqueeze(-1).expand(window_number, window_size)

        row = torch.arange(0,window_size,device=x.device).unsqueeze(0).expand(window_number,window_size)+left
        column = torch.arange(0,window_size*W,W, device = x.device).unsqueeze(0).expand(window_number, window_size)+top*W        

        in_window_mask_number = int(window_size*window_size*mask_ratio)  

        assert in_window_mask_number>=1
        in_window_patches =row.unsqueeze(1).expand(window_number,window_size,window_size)  + column.unsqueeze(-1).expand(left.shape[0],window_size,window_size)
        in_window_patches = in_window_patches.view(window_number,-1)

        # For neighbourhood
        if neigh_ratio>0:
            indices_neigh = torch.arange(0,W*H,device=x.device).expand(window_number,W*H)
            indices_neigh = indices_neigh.scatter(-1,in_window_patches,-1)
            indices_neigh,_ = torch.sort(indices_neigh,dim=-1)
            indices_neigh = indices_neigh.expand(N,window_number,W*H)
            indices_neigh_keep = indices_neigh[:,:,(window_size*window_size):]
            noise_neigh = torch.rand(N,window_number,indices_neigh_keep.size(2),device=x.device)
            ids_neigh_shuffle = torch.argsort(noise_neigh,dim=-1)
            out_window_mask_number = int((window_size*window_size)*neigh_ratio)
            ids_keep_neigh = ids_neigh_shuffle[:,:,:out_window_mask_number]
            neigh_keeps = torch.gather(indices_neigh_keep, -1, ids_keep_neigh)
        else:
            neigh_keeps=[0]

        # sample the masked patch ids
        ids_mask_in_window, ids_unmask_in_window, ids_restore = self.sample_patch_index(x,in_window_patches,in_window_mask_number)

        patches_to_keep = in_window_patches.unsqueeze(0).expand(N, window_number,window_size* window_size)
        x = x.view(N,H*W,D).unsqueeze(0).repeat(window_number,1, 1,1).view(N*window_number,H*W,D)

        sorted_patch_to_keep,_ = torch.sort(patches_to_keep,dim=-1)
        sorted_patch_to_keep = sorted_patch_to_keep.view(N*window_number,-1)
        ids_restore = ids_restore.view(N*window_number,-1)


        ids_mask_in_window = ids_mask_in_window.view(N*window_number, -1)
        ids_unmask_in_window = ids_unmask_in_window.view(N*window_number, -1)

        mask_indices = ((sorted_patch_to_keep.unsqueeze(-1)- ids_mask_in_window.unsqueeze(1))==0).sum(-1)==1

        # x_masked = torch.gather(x, dim=1, index=ids_mask_in_window.unsqueeze(-1).repeat(1, 1, D)).clone()
        x_unmasked = torch.gather(x, dim=1, index=ids_unmask_in_window.unsqueeze(-1).repeat(1, 1, D)).clone()
        
        #faizan's code
        if neigh_ratio > 0:
            neigh_keeps = neigh_keeps.view(N*window_number,-1)
            x_neigh = torch.gather(x, dim=1, index=neigh_keeps.unsqueeze(-1).repeat(1, 1, D)).clone()
            x_unmasked =  torch.cat((x_unmasked, x_neigh), dim=1)

        return x_unmasked, sorted_patch_to_keep, mask_indices, ids_restore, neigh_keeps

    def forward_encoder(self, x, window_size, num_window, mask_ratio, neigh_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # faizan
        N, _, C = x.shape
        H = W = self.img_size // self.patch_size
        x= x.view(N,H,W,C)

        assert window_size<= H and window_size <=W

        # sample window coordinates
        rand_top_locations = torch.randperm(H-window_size+1,device=x.device)[:num_window]
        rand_left_locations = torch.randperm(W-window_size+1,device=x.device)[:num_window]

        x_unmasked, sorted_patch_to_keep, mask, ids_restore, neigh_indices = self.generate_window_patches(x, rand_left_locations, rand_top_locations, window_size, mask_ratio, neigh_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_unmasked.shape[0], -1, -1)
        x_unmasked = torch.cat((cls_tokens, x_unmasked), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x_unmasked = blk(x_unmasked)

        x_unmasked = self.norm(x_unmasked)
        out_window_mask_number = int((window_size*window_size)*neigh_ratio)
        if out_window_mask_number:
            x_unmasked, x_neigh = x_unmasked[:,:-out_window_mask_number,:], x_unmasked[:,-out_window_mask_number:,:]
        else:
            x_neigh = [0]

        return x_unmasked, sorted_patch_to_keep, mask, ids_restore, neigh_indices, x_neigh

    def forward_decoder(self, x, ids_restore, sorted_patch_to_keep, neigh_indices, x_neigh):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        if len(neigh_indices) > 1:
            x_neigh = self.decoder_embed(x_neigh)
            x_ = torch.cat([x_, x_neigh], dim=1)  # no cls token
            indices = torch.cat((sorted_patch_to_keep,neigh_indices), dim=1)
        else:
            indices = sorted_patch_to_keep

        # add pos embed
        x_ = x_ + torch.gather(self.decoder_pos_embed.repeat(x_.shape[0],1,1), dim=1, index = indices.unsqueeze(-1).repeat(1, 1, x_.shape[-1])+1)        
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        # remove cls token and neigh tokens
        x = x[:, 1:, :]
        x = x[:,:sorted_patch_to_keep.shape[1],:]

        return x

    def forward_loss(self, imgs, pred, mask, num_window, sorted_patch_to_keep):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """


        target = self.patchify(imgs)
        N,P,H = target.shape
        target = target.unsqueeze(0).repeat(num_window,1,1,1).view(-1,P,H)
        target = torch.gather(target,dim=1, index=sorted_patch_to_keep.unsqueeze(-1).repeat(1, 1, target.shape[-1]))


        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # print(loss.mean(),'ok',mask.sum())
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # print(loss.mean(),'jj')
        # exit()
        return loss

    def forward(self, imgs, window_size, num_window, mask_ratio=0.75, neigh_ratio=0.05):
        latent, sorted_patch_to_keep, mask, ids_restore, neigh_indices, x_neigh = self.forward_encoder(imgs, window_size, num_window, mask_ratio, neigh_ratio)
        pred = self.forward_decoder(latent, ids_restore, sorted_patch_to_keep, neigh_indices, x_neigh)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask, num_window, sorted_patch_to_keep)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
