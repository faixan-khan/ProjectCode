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

import numpy as np

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from util.gaussian_sampler import mixture_gaussians_1D, generate_binary_mask_1D, sample_indices_1D

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., num_gaussians= 5, gaussian_size = 7, 
                 norm_layer=nn.LayerNorm, norm_pix_loss=False):
        """
        - img_size: the width and height of the image [ex: 224 for 224x224]
        - patch_size: the dimension in pixels of a single patch [ex: 16 for 16x16]
        - in_chans: the number of input channels
        - embed_dim: the embedding dimension for the encoder
        - depth: the encoder depth
        - num_heads: the encoder number of heads
        - decoder_embed_dim: the embedding dimension for the decoder
        - decoder_depth: the decoder depth
        - decoder_num_heads: the decoder number of heads
        - mlp_ratio: the mlp ratio
        - num_gaussians: the number of generated gaussians for each image
        - gaussian_size: the diameter of a single gaussian in terms of patches
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.__num_gaussians = num_gaussians
        self.__gaussian_size = gaussian_size
        
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

    def gaussian_masking(self, x, num_samples):
        """
        Perform masking according to the random gaussian
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        h = w = int(L**.5)
        assert h * w == L
        
        # Generate window sizes; all are essentially of the same size
        window_sizes = torch.ones(self.__num_gaussians, dtype=torch.uint8)*self.__gaussian_size

        p_mask, means_mask = mixture_gaussians_1D(N,self.__num_gaussians, 
                                                  window_sizes, h, w)
        
        means_mask = means_mask.to(x.device)
        p_mask = p_mask.to(x.device)
        
        # --------------------------------------------------------------------------
        # Now we  have the  probability  distribution from which  we will sample the
        # reconstructed [p_mask], and the means to be used for reconstruction.  Yet,
        # we shuffle the data further, so the means are not always in order
        # --------------------------------------------------------------------------
        # means_idx = means_mask.nonzero()[:,1].reshape(N,-1)

        # Shuffle the means, 
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)


        means_mask = torch.gather(means_mask, dim=1, index=ids_shuffle)
        mask_sort_ids = torch.argsort(1-means_mask, dim=1)
        mask_ids_restore = torch.argsort(mask_sort_ids, dim=1)
        # --------------------------------------------------------------------------
        # Now, these indices represent the shuffled data
        means_idx = mask_sort_ids[:,:self.__num_gaussians]
        x = torch.gather(x, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, D))
        x_masked = torch.gather(x, dim=1, index=means_idx.unsqueeze(-1).repeat(1,
                                                                             1, D))
        
        # --------------------------------------------------------------------------
        # Now, sample the patches for reconstruction
        # --------------------------------------------------------------------------
        sample_idx = sample_indices_1D(p_mask, num_samples)
        # 1 -> for patches to be reconstructed
        mask = generate_binary_mask_1D(p_mask.shape, sample_idx)
        # --------------------------------------------------------------------------
        # now modify the values s.t 0 is [DOES NOT MATTER], 1 is remove
        mask = mask.to(x.device)
        # --------------------------------------------------------------------------
        # The below is just to make sure we did not sample a single patch for both
        # --------------------------------------------------------------------------
        means_mask = torch.gather(means_mask, dim=1, index=ids_restore)
        assert (mask*means_mask).sum() == 0

        return x_masked, mask_ids_restore, ids_restore, mask                                                  

    def forward_encoder(self, x, reconstruction_per_gaussian):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        num_samples = np.clip(int(self.__num_gaussians*reconstruction_per_gaussian),
                              1,x.shape[1]-self.__num_gaussians)

        # --------------------------------------------------------------------------
        # masking: get the patches used for reconstruction, two indices for unshuff- 
        # le, and the mask for our target tokens
        x, mask_ids_restore,ids_restore,mask = self.gaussian_masking(x, num_samples)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask_ids_restore, ids_restore, mask

    def forward_decoder(self, x, mask_ids_restore, ids_restore):
        
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # Back to the original shuffled
        x_ = torch.gather(x_, dim=1, index=mask_ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) 
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, reconstruction_per_gaussian=3,num_gaussians=5, gaussian_size=6):
        self.__num_gaussians = num_gaussians
        self.__gaussian_size = gaussian_size
        latent, mask_ids_restore, ids_restore, mask = self.forward_encoder(imgs,
                                                    reconstruction_per_gaussian)
        pred = self.forward_decoder(latent, mask_ids_restore, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
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
