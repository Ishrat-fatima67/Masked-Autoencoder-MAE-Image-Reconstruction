"""
model.py  —  Masked Autoencoder  (MAEViT)
==========================================
Exact architecture used during training.
Checkpoint key used by the notebook: 'model_state'

Load example:
    from model import MAEViT
    model = MAEViT()
    ckpt  = torch.load('mae_best.pt', map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model.eval()
"""

import math
import torch
import torch.nn as nn


# ── Building blocks ────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden    = int(dim * mlp_ratio)
        self.fc1  = nn.Linear(dim, hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio=mlp_ratio)

    def forward(self, x):
        x_norm      = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.grid_size   = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                    # (B, embed_dim, grid, grid)
        x = x.flatten(2).transpose(1, 2)   # (B, num_patches, embed_dim)
        return x


# ── Full MAE model ─────────────────────────────────────────────────────────────

class MAEViT(nn.Module):
    """
    Asymmetric Masked Autoencoder with Vision Transformer backbone.

    Encoder : ViT-Base  (dim=768, depth=12, heads=12)  ~86 M params
    Decoder : ViT-Small (dim=384, depth=12, heads=6)   ~22 M params

    Encoder sees ONLY the visible 25% of patches.
    Decoder receives encoded tokens + learnable mask tokens for the 75% masked.
    Loss is MSE computed only on masked patches.
    """

    def __init__(
        self,
        img_size:   int   = 224,
        patch_size: int   = 16,
        in_chans:   int   = 3,
        enc_dim:    int   = 768,
        enc_depth:  int   = 12,
        enc_heads:  int   = 12,
        dec_dim:    int   = 384,
        dec_depth:  int   = 12,
        dec_heads:  int   = 6,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.in_chans   = in_chans
        self.mask_ratio = mask_ratio

        self.patch_embed  = PatchEmbed(img_size, patch_size, in_chans, enc_dim)
        self.num_patches  = self.patch_embed.num_patches        # 196
        self.patch_dim    = patch_size * patch_size * in_chans  # 768

        # Encoder
        self.enc_pos_embed  = nn.Parameter(torch.zeros(1, self.num_patches, enc_dim))
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(enc_dim, enc_heads) for _ in range(enc_depth)
        ])
        self.enc_norm = nn.LayerNorm(enc_dim)

        # Bridge
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))

        # Decoder
        self.dec_pos_embed  = nn.Parameter(torch.zeros(1, self.num_patches, dec_dim))
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(dec_dim, dec_heads) for _ in range(dec_depth)
        ])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.enc_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token,    std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Patch utilities ───────────────────────────────────────────────────────

    def patchify(self, imgs):
        """(B, C, H, W) → (B, num_patches, patch_dim)"""
        p = self.patch_size
        b, c, h, w = imgs.shape
        x = imgs.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.reshape(b, (h // p) * (w // p), p * p * c)

    def unpatchify(self, patches):
        """(B, num_patches, patch_dim) → (B, C, H, W)"""
        p = self.patch_size
        b, n, _ = patches.shape
        h = w = int(math.sqrt(n))
        x = patches.reshape(b, h, w, p, p, self.in_chans)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(b, self.in_chans, h * p, w * p)

    # ── Masking ───────────────────────────────────────────────────────────────

    def random_masking(self, x):
        """Keep (1-mask_ratio) random patches per sample."""
        b, n, d  = x.shape
        len_keep = int(n * (1.0 - self.mask_ratio))   # default: 49

        noise       = torch.rand(b, n, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_vis    = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, d))

        mask = torch.ones([b, n], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_vis, mask, ids_restore

    # ── Sub-passes ────────────────────────────────────────────────────────────

    def forward_encoder(self, imgs):
        x = self.patch_embed(imgs) + self.enc_pos_embed
        x_vis, mask, ids_restore = self.random_masking(x)
        for blk in self.encoder_blocks:
            x_vis = blk(x_vis)
        x_vis = self.enc_norm(x_vis)
        return x_vis, mask, ids_restore

    def forward_decoder(self, x_vis, ids_restore):
        x = self.enc_to_dec(x_vis)
        b, n_vis, d = x.shape
        n_full = ids_restore.shape[1]

        mask_tokens = self.mask_token.repeat(b, n_full - n_vis, 1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, 1,
            ids_restore.unsqueeze(-1).repeat(1, 1, d)
        )
        x_full = x_full + self.dec_pos_embed

        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        x_full = self.dec_norm(x_full)
        return self.dec_pred(x_full)

    def forward_loss(self, imgs, pred, mask):
        target         = self.patchify(imgs)
        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)
        loss           = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1.0)
        return loss, target

    # ── Full forward ──────────────────────────────────────────────────────────

    def forward(self, imgs):
        """
        Args:
            imgs : (B, 3, H, W)  float32 in [0, 1]
        Returns:
            loss   — scalar MSE on masked patches
            pred   — (B, N, patch_dim)  reconstructed patches
            mask   — (B, N)  1 = masked, 0 = visible
            target — (B, N, patch_dim)  ground-truth patches
        """
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred                      = self.forward_decoder(latent, ids_restore)
        loss, target              = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, target
