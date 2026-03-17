"""
app.py  —  Gradio App for MAE Image Reconstruction
====================================================
Deployment: Hugging Face Spaces (Gradio SDK)

Folder structure on HF Space:
    app.py
    model.py
    checkpoints/
        mae_best.pt        ← your trained checkpoint from Kaggle

Run locally:
    pip install gradio torch torchvision pillow numpy
    python app.py
"""

import math
import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import MAEViT


# ── Config ─────────────────────────────────────────────────────────────────────

IMG_SIZE   = 224
PATCH_SIZE = 16

# Checkpoint path — works both locally and on HF Spaces
CKPT_PATH = Path(__file__).parent / "checkpoints" / "mae_best.pt"


# ── Model loading (cached so it only loads once) ───────────────────────────────

_model_cache = {}

def get_model():
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["device"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MAEViT(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=3,
        enc_dim=768, enc_depth=12, enc_heads=12,
        dec_dim=384, dec_depth=12, dec_heads=6,
        mask_ratio=0.75,
    )

    if CKPT_PATH.exists():
        print(f"Loading checkpoint from {CKPT_PATH} ...")
        ckpt = torch.load(str(CKPT_PATH), map_location=device)

        # Notebook saves under key 'model_state'
        state_dict = ckpt.get("model_state", ckpt)
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    else:
        print(f"WARNING: Checkpoint not found at {CKPT_PATH}.")
        print("Running with random weights — output will not be meaningful.")
        print("Place mae_best.pt inside a 'checkpoints/' folder next to app.py")

    model = model.to(device).eval()
    _model_cache["model"]  = model
    _model_cache["device"] = device
    return model, device


# ── Core inference ─────────────────────────────────────────────────────────────

def run_inference(pil_image: Image.Image, mask_ratio: float):
    """
    Given a PIL image and mask_ratio, returns three numpy images:
      masked_input  — original with masked patches zeroed out
      reconstruction — model's pixel prediction
      ground_truth   — resized original (what the model sees as target)
    """
    model, device = get_model()

    # Override mask ratio at inference time
    model.mask_ratio = float(mask_ratio)

    # Preprocess: resize → tensor → [0,1]
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),                   # → [0, 1]
    ])
    x = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)  # (1,3,224,224)

    with torch.no_grad():
        loss, pred, mask, _ = model(x)

        # Reconstruct image from predicted patches
        recon = model.unpatchify(pred).clamp(0, 1)   # (1,3,224,224)

        # Build masked input visualization (zero out masked patches)
        masked = x.clone()
        gh = gw = IMG_SIZE // PATCH_SIZE             # 14 x 14 grid
        mask_grid = mask.reshape(1, gh, gw)
        p = PATCH_SIZE
        for i in range(gh):
            for j in range(gw):
                if mask_grid[0, i, j] > 0.5:
                    masked[0, :, i*p:(i+1)*p, j*p:(j+1)*p] = 0.0

    def to_uint8(t):
        """(1,3,H,W) tensor → (H,W,3) uint8 numpy"""
        arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (arr * 255.0).clip(0, 255).astype(np.uint8)

    masked_np = to_uint8(masked)
    recon_np  = to_uint8(recon)
    gt_np     = to_uint8(x)

    n_total   = gh * gw
    n_masked  = int(mask_ratio * n_total)
    n_visible = n_total - n_masked

    info = (
        f"**Mask ratio:** {mask_ratio:.0%}  |  "
        f"**Visible patches:** {n_visible}/196  |  "
        f"**Masked patches:** {n_masked}/196"
    )

    return masked_np, recon_np, gt_np, info


# ── Gradio UI ──────────────────────────────────────────────────────────────────

def build_demo():
    with gr.Blocks(
        title="MAE — Masked Autoencoder Reconstruction",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            """
            # 🖼️ Masked Autoencoder (MAE) — Image Reconstruction
            **Self-Supervised Learning with Vision Transformers**

            Upload any image, choose a masking ratio, and see the MAE reconstruct
            the hidden patches from context alone.
            """
        )

        with gr.Row():
            # ── Left column: inputs ──────────────────────────────────────────
            with gr.Column(scale=1):
                in_img = gr.Image(
                    type="pil",
                    label="Upload Image",
                )
                mask_slider = gr.Slider(
                    minimum=0.10,
                    maximum=0.95,
                    value=0.75,
                    step=0.05,
                    label="Masking Ratio  (0.75 = 75% patches hidden)",
                )
                run_btn = gr.Button("🔍 Reconstruct", variant="primary")

                gr.Markdown(
                    """
                    ### ℹ️ Model Info
                    | Component | Config |
                    |-----------|--------|
                    | Encoder | ViT-Base (768-dim, 12 layers, 12 heads) |
                    | Decoder | ViT-Small (384-dim, 12 layers, 6 heads) |
                    | Patch size | 16 × 16 px |
                    | Image size | 224 × 224 px |
                    | Total patches | 196 (14×14 grid) |
                    | Default mask | 75% (147 patches hidden) |
                    """
                )

            # ── Right column: outputs ────────────────────────────────────────
            with gr.Column(scale=2):
                info_box = gr.Markdown("Upload an image and click Reconstruct.")

                with gr.Row():
                    out_masked = gr.Image(
                        type="numpy",
                        label="① Masked Input  (patches zeroed out)",
                    )
                    out_recon = gr.Image(
                        type="numpy",
                        label="② Model Reconstruction",
                    )
                    out_gt = gr.Image(
                        type="numpy",
                        label="③ Ground Truth  (resized original)",
                    )

        # ── Examples ─────────────────────────────────────────────────────────
        gr.Markdown("### 🧪 Try different mask ratios")
        gr.Examples(
            examples=[
                ["examples/sample1.jpg", 0.75],
                ["examples/sample1.jpg", 0.50],
                ["examples/sample1.jpg", 0.90],
            ],
            inputs=[in_img, mask_slider],
            outputs=[out_masked, out_recon, out_gt, info_box],
            fn=run_inference,
            cache_examples=False,
            label="Example inputs (add your own images to examples/ folder)",
        )

        # ── About section ─────────────────────────────────────────────────────
        with gr.Accordion("📚 How MAE works", open=False):
            gr.Markdown(
                """
                ### Masked Autoencoder — Step by Step

                1. **Patchify** — The 224×224 image is split into a 14×14 grid of 16×16 patches (196 total).
                2. **Random Mask** — A chosen fraction of patches is hidden. Default = 75% (147 patches hidden, 49 visible).
                3. **Encode** — Only the **visible** patches are passed to the large ViT-Base encoder.  
                   Mask tokens are **NOT** given to the encoder — this forces real learning.
                4. **Decode** — The lightweight ViT-Small decoder receives:  
                   • Encoded visible tokens (projected to 384-dim)  
                   • Learnable mask tokens for missing positions  
                   • Positional embeddings for all 196 positions  
                5. **Reconstruct** — The decoder outputs pixel values for all patches.
                6. **Loss** — MSE is computed **only on the masked patches** — the model must infer hidden content.

                ### Why is this useful?
                After pre-training, the encoder has learned **rich visual representations** that transfer
                efficiently to downstream tasks like image classification, object detection, and segmentation.

                ### Training details
                - Dataset: TinyImageNet (100K images, 200 classes)
                - Optimizer: AdamW (lr=1.5e-4, weight_decay=0.05)
                - Scheduler: Cosine Annealing
                - Mixed precision: Yes (torch.cuda.amp)
                - Hardware: Kaggle T4 × 2 (DataParallel)
                """
            )

        # ── Wire up button ────────────────────────────────────────────────────
        run_btn.click(
            fn=run_inference,
            inputs=[in_img, mask_slider],
            outputs=[out_masked, out_recon, out_gt, info_box],
        )

        # Also trigger on slider change for live preview
        mask_slider.release(
            fn=run_inference,
            inputs=[in_img, mask_slider],
            outputs=[out_masked, out_recon, out_gt, info_box],
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
