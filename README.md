<<<<<<< HEAD
# Masked Autoencoder (MAE) | Image Reconstruction

---

## What is this?

We trained a model that learns to reconstruct images by looking at only **25% of an image** and filling in the rest. It's like solving a jigsaw puzzle, the model figures out the missing pieces on its own, without any labels.

This is called **self-supervised learning** ,  the model teaches itself just by looking at images.

---

## How it works

1. Take an image and split it into small 16×16 patches
2. Randomly hide **75% of the patches** (147 out of 196)
3. Show only the remaining **25% (49 patches)** to the encoder
4. The decoder tries to reconstruct the full image
5. Loss is computed only on the hidden patches

---

## Results

| Metric | Score |
|--------|-------|
| PSNR | 18.65 dB |
| SSIM | 0.5569 |

Trained for **50 epochs** on TinyImageNet (100K images) using **2× Kaggle T4 GPUs**.

---

## Project Files

```
├── app.py                             # Gradio web app
├── model.py                           # Model architecture
├── requirements.txt                   # Dependencies
├── 22F-3617__22F-2616_assign2.ipynb   # Training notebook
└── checkpoints/
    └── mae_best.pt                    # Trained weights (download separately)
```

> ⚠️ `mae_best.pt` is 1.29 GB so it's not uploaded here.  
> Download it from our **[Hugging Face Space](https://huggingface.co/spaces/SabahatJahangir/Masked_Autoencoders)**

---

## Run Locally

```bash
git clone https://github.com/Sabahat-Jahangir/Masked-Autoencoder-MAE-Image-Reconstruction
cd YOUR_REPO

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:7860** in your browser.

---

## Live Demo

👉 **[Try it on Hugging Face](https://huggingface.co/spaces/SabahatJahangir/Masked_Autoencoders)**

Upload any image, adjust the masking ratio, and watch the model reconstruct it in real time.

---

## Tech Stack

- PyTorch (pure base layers, no pretrained models)
- Gradio
- Trained on Kaggle T4 × 2

---
=======
# Masked-Autoencoder-MAE-Image-Reconstruction
>>>>>>> 4446a1ccc4af8dc25e7dd55b7237eb95dbc5ce53
