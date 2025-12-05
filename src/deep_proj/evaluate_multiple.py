# src/deep_proj/evaluate_multiple.py
#
# Compare latent spaces of three trained models (Gaussian, Dirichlet, CC)
#
# Usage (from project root):
#   python -m src.deep_proj.evaluate_multiple

import os
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig
from sklearn.manifold import TSNE

from .data import get_dataloaders
from .model import (
    GaussianVAE,
    DirVAE,
    CCVAE,
)



# -------------------------------------------------------------------
# Project root (repo root)
# -------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# -------------------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------------------
# Number of images per row in the reconstruction comparison figure.
# Change this to 10, 12, ... if you want more examples.
N_RECON_SAMPLES = 12
GAUSS_FINAL_MODEL = "final_sweep/mnist_gaussian_z8_lr0.0007_best.pt"
DIR_FINAL_MODEL = "final_sweep/mnist_dirichlet_z8_lr0.0007_best.pt"
CC_FINAL_MODEL = "final_sweep/mnist_cc_z3_lr0.0003_best.pt"  # placeholder for CC


# -------------------------------------------------------------------
# Device helper
# -------------------------------------------------------------------
def get_device(device_str: str = "auto"):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# -------------------------------------------------------------------
# Build model from checkpoint config
# -------------------------------------------------------------------
def build_model_from_config(cfg: DictConfig, device: torch.device):
    input_dim = 28 * 28
    latent_dim = cfg.latent_dim
    name = cfg.model_name.lower()

    if name in ("gaussian", "gaus", "gauss"):
        model = GaussianVAE(
            input_dim=input_dim,
            enc_hidden_dims=[500, 500],
            dec_hidden_dims=[500],
            latent_dim=latent_dim,
        ).to(device)
    elif name in ("dirichlet", "dir"):
        model = DirVAE(
            input_dim=input_dim,
            enc_hidden_dims=[500, 500],
            dec_hidden_dims=[500],
            latent_dim=latent_dim,
            prior_alpha=cfg.alpha_init,
        ).to(device)
    elif name in ("cc", "ccvae", "continuous_categorical"):
        # TODO: replace with CCVAE once implemented  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        model = CCVAE(
            input_dim=input_dim,
            enc_hidden_dims=[500, 500],
            dec_hidden_dims=[500],
            latent_dim=latent_dim,
            prior_lambda=None,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_name in cfg: {cfg.model_name}")

    return model


# -------------------------------------------------------------------
# Extract t-SNE embedding from a model on the test loader
# -------------------------------------------------------------------
def get_tsne_embedding(model, cfg, test_loader, device, tsne_samples=4000):
    model.eval()
    z_all, y_all = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)

            # Undo normalization just like in training
            if cfg.dataset.lower() == "mnist":
                xb = xb * 0.3081 + 0.1307
            elif cfg.dataset.lower() == "medmnist":
                xb = xb * 0.5 + 0.5

            xb = xb.view(xb.size(0), -1)

            mname = cfg.model_name.lower()
            if mname in ("gaussian", "gaus", "gauss"):
                _, mu, logvar, z = model(xb)
            elif mname in ("dirichlet", "dir"):
                _, z, _, _ = model(xb)
            elif mname in ("cc", "ccvae", "continuous_categorical"):
                _, z, _= model(xb)
            else:
                raise ValueError(f"Unknown model_name: {cfg.model_name}")

            z_all.append(z.detach().cpu())
            y_all.append(yb)

            if sum(len(t) for t in z_all) >= tsne_samples:
                break

    z_all = torch.cat(z_all, dim=0)[:tsne_samples].numpy()
    y_all = torch.cat(y_all, dim=0)[:tsne_samples].numpy()

    # Drop NaNs if any
    mask = np.isfinite(z_all).all(axis=1)
    z_all = z_all[mask]
    y_all = y_all[mask]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(z_all)

    return z_2d, y_all


# -------------------------------------------------------------------
# Main: load three checkpoints and plot comparison
# -------------------------------------------------------------------
def main():
    device = get_device("auto")

    # Hard-coded checkpoints (adjust filenames to match your models)
    ckpts = [
        ("Gaussian-VAE (M=8)", os.path.join(project_root, "models", GAUSS_FINAL_MODEL)),
        ("Dirichlet-VAE (M=8)", os.path.join(project_root, "models", DIR_FINAL_MODEL)),
        ("CC-VAE (M=3)", os.path.join(project_root, "models", CC_FINAL_MODEL)),  # placeholder for CC
    ]

    embeddings = []
    labels_list = []
    titles = []

    # For reconstructions
    base_batch = None        # normalized test images (N, 1, 28, 28)
    base_dataset = None
    recon_rows = []          # list of tensors (N, 784) per model

        # <- NEW: will hold [0,1,4] etc. if present in cfg
    legend_classes_from_cfg = None

    for idx, (title, ckpt_path) in enumerate(ckpts):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"\n=== Loading checkpoint: {ckpt_path} ===")
        ckpt = torch.load(ckpt_path, map_location=device)

        if "config" not in ckpt:
            raise KeyError(
                "Checkpoint does not contain a 'config' field. "
                "Make sure you saved it in train.py via OmegaConf.to_container(cfg, resolve=True)."
            )

        cfg = OmegaConf.create(ckpt["config"])
        if not isinstance(cfg, DictConfig):
            cfg = DictConfig(cfg)

        print("  model_name:", cfg.model_name)
        print("  dataset   :", cfg.dataset)
        print("  latent_dim:", cfg.latent_dim)

        # NEW: grab class subset once (they should be same across models)
        if legend_classes_from_cfg is None:
            if getattr(cfg, "mnist_classes", None) is not None:
                legend_classes_from_cfg = list(cfg.mnist_classes)
            elif getattr(cfg, "medmnist_classes", None) is not None:
                legend_classes_from_cfg = list(cfg.medmnist_classes)

        # Build dataloaders and model
        loaders = get_dataloaders(cfg)
        test_loader = loaders["test"]

        model = build_model_from_config(cfg, device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        # ------------------------------------------------------------------
        # Fix a shared batch from the test set (first time only) for recon
        # ------------------------------------------------------------------
        if base_batch is None:
            xb0, _ = next(iter(test_loader))   # xb0 is normalized
            base_batch = xb0[:N_RECON_SAMPLES].clone()
            base_dataset = cfg.dataset

        # Compute reconstructions for this model on the shared batch
        with torch.no_grad():
            xb = base_batch.to(device)

            # denormalize for the model input and for plotting
            if base_dataset.lower() == "mnist":
                xb_denorm = xb * 0.3081 + 0.1307
            elif base_dataset.lower() == "medmnist":
                xb_denorm = xb * 0.5 + 0.5
            else:
                xb_denorm = xb

            xb_flat = xb_denorm.view(xb_denorm.size(0), -1)

            mname = cfg.model_name.lower()
            if mname in ("gaussian", "gaus", "gauss"):
                logits, mu, logvar, z = model(xb_flat)
            elif mname in ("dirichlet", "dir"):
                logits, z, _, _ = model(xb_flat)
            elif mname in ("cc", "ccvae", "continuous_categorical"):
                logits, z, lambda_norm = model(xb_flat)
            else:
                raise ValueError(f"Unknown model_name: {cfg.model_name}")

            recon = torch.sigmoid(logits).detach().cpu()  # (N, 784)
            recon_rows.append(recon)

        # ------------------------------------------------------------------
        # Get t-SNE embedding on test set
        # ------------------------------------------------------------------
        z_2d, y_all = get_tsne_embedding(model, cfg, test_loader, device, tsne_samples=5000)
        embeddings.append(z_2d)
        labels_list.append(y_all)
        titles.append(title)

    # ----------------------------------------------------------------
    # 1) Plot side-by-side comparison of latent spaces with clean legend
    # ----------------------------------------------------------------
    out_dir = os.path.join(project_root, "reports", "figures", "multi_eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path_latent = os.path.join(out_dir, "latent_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Decide which classes should be in the legend
    if legend_classes_from_cfg is not None:
        unique_classes = np.array(legend_classes_from_cfg, dtype=int)
    else:
        unique_classes = np.unique(np.concatenate(labels_list)).astype(int)

    # Build a stable color map: class_id -> color (inspired by simplex plot)
    base_cmap = plt.get_cmap("tab10")
    try:
        palette = list(base_cmap.colors)
    except AttributeError:
        palette = [base_cmap(i) for i in range(base_cmap.N)]

    color_map = {int(cls): palette[int(cls) % len(palette)] for cls in unique_classes}

    # Plot each model using the SAME class→color mapping
    for ax, emb, labels, title in zip(axes, embeddings, labels_list, titles):
        labels = labels.astype(int)
        point_colors = [color_map[int(lbl)] for lbl in labels]

        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=point_colors,   # explicit RGBA colors, no normalization
            s=15,
            alpha=0.7,
        )
        ax.set_title(title, fontsize=35)
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend that matches exactly the scatter colors
    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color=color_map[int(cls)],
            linestyle="",
            markersize=10,
            label=str(int(cls)),
        )
        for cls in unique_classes
    ]

    fig.legend(
        handles=legend_handles,
        ncol=len(unique_classes),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        fontsize=20,
    )


    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path_latent, dpi=200, bbox_inches="tight")
    plt.savefig(out_path_latent.replace(".png", ".pdf"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved 3-model latent comparison figure to:\n  {out_path_latent}\n")

    # ----------------------------------------------------------------
    # 2) Plot 4xN reconstruction comparison (Original + 3 models)
    # ----------------------------------------------------------------
    out_path_recon = os.path.join(out_dir, "reconstruction_comparison.png")

    # Prepare original images (denormalized) for row 0
    xb_orig = base_batch
    if base_dataset.lower() == "mnist":
        xb_orig_denorm = xb_orig * 0.3081 + 0.1307
    elif base_dataset.lower() == "medmnist":
        xb_orig_denorm = xb_orig * 0.5 + 0.5
    else:
        xb_orig_denorm = xb_orig

    xb_orig_denorm = xb_orig_denorm.cpu()  # (N, 1, 28, 28)
    n_samples = xb_orig_denorm.size(0)

    # recon_rows: [gauss_recon, dir_recon, cc_recon], each (N, 784)
    row_titles = ["Original Image", "Gaussian-VAE", "Dirichlet-VAE", "CC-VAE"]

    fig, axes = plt.subplots(4, n_samples, figsize=(1.4 * n_samples, 5.5))

    # Row 0: originals
    orig_imgs = xb_orig_denorm.view(n_samples, 28, 28)
    for j in range(n_samples):
        ax = axes[0, j]
        ax.imshow(orig_imgs[j], cmap="gray")
        ax.axis("off")

    # Rows 1..3: reconstructions
    for row_idx, recon in enumerate(recon_rows, start=1):
        imgs = recon.view(n_samples, 28, 28)
        for j in range(n_samples):
            ax = axes[row_idx, j]
            ax.imshow(imgs[j], cmap="gray")
            ax.axis("off")

    # --------------------------------------------------------
    # Add row titles to the LEFT of each row (outside the axes)
    # --------------------------------------------------------
    # Leave space on the left for the labels
    plt.tight_layout(rect=[0.08, 0, 1, 1])

    for row_idx, label in enumerate(row_titles):
        # Take the first axis in this row to get its position
        ax0 = axes[row_idx, 0]
        bbox = ax0.get_position()
        y_center = bbox.y0 + bbox.height / 2.0

        fig.text(
            0.07,                 # x-position in figure coordinates (left margin)
            y_center,
            label,
            ha="right",
            va="center",
            fontsize=40,
        )
    plt.savefig(out_path_recon, dpi=200, bbox_inches="tight")
    plt.savefig(out_path_recon.replace(".png", ".pdf"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved reconstruction comparison figure to:\n  {out_path_recon}\n")


if __name__ == "__main__":
    main()
