# src/deep_proj/evaluate.py
#
# Simple evaluation script that does NOT use Hydra configs.
# It only needs a checkpoint path. The checkpoint already contains
# the full training config, so we rebuild everything from that.
#
# Usage (from project root):
#   python -m src.deep_proj.evaluate --checkpoint models/mnist_dirichlet_z10_lr0.0005_best.pt
#
# or, just the filename:
#   python -m src.deep_proj.evaluate --checkpoint mnist_dirichlet_z10_lr0.0005_best.pt

import os
import argparse
import torch
from omegaconf import OmegaConf, DictConfig

from .simplex import plot_latent_simplex
from .visualize import plot_latent, plot_recons
from .data import get_dataloaders
from .model import (
    GaussianVAE,
    DirVAE,
    CCVAE,
    dirvae_elbo_loss,
    gaussian_vae_elbo_loss,
    ccvae_elbo_loss,
)
from .train import evaluate_split  # reuse your existing helper

# project root = repo root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ----------------------------
# Helper: device selection
# ----------------------------
def get_device(device_str: str = "auto"):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ----------------------------
# Helper: build model from cfg
# ----------------------------
def build_model_from_config(cfg: DictConfig, device: torch.device):
    input_dim = 28 * 28
    latent_dim = cfg.latent_dim
    model_name = cfg.model_name.lower()

    if model_name in ("dirichlet", "dir"):
        model = DirVAE(
            input_dim=input_dim,
            enc_hidden_dims=[500, 500],
            dec_hidden_dims=[500],
            latent_dim=latent_dim,
            prior_alpha=cfg.alpha_init,
        ).to(device)
        loss_fn = dirvae_elbo_loss

    elif model_name in ("gaussian", "gaus", "gauss"):
        model = GaussianVAE(
            input_dim=input_dim,
            enc_hidden_dims=[500, 500],
            dec_hidden_dims=[500],
            latent_dim=latent_dim,
        ).to(device)
        loss_fn = gaussian_vae_elbo_loss
    
    elif model_name in ("cc"):
        model = CCVAE(
            input_dim=input_dim,
            enc_hidden_dims=[500, 500],
            dec_hidden_dims=[500],
            latent_dim=latent_dim,
        ).to(device)
        loss_fn = ccvae_elbo_loss

    else:
        raise ValueError(f"Unknown model_name: {cfg.model_name}")

    return model, loss_fn


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint .pt file (relative to project root or absolute).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device: "auto", "cpu", or "cuda"',
    )
    args = parser.parse_args()

    # If user only passed a filename, assume models/ subdir
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path) and not os.path.exists(ckpt_path):
        ckpt_path = os.path.join("models", ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = get_device(args.device)

    print(f"\n=== Loading checkpoint: {ckpt_path} ===")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "config" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain a 'config' field. "
            "Make sure you saved it in train.py via OmegaConf.to_container(cfg, resolve=True)."
        )

    # Restore config used during training
    cfg = OmegaConf.create(ckpt["config"])
    if not isinstance(cfg, DictConfig):
        cfg = DictConfig(cfg)

    print("\n=== Config from checkpoint ===")
    print(OmegaConf.to_yaml(cfg))

    # Build the same run_id convention as in train.py / checkpoints
    run_id = f"{cfg.dataset}_{cfg.model_name}_z{cfg.latent_dim}_lr{cfg.lr}"
    print(f"\nRun ID (used for folders): {run_id}")

    # Build model + loss
    model, loss_fn = build_model_from_config(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Build data loaders for this exact dataset / splits
    loaders = get_dataloaders(cfg)
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # Info
    print("\n=== Dataset Sizes ===")
    print(f"Validation set size: {len(val_loader.dataset)} samples")
    print(f"Test set size:       {len(test_loader.dataset)} samples\n")

    # Evaluate
    val_loss, val_recon, val_kl = evaluate_split(
        model, val_loader, loss_fn, cfg, device
    )
    test_loss, test_recon, test_kl = evaluate_split(
        model, test_loader, loss_fn, cfg, device
    )

    print("=== Evaluation Results ===")
    print(
        f"Validation  | Loss {val_loss:.4f} | Recon {val_recon:.4f} | KL {val_kl:.4f}"
    )
    print(
        f"Test        | Loss {test_loss:.4f} | Recon {test_recon:.4f} | KL {test_kl:.4f}"
    )
    print("=========================================\n")

    # -------------------------------------------------
    # Evaluation figure directory (mirrors train.py)
    # reports/figures/<run_id>/evaluation/
    # -------------------------------------------------
    eval_dir = os.path.join(project_root, "reports", "figures", run_id, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    print(f"Saving evaluation plots to: {eval_dir}")

    # --------- Plotting latent simplex ---------
    model_name = cfg.model_name.lower()
    if model_name in ("dir", "dirichlet"):
        model_type = "dirichlet"
    elif model_name in ("gaus", "gaussian", "gauss"):
        model_type = "gaussian"
    elif model_name in ("cc", "ccvae"):
        model_type = "cc"
    else:
        raise ValueError(f"Unknown model_name: {cfg.model_name}")




    simplex_path = plot_latent_simplex(
        model=model,
        dataset_name=cfg.dataset,
        loader=test_loader,
        device=device,
        model_type=model_type,
        class_labels=cfg.mnist_classes,
        n_samples=5000,
        save_dir=eval_dir,
        model_name=run_id,  # so the file name also matches the run_id
    )
    print(f"Saved simplex latent plot to: {simplex_path}")

    # --------- Plotting t-SNE latent and reconstructions ---------
    latent_path = plot_latent(
        model=model,
        epoch=0,                    # epoch is irrelevant for eval
        loader=test_loader,
        device=device,
        save_dir=eval_dir,
        model_name=cfg.model_name.lower(),
        dataset_name=cfg.dataset,
        tsne_samples=3000,
        eval=True,
    )
    print("Saved t-SNE latent plot:", latent_path)

    recon_path = plot_recons(
        model=model,
        epoch=0,
        loader=test_loader,
        device=device,
        save_dir=eval_dir,
        model_name=cfg.model_name.lower(),
        dataset_name=cfg.dataset,
        n_samples=10,
        eval=True,
    )
    print("Saved reconstructions:", recon_path)


if __name__ == "__main__":
    main()
