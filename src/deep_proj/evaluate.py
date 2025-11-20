# src/deep_proj/evaluate.py
import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

from .data import get_dataloaders
from .model import (
    GaussianVAE,
    DirVAE,
    dirvae_elbo_loss,
    gaussian_vae_elbo_loss,
)
from .train import evaluate_split  # reuse helper


def get_device(cfg: DictConfig):
    if getattr(cfg, "device", "auto") == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def build_model_from_config(cfg: DictConfig, device):
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
    else:
        raise ValueError(f"Unknown model_name: {cfg.model_name}")

    return model, loss_fn


@hydra.main(config_path="../../configs", config_name="base_config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Evaluate a saved checkpoint on val + test.

    Usage:
        python -m src.deep_proj.evaluate checkpoint_name=mnist_gaussian_z20_lr0.0003_best.pt
    """
    device = get_device(cfg)

    # ------------------------------------------------------
    # Locate checkpoint
    # ------------------------------------------------------
    project_root = get_original_cwd()
    ckpt_dir = os.path.join(project_root, "models")
    ckpt_name = cfg.get("checkpoint_name", None)

    if ckpt_name is None:
        raise ValueError(
            "Please provide checkpoint_name, e.g. checkpoint_name=mnist_gaussian_z20_lr0.0003_best.pt"
        )

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\n=== Loading checkpoint: {ckpt_path} ===")
    ckpt = torch.load(ckpt_path, map_location=device)

    # ------------------------------------------------------
    # Restore config saved in checkpoint
    # ------------------------------------------------------
    if "config" in ckpt:
        ckpt_cfg = OmegaConf.create(ckpt["config"])
        cfg = OmegaConf.merge(ckpt_cfg, cfg)

    # ------------------------------------------------------
    # Build model
    # ------------------------------------------------------
    model, loss_fn = build_model_from_config(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------
    loaders = get_dataloaders(cfg)
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # ------------------------------------------------------
    # >>> NEW: Print dataset sizes
    # ------------------------------------------------------
    print("\n=== Dataset Sizes ===")
    print(f"Validation set size: {len(val_loader.dataset)} samples")
    print(f"Test set size:       {len(test_loader.dataset)} samples\n")

    # ------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------
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

    # Simplex plot
    from .simplex import plot_latent_simplex
    plot_latent_simplex(model=model, loader=test_loader, device=device, model_type="dirichlet", save_dir="plots", model_name=cfg.model_name, n_samples=1000)

if __name__ == "__main__":
    main()
