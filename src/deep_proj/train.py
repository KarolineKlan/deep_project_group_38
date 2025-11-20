# src/deep_proj/train.py
import os
import torch
import wandb  # W&B
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
from .visualize import (
    plot_training_progress,
    plot_final_results,
    plot_dirichlet_simplex_nD,
    visualize_model,
)


def get_device(cfg: DictConfig):
    if getattr(cfg, "device", "auto") == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def evaluate_split(model, loader, loss_fn, cfg, device):
    """Compute avg loss, recon, KL on a given dataloader (no grad)."""
    model.eval()
    tot_loss = 0.0
    tot_recon = 0.0
    tot_kl = 0.0
    n = 0

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)

            # undo normalization so inputs are ~[0,1] for Bernoulli likelihood
            if cfg.dataset.lower() == "mnist":
                xb = xb * 0.3081 + 0.1307
            elif cfg.dataset.lower() == "medmnist":
                xb = xb * 0.5 + 0.5

            xb = xb.view(xb.size(0), -1)
            loss, recon, kl = loss_fn(model, xb, reduction="mean")

            bs = xb.size(0)
            n += bs
            tot_loss += loss.item() * bs
            tot_recon += recon.item() * bs
            tot_kl += kl.item() * bs

    if n == 0:
        return 0.0, 0.0, 0.0

    return tot_loss / n, tot_recon / n, tot_kl / n


@hydra.main(config_path="../../configs", config_name="base_config", version_base="1.3")
def main(cfg: DictConfig):
    device = get_device(cfg)
    torch.manual_seed(cfg.seed)

    loaders = get_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # ----------------------------------------------------------
    # 1) Base hyperparams from Hydra (defaults or manual CLI)
    # ----------------------------------------------------------
    base_model_name = cfg.model_name
    base_latent_dim = cfg.latent_dim
    base_lr = cfg.lr
    base_dataset = cfg.dataset
    base_seed = cfg.seed

    torch.manual_seed(base_seed)

    loaders = get_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # ----------------------------------------------------------
    # 2) Init W&B and override cfg from wandb.config (for sweeps)
    # ----------------------------------------------------------
    wandb_run = None
    if getattr(cfg, "wandb", None) and cfg.wandb.enabled:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=getattr(cfg.wandb, "entity", None),
            group=getattr(cfg.wandb, "group", None),
            mode=getattr(cfg.wandb, "mode", "online"),
            config={
                "model_name": base_model_name,
                "latent_dim": base_latent_dim,
                "lr": base_lr,
                "dataset": base_dataset,
                "seed": base_seed,
            },
        )
        wcfg = wandb.config
        run_name = f"{wcfg.dataset}-{wcfg.model_name}-z{wcfg.latent_dim}-lr{wcfg.lr}"
        wandb.run.name = run_name
        wandb.run.tags = [str(wcfg.dataset), str(wcfg.model_name), f"z{wcfg.latent_dim}"]

        # overwrite cfg with sweep values if present
        cfg.model_name = str(getattr(wcfg, "model_name", base_model_name))
        cfg.latent_dim = int(getattr(wcfg, "latent_dim", base_latent_dim))
        cfg.lr = float(getattr(wcfg, "lr", base_lr))
        cfg.dataset = str(getattr(wcfg, "dataset", base_dataset))

        torch.manual_seed(cfg.seed)

    # ----------------------------------------------------------
    # 3) Use updated cfg.* for the rest of training
    # ----------------------------------------------------------
    input_dim = 28 * 28
    latent_dim = cfg.latent_dim
    learning_rate = cfg.lr
    epochs = cfg.epochs

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
        tag = "DIR"
    elif model_name in ("gaussian", "gaus", "gauss"):
        model = GaussianVAE(
            input_dim=input_dim,
            enc_hidden_dims=[500, 500],
            dec_hidden_dims=[500],
            latent_dim=latent_dim,
        ).to(device)
        loss_fn = gaussian_vae_elbo_loss
        tag = "GAUSS"
    else:
        raise ValueError(f"Unknown model_name: {cfg.model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if wandb_run is not None:
        wandb.watch(model, log="all", log_freq=100)

    # ----------------------------------------------------------
    # Checkpoint setup (project_root/models/)
    # ----------------------------------------------------------
    project_root = get_original_cwd()
    ckpt_dir = os.path.join(project_root, "models")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_ckpt_path = None
    last_val_loss = None

    # training history
    train_loss_hist = []
    recon_hist = []
    kl_hist = []

    val_loss_hist = []
    val_recon_hist = []
    val_kl_hist = []

    # path for plots
    plot_path = os.path.join("reports", "figures", cfg.model_name)
    os.makedirs(plot_path, exist_ok=True)

    # ----------------------------------------------------------
    # Plot epoch 0 (untrained model) once before training loop
    # ----------------------------------------------------------
    epoch0_img_path = plot_training_progress(
        model,
        train_loader.dataset,
        epoch=0,
        bottleneck=model_name,
        device=device,
        n_samples=10000,
        save_path=plot_path,
    )
    if wandb_run is not None and epoch0_img_path is not None:
        wandb.log(
            {
                "epoch": 0,
                "plots/progress_epoch_000": wandb.Image(epoch0_img_path),
            }
        )

    # ----------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        tot_recon = 0.0
        tot_kl = 0.0
        n = 0

        for xb, _ in train_loader:
            xb = xb.to(device)

            # undo normalization so inputs are ~[0,1] for Bernoulli likelihood
            if cfg.dataset.lower() == "mnist":
                xb = xb * 0.3081 + 0.1307
            elif cfg.dataset.lower() == "medmnist":
                xb = xb * 0.5 + 0.5

            xb = xb.view(xb.size(0), -1)

            optimizer.zero_grad()
            loss, recon, kl = loss_fn(model, xb, reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bs = xb.size(0)
            n += bs
            tot_loss += loss.item() * bs
            tot_recon += recon.item() * bs
            tot_kl += kl.item() * bs

        avg_loss = tot_loss / n
        avg_recon = tot_recon / n
        avg_kl = tot_kl / n

        # --- validation pass ---
        val_loss, val_recon, val_kl = evaluate_split(
            model, val_loader, loss_fn, cfg, device
        )
        last_val_loss = val_loss

        print(
            f"Epoch {epoch:02d} {tag} | "
            f"Train Loss {avg_loss:.4f} | Recon {avg_recon:.4f} | KL {avg_kl:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Recon {val_recon:.4f} | Val KL {val_kl:.4f}"
        )

        # history
        train_loss_hist.append(avg_loss)
        recon_hist.append(avg_recon)
        kl_hist.append(avg_kl)
        val_loss_hist.append(val_loss)
        val_recon_hist.append(val_recon)
        val_kl_hist.append(val_kl)

        # ------------------------------------------------------
        # Checkpoint: save best model w.r.t validation loss
        # ------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_name = (
                f"{cfg.dataset}_{cfg.model_name}_z{latent_dim}_lr{learning_rate}_best.pt"
            )
            best_ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                best_ckpt_path,
            )
            print(f"Saved new best checkpoint to: {best_ckpt_path}")

        # W&B: log scalars
        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": avg_loss,
                    "train/recon": avg_recon,
                    "train/kl": avg_kl,
                    "val/loss": val_loss,
                    "val/recon": val_recon,
                    "val/kl": val_kl,
                }
            )

        # Optional: plot every viz_every epochs
        if epoch % cfg.viz_every == 0 or epoch == epochs:
            img_path=visualize_model(model, epoch, train_loader, device, plot_path,
                    cfg.model_name, n_samples=8, tsne_samples=1000)
            # W&B: log the image
            if wandb_run is not None and img_path is not None:
                wandb.log({f"plots/progress_epoch_{epoch:03d}": wandb.Image(img_path)})

    # ----------------------------------------------------------
    # Save final ("last") checkpoint
    # ----------------------------------------------------------
    last_ckpt_name = (
        f"{cfg.dataset}_{cfg.model_name}_z{latent_dim}_lr{learning_rate}_last.pt"
    )
    last_ckpt_path = os.path.join(ckpt_dir, last_ckpt_name)
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": last_val_loss,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        last_ckpt_path,
    )
    print(f"Saved final checkpoint to: {last_ckpt_path}")

    # Visualize final results (still using training logs only)
    training_logs = {
        "loss": train_loss_hist,
        "recon": recon_hist,
        "kl": kl_hist,
        # later add val_* in visualize,
        # "val_loss": val_loss_hist,
        # "val_recon": val_recon_hist,
        # "val_kl": val_kl_hist,
    }

    final_proj_path, final_recon_path = plot_final_results(
        model,
        train_loader.dataset,
        training_logs,
        bottleneck=model_name,
        device=device,
        n_samples=1000,
        save_path=plot_path,
    )

    if model_name in ("dirichlet", "dir", "cc"):
        simplex_path = plot_dirichlet_simplex_nD(
            model,
            train_loader.dataset,
            bottleneck=model_name,
            device=device,
            n_points=1000,
            save_path=plot_path,
        )
    else:
        simplex_path = None

    # W&B: log final figures
    if wandb_run is not None:
        if final_proj_path is not None:
            wandb.log({"plots/final_projections": wandb.Image(final_proj_path)})
        if final_recon_path is not None:
            wandb.log({"plots/final_recons": wandb.Image(final_recon_path)})
        if simplex_path is not None:
            wandb.log({"plots/dirichlet_simplex": wandb.Image(simplex_path)})

        wandb_run.finish()


if __name__ == "__main__":
    main()
