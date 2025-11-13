# src/deep_proj/train.py
import hydra
from omegaconf import DictConfig
import torch

from data import get_dataloaders
from model import (
    GaussianVAE,
    DirVAE,
    dirvae_elbo_loss,
    gaussian_vae_elbo_loss,
)


def get_device(cfg: DictConfig):
    if getattr(cfg, "device", "auto") == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


@hydra.main(config_path="../../configs", config_name="base_config", version_base="1.3")
def main(cfg: DictConfig):
    device = get_device(cfg)
    torch.manual_seed(cfg.seed)

    loaders = get_dataloaders(cfg)
    train_loader = loaders["train"]

    # MNIST-like images -> 28x28
    input_dim = 28 * 28
    latent_dim = cfg.latent_dim
    learning_rate = cfg.lr
    epochs = cfg.epochs

    # Choose model
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

            xb = xb.view(xb.size(0), -1)  # flatten

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

        print(
            f"Epoch {epoch:02d} {tag} | "
            f"Loss {tot_loss / n:.4f} | "
            f"Recon {tot_recon / n:.4f} | "
            f"KL {tot_kl / n:.4f}"
        )


if __name__ == "__main__":
    main()
