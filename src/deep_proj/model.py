# src/deep_proj/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, bottle):
        super().__init__()
        self.bottle = bottle
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        self.net = nn.Sequential(*layers)

        # For DirVAE
        self.alpha_layer = nn.Linear(last, latent_dim)
        # For Gaussian VAE
        self.mu_layer = nn.Linear(last, latent_dim)
        self.logvar_layer = nn.Linear(last, latent_dim)

    def forward(self, x):
        h = self.net(x)
        if self.bottle == "dir":
            alpha_hat = F.softplus(self.alpha_layer(h))
            alpha_hat = alpha_hat.clamp(min=1e-3, max=50)
            return alpha_hat
        elif self.bottle == "gaus":
            mu = self.mu_layer(h)
            logvar = self.logvar_layer(h)
            return mu, logvar
        else:
            raise ValueError(f"Unknown bottle type: {self.bottle}")


class BernoulliDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        last = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        logits = self.net(z)
        return logits  # logits used with BCEWithLogits
        

class GaussianVAE(nn.Module):
    def __init__(self, input_dim, enc_hidden_dims, dec_hidden_dims, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = MLPEncoder(input_dim, enc_hidden_dims, latent_dim, "gaus")
        self.decoder = BernoulliDecoder(latent_dim, dec_hidden_dims, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar, z


class DirVAE(nn.Module):
    def __init__(self, input_dim, enc_hidden_dims, dec_hidden_dims, latent_dim,
                 prior_alpha=None, beta=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = MLPEncoder(input_dim, enc_hidden_dims, latent_dim, "dir")
        self.decoder = BernoulliDecoder(latent_dim, dec_hidden_dims, input_dim)

        if prior_alpha is None:
            prior_alpha = torch.ones(latent_dim) * 0.98
        elif torch.is_tensor(prior_alpha):
            if prior_alpha.numel() == 1:
                prior_alpha = prior_alpha.repeat(latent_dim)
        else:
            prior_alpha = torch.tensor(float(prior_alpha)).repeat(latent_dim)

        self.register_buffer("prior_alpha", prior_alpha.float())
        self.beta = float(beta)

    def inverse_gamma_cdf_approx(self, u, alpha):
        log_gamma = torch.lgamma(alpha)
        a_gamma = alpha * torch.exp(log_gamma)
        u = u.clamp(min=EPS, max=1.0 - 1e-12)
        base = (u * a_gamma).clamp(min=EPS)
        samples = base ** (1.0 / alpha)
        samples = samples / (self.beta + 0.0)
        return samples

    def sample_dirichlet_from_alpha(self, alpha_hat):
        u = torch.rand_like(alpha_hat)
        v = self.inverse_gamma_cdf_approx(u, alpha_hat)
        denom = v.sum(dim=1, keepdim=True).clamp(min=EPS)
        z = v / denom
        return z, v, u

    def forward(self, x):
        alpha_hat = self.encoder(x)
        z, v, u = self.sample_dirichlet_from_alpha(alpha_hat)
        logits = self.decoder(z)
        return logits, z, alpha_hat, v


def multi_gamma_kl(alpha_hat, prior_alpha, reduction="batchmean"):
    if prior_alpha.dim() == 1:
        prior = prior_alpha.unsqueeze(0).expand_as(alpha_hat)
    else:
        prior = prior_alpha

    term1 = torch.lgamma(prior) - torch.lgamma(alpha_hat)
    term2 = (alpha_hat - prior) * torch.digamma(alpha_hat)
    kl_comp = term1 + term2
    kl = kl_comp.sum(dim=1)

    if reduction == "batchmean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    else:
        return kl


def dirvae_elbo_loss(model, x, reduction="mean"):
    logits, z, alpha_hat, v = model(x)
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    recon_per_sample = bce.sum(dim=1)
    if reduction == "mean":
        recon_loss = recon_per_sample.mean()
    else:
        recon_loss = recon_per_sample.sum()

    kl = multi_gamma_kl(alpha_hat, model.prior_alpha, reduction="batchmean")
    loss = recon_loss + kl
    return loss, recon_loss, kl


def gaussian_vae_elbo_loss(model, x, reduction="mean"):
    logits, mu, logvar, z = model(x)

    bce = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    recon_per_sample = bce.sum(dim=1)
    if reduction == "mean":
        recon_loss = recon_per_sample.mean()
    else:
        recon_loss = recon_per_sample.sum()

    kl_per_sample = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=1
    )
    if reduction == "mean":
        kl = kl_per_sample.mean()
    else:
        kl = kl_per_sample.sum()

    loss = recon_loss + kl
    return loss, recon_loss, kl


if __name__ == "__main__":
    model = MLPEncoder()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
