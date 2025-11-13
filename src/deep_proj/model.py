from torch import nn
import torch

class Model(nn.Module):
    """Just a dummy model to show how to structure your code"""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        self.net = nn.Sequential(*layers)
        self.alpha_layer = nn.Linear(last, latent_dim)

    def forward(self, x):
        h = self.net(x)
        alpha_hat = F.softplus(self.alpha_layer(h)) + 1e-6
        alpha_hat = torch.clamp(alpha_hat, min=1e-3, max=1e3)
        return alpha_hat


class DirVAE(nn.Module):
    def __init__(self, input_dim, enc_hidden_dims, dec_hidden_dims, latent_dim,
                 prior_alpha=None, beta=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = MLPEncoder(input_dim, enc_hidden_dims, latent_dim)
        self.decoder = BernoulliDecoder(latent_dim, dec_hidden_dims, input_dim)

        if prior_alpha is None:
            prior_alpha = torch.ones(latent_dim) * 0.98
        elif torch.is_tensor(prior_alpha):
            if prior_alpha.numel() == 1:
                prior_alpha = prior_alpha.repeat(latent_dim)
        else:
            prior_alpha = torch.tensor(float(prior_alpha)).repeat(latent_dim)

        self.register_buffer('prior_alpha', prior_alpha.float())
        self.beta = float(beta)

    def inverse_gamma_cdf_approx(self, u, alpha):
        log_gamma = torch.lgamma(alpha)
        a_gamma = alpha * torch.exp(log_gamma)
        u = u.clamp(min=EPS, max=1.0 - 1e-12)
        base = (u * a_gamma).clamp(min=EPS)
        samples = (base) ** (1.0 / alpha)
        samples = samples / (self.beta + 0.0)
        return samples

    def sample_dirichlet_from_alpha(self, alpha_hat):
        batch = alpha_hat.shape[0]
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


if __name__ == "__main__":
    model = Model()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
