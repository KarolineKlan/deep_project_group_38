# src/deep_proj/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
        # output positive concentration parameters \hat{alpha} for each latent dim
        self.net = nn.Sequential(*layers)

        # For DirVAE
        self.alpha_layer = nn.Linear(last, latent_dim)
        # For Gaussian VAE
        self.mu_layer = nn.Linear(last, latent_dim)
        self.logvar_layer = nn.Linear(last, latent_dim)
        # We only need the lambda_layer for CC-VAE
        self.lambda_layer = nn.Linear(last, latent_dim)

    def forward(self, x):
        h = self.net(x)
        if self.bottle == "dir":
            alpha_hat = F.softplus(self.alpha_layer(h)) # softplus to ensure positive alpha_hat; add small bias to avoid zero
            alpha_hat = alpha_hat.clamp(min=1e-3, max=50)
            return alpha_hat
        elif self.bottle == "gaus":
            mu = self.mu_layer(h)
            logvar = self.logvar_layer(h)
            return mu, logvar
        elif self.bottle == "cc":
            # softplus ensures positive lambda_hat (concentration parameter)
            lambda_hat = F.softplus(self.lambda_layer(h))
            lambda_hat = lambda_hat.clamp(min=EPS)
            return lambda_hat
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
        """
        input_dim: flattened input size (e.g. 28*28)
        enc_hidden_dims: list of encoder hidden sizes
        dec_hidden_dims: list of decoder hidden sizes
        latent_dim: K
        prior_alpha: vector or scalar for Dirichlet prior concentration alpha (if scalar, replicate)
        beta: rate parameter for Gammas (paper uses beta=1)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = MLPEncoder(input_dim, enc_hidden_dims, latent_dim, "dir")
        self.decoder = BernoulliDecoder(latent_dim, dec_hidden_dims, input_dim)

        if prior_alpha is None:
            # default weak symmetric prior; user can override
            prior_alpha = torch.ones(latent_dim) * 0.98
        elif torch.is_tensor(prior_alpha):
            if prior_alpha.numel() == 1:
                prior_alpha = prior_alpha.repeat(latent_dim)
        else:
            prior_alpha = torch.tensor(float(prior_alpha)).repeat(latent_dim)

        self.register_buffer("prior_alpha", prior_alpha.float())
        self.beta = float(beta)

    def inverse_gamma_cdf_approx(self, u, alpha):
        """
        Approximate inverse CDF for X ~ Gamma(alpha, beta) using:
        F^{-1}(u; alpha, beta) ≈ beta^{-1} * (u * alpha * Gamma(alpha))^{1/alpha}
        u: uniform samples in (0,1), shape (batch, K)
        alpha: shape (batch, K) or (K,)
        returns: approx gamma samples shape (batch, K)
        """
        # alpha * Gamma(alpha) = alpha * exp(lgamma(alpha))
        # note: torch.lgamma for log Gamma
        # shapes broadcast
        log_gamma = torch.lgamma(alpha)
        a_gamma = alpha * torch.exp(log_gamma)
        u = u.clamp(min=EPS, max=1.0 - 1e-12)
        base = (u * a_gamma).clamp(min=EPS)
        samples = base ** (1.0 / alpha)
        samples = samples / (self.beta + 0.0)
        return samples

    def sample_dirichlet_from_alpha(self, alpha_hat):
        """
        Given alpha_hat (batch, K), produce reparam samples z on simplex:
          1) draw u ~ Uniform(0,1) per component
          2) approximate gamma sample via inverse Gamma CDF approx
          3) normalize v -> z = v / sum_k v_k
        """
        # Uniform draws per component
        u = torch.rand_like(alpha_hat)# Uniform(0,1)
        v = self.inverse_gamma_cdf_approx(u, alpha_hat)
        # Normalize to get Dirichlet sample
        denom = v.sum(dim=1, keepdim=True).clamp(min=EPS)
        z = v / denom
        return z, v, u

    def forward(self, x):
        """
        x: flattened input (batch, input_dim) with values in [0,1] for Bernoulli decoding
        returns: reconstruction logits, z, alpha_hat, v
        """
        alpha_hat = self.encoder(x)
        z, v, u = self.sample_dirichlet_from_alpha(alpha_hat)# z in simplex
        logits = self.decoder(z)
        return logits, z, alpha_hat, v


def multi_gamma_kl(alpha_hat, prior_alpha, reduction="batchmean"):
    """
    KL between MultiGamma(alpha_hat, beta=1) and MultiGamma(prior_alpha, beta=1)
    Per paper (Equation 3):
      KL(Q||P) = sum_k [ log Gamma(alpha_k) - log Gamma(alpha_hat_k) + (alpha_hat_k - alpha_k) * psi(alpha_hat_k) ]
    alpha_hat: (batch, K)
    prior_alpha: (K,) or (batch, K)
    reduction: 'batchmean', 'sum', 'none'
    Returns scalar KL (averaged over batch if batchmean)
    """
    # broadcast prior_alpha to batch if necessary
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
    """
    Compute negative ELBO (loss to minimize) for Bernoulli decoder.
    x: (batch, input_dim) values in {0,1} or [0,1]
    returns loss (scalar), recon_loss (scalar), kl (scalar)
    """
    logits, z, alpha_hat, v = model(x)
    # Reconstruction: bernoulli likelihood -> BCEWithLogits
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    recon_per_sample = bce.sum(dim=1) # per example reconstruction negative log-likelihood
    if reduction == "mean":
        recon_loss = recon_per_sample.mean()
    else:
        recon_loss = recon_per_sample.sum()
    # KL between MultiGamma post (alpha_hat) and prior MultiGamma (prior_alpha)
    kl = multi_gamma_kl(alpha_hat, model.prior_alpha, reduction="batchmean")
    # ELBO = E_q[log p(x|z)] - KL -> loss = -ELBO = recon_loss + KL
    loss = recon_loss + kl
    return loss, recon_loss, kl


def gaussian_vae_elbo_loss(model, x, reduction="mean"):
    """
    Negative ELBO for Gaussian VAE with Bernoulli likelihood.
    """
    logits, mu, logvar, z = model(x)

    # Reconstruction loss
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    recon_per_sample = bce.sum(dim=1)
    if reduction == "mean":
        recon_loss = recon_per_sample.mean()
    else:
        recon_loss = recon_per_sample.sum()

    # KL divergence between q(z|x) and N(0, I)
    kl_per_sample = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=1
    )
    if reduction == "mean":
        kl = kl_per_sample.mean()
    else:
        kl = kl_per_sample.sum()

    loss = recon_loss + kl
    return loss, recon_loss, kl

# =======================
# Continuous Categorical (CC) helpers and VAE
# =======================

def lambda_to_eta(lam: Tensor) -> Tensor:
    """Converts mean parameter lambda [B, K] to natural parameter eta [B, K-1]."""
    # Uses the correct log-ratio logic from the first block
    lam = lam.clamp(min=EPS, max=1.0) 
    last = lam[:, -1].unsqueeze(1)
    eta_full = torch.log(lam) - torch.log(last + EPS)
    return eta_full[:, :-1]

def inv_cdf_torch(u, l):
    """
    Inverse CDF of the continuous Bernoulli distribution (from first block).
    Used for reparameterization in CC sampling.
    """
    near_half = (l > 0.499) & (l < 0.501)
    safe_l = l.clamp(EPS, 1 - EPS)
    u = u.clamp(EPS, 1 - EPS)
    
    num = torch.log(u * (2 * safe_l - 1) + 1 - safe_l) - torch.log(1 - safe_l)
    den = torch.log(safe_l) - torch.log(1 - safe_l)
    x = num / den
    return torch.where(near_half, u, x)

def sample_cc_ordered_reparam(lam):
    """
    Ordered Reparameterized Sampler (from first block - Differentiable).
    Replaces the non-differentiable sample_cc_perm logic.
    lam: [B, K]
    Returns: [B, K] sample on the simplex.
    """
    B, K = lam.shape
    lam_sorted, indices = torch.sort(lam, dim=1, descending=True)
    lam_1 = lam_sorted[:, 0].unsqueeze(1) 
    lam_rest = lam_sorted[:, 1:] 
    
    cb_params = lam_rest / (lam_rest + lam_1 + EPS)

    final_x_rest = torch.zeros_like(lam_rest)
    active_mask = torch.ones(B, dtype=torch.bool, device=lam.device)
    max_attempts = 1000 
    
    for _ in range(max_attempts):
        if not active_mask.any():
            break
        
        n_active = active_mask.sum()
        u = torch.rand(n_active, K-1, device=lam.device, dtype=lam.dtype)
        active_params = cb_params[active_mask]
        x_cand = inv_cdf_torch(u, active_params) 
        sums = x_cand.sum(dim=1)
        accepted_now = (sums <= 1.0)
        
        active_indices = torch.nonzero(active_mask).squeeze(-1)
        accepted_indices = active_indices[accepted_now]
        
        if accepted_indices.numel() > 0:
            final_x_rest[accepted_indices] = x_cand[accepted_now]
            active_mask = active_mask.clone()
            active_mask[accepted_indices] = False
            
    x_1 = (1.0 - final_x_rest.sum(dim=1, keepdim=True)).clamp(min=EPS)
    x_sorted = torch.cat([x_1, final_x_rest], dim=1)
    
    x_final = torch.zeros_like(lam)
    x_final.scatter_(1, indices, x_sorted)
    
    return x_final

def cc_log_norm_const(eta: Tensor) -> Tensor:
    """
    Calculates log C(eta) using the Exact Formula (from the first block, simpler/correct version).
    Input: eta: (n, K-1) tensor.
    Returns: log_C : (n,) tensor equal to log C_K(eta)
    """
    original_dtype = eta.dtype
    eta = eta.double()

    B, K_minus_1 = eta.shape
    K = K_minus_1 + 1
    device = eta.device
    
    # 1. Construct full eta (append 0 for the Kth component)
    eta_full = torch.cat([eta, torch.zeros(B, 1, device=device, dtype=eta.dtype)], dim=1)
    
    # 2. Add Jitter (Crucial for stability in the first block's implementation)
    jitter = torch.arange(K, device=device) * 1e-5
    eta_full = eta_full + jitter.unsqueeze(0)

    # 3. Compute the denominator product: prod_{i!=k} (eta_i - eta_k) 
    eta_i = eta_full.unsqueeze(1) 
    eta_k = eta_full.unsqueeze(2)
    diffs = eta_i - eta_k
    
    eye_mask = torch.eye(K, device=device).bool().unsqueeze(0).expand(B, -1, -1)
    diffs[eye_mask] = 1.0 
    
    log_diffs_abs = diffs.abs().log()
    diffs_sign = diffs.sign()
    
    log_denom = log_diffs_abs.sum(dim=1) 
    denom_sign = diffs_sign.prod(dim=1) 
    
    log_terms_mag = eta_full - log_denom
    terms_sign = denom_sign
    
    # 4. Sum the terms: S = sum_k (T_k)
    max_log_mag, _ = log_terms_mag.max(dim=1, keepdim=True)
    sum_scaled = torch.sum(terms_sign * torch.exp(log_terms_mag - max_log_mag), dim=1)
    
    # 5. Multiply by (-1)^(K+1) 
    global_sign = (-1)**(K + 1)
    total_sum_signed = global_sign * sum_scaled
    
    log_inv_C = max_log_mag.squeeze() + torch.log(total_sum_signed.clamp(min=EPS))
    
    # Return log C = - log(C^-1)
    return -log_inv_C.to(dtype=original_dtype)

def cc_log_prob(sample: Tensor, eta: Tensor) -> Tensor:
    """
    Calculates the log-density p(z | eta) = eta^T * z + log C(eta).
    sample: [B, K], eta: [B, K-1]
    Returns: [B]
    """
    n, K_minus_1 = eta.shape
    aug_eta = torch.cat([eta, torch.zeros(n, 1, device=eta.device, dtype=eta.dtype)], dim=-1)
    
    # Exponent term: eta^T * z
    exponent = torch.sum(sample * aug_eta, dim=1) 
    
    # Log Normalizer term
    log_norm_const = cc_log_norm_const(eta)
    
    return exponent + log_norm_const 

def cc_kl(lambda_hat, prior_lambda, reduction='batchmean'):
    """
    KL Divergence (Analytical form based on the second code block's KL formulation)
    KL(Q||P) = E_q[eta_q^T z] - E_q[eta_p^T z] - log C(eta_q) + log C(eta_p)
    KL(Q||P) = [E_q[eta_q^T z] + log C(eta_q)] - [E_q[eta_p^T z] + log C(eta_p)]
    
    Since E_q[z] is lambda_hat (the mean parameter), this simplifies:
    KL(Q||P) = (eta_hat - eta_prior)^T lambda_hat + log C(eta_prior) - log C(eta_hat)
    
    lambda_hat: (batch, K)
    prior_lambda: (K,) or (batch, K)
    """
    if prior_lambda.dim() == 1:
        prior = prior_lambda.unsqueeze(0).expand_as(lambda_hat)
    else:
        prior = prior_lambda
        
    eta_hat = lambda_to_eta(lambda_hat)      # q params (B, K-1)
    eta_prior = lambda_to_eta(prior)         # p params (B, K-1)

    # For calculation ease, use augmented eta and lambda_hat (mean vector)
    aug_eta_hat = torch.cat([eta_hat, torch.zeros(eta_hat.size(0), 1, device=eta_hat.device, dtype=eta_hat.dtype)], dim=-1) # (B, K)
    aug_eta_prior = torch.cat([eta_prior, torch.zeros(eta_prior.size(0), 1, device=eta_prior.device, dtype=eta_prior.dtype)], dim=-1) # (B, K)

    # Term 1: E_q[eta_q^T z - eta_p^T z] = E_q[(eta_q - eta_p)^T z] = (eta_q - eta_p)^T E_q[z]
    # E_q[z] is lambda_hat (mean parameter)
    diff_eta = aug_eta_hat - aug_eta_prior
    term1 = torch.sum(diff_eta * lambda_hat, dim=1) # (B)

    # Term 2: log C(eta_prior) - log C(eta_hat)
    logC_hat = cc_log_norm_const(eta_hat)
    logC_prior = cc_log_norm_const(eta_prior)
    term2 = logC_prior - logC_hat

    kl = term1 + term2
    if reduction == 'batchmean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    else:
        return kl 


class CCVAE(nn.Module):
    def __init__(self, input_dim, enc_hidden_dims, dec_hidden_dims, latent_dim, prior_lambda=None):
        super().__init__()
        self.latent_dim = latent_dim
        # Use CC bottleneck configuration
        self.encoder = MLPEncoder(input_dim, enc_hidden_dims, latent_dim, "cc")
        # Decoder takes K dimensions (the full simplex vector)
        self.decoder = BernoulliDecoder(latent_dim, dec_hidden_dims, input_dim)

        if prior_lambda is None:
            prior_lambda = torch.ones(latent_dim) # Flat prior, as used in other CC papers
        elif torch.is_tensor(prior_lambda):
            if prior_lambda.numel() == 1:
                prior_lambda = prior_lambda.repeat(latent_dim)
        else:
            prior_lambda = torch.tensor(float(prior_lambda)).repeat(latent_dim)

        # Normalize prior_lambda to sum to 1 (mean vector of the uniform prior)
        prior_lambda = prior_lambda / prior_lambda.sum()

        self.register_buffer('prior_lambda', prior_lambda.float())
        
    def forward(self, x):
        """
        x: flattened input (batch, input_dim) 
        returns: reconstruction logits, z_K (full simplex sample), lambda_hat
        """
        lambda_hat = self.encoder(x)  # (batch, K) - concentration parameter
        # Normalize lambda_hat to get the mean vector (lambda)
        lambda_norm = lambda_hat / lambda_hat.sum(dim=1, keepdim=True)
        
        # Reparameterized sampling to get z_K (full simplex sample)
        z_K = sample_cc_ordered_reparam(lambda_norm)
        
        # Note: The decoder in the second block takes the full K-dimensional simplex vector.
        logits = self.decoder(z_K)  # (batch, input_dim)
        
        return logits, z_K, lambda_norm

def ccvae_elbo_loss(model, x, reduction='mean'):
    """
    Compute negative ELBO (loss to minimize) for Bernoulli decoder (without beta/free-bits).
    x: (batch, input_dim) values in [0,1]
    returns loss (scalar), recon_loss (scalar), kl (scalar)
    """
    # Note: lam here is lambda_norm (the mean parameter)
    logits, z_K, lambda_norm = model(x)
    
    # 1. Reconstruction Loss (Negative Log-Likelihood)
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction='none')
    recon_per_sample = bce.sum(dim=1)  # per example NLL
    
    if reduction == 'mean':
        recon_loss = recon_per_sample.mean()
    else:
        recon_loss = recon_per_sample.sum()
        
    # 2. KL Divergence 
    kl = cc_kl(lambda_norm, model.prior_lambda, reduction='batchmean')
    kl = torch.abs(kl)
    # Loss = -ELBO = Recon Loss + KL
    loss = recon_loss + kl
    return loss, recon_loss, kl

if __name__ == "__main__":
    model = MLPEncoder()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
