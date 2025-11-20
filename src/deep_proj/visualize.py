import os
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE, MDS
import umap
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from sklearn.decomposition import PCA


# ---------- NEW: helper to drop NaN/inf latents safely ----------


def _sanitize_latents(z, labels, what="embedding", min_points=50):
    """
    Remove rows with NaN/inf from z (and corresponding labels).
    Returns (z_clean, labels_clean) as numpy arrays, or (None, None) if too few.
    """
    if isinstance(z, torch.Tensor):
        z_np = z.detach().cpu().numpy()
    else:
        z_np = np.asarray(z)

    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    if z_np.ndim != 2:
        z_np = np.reshape(z_np, (z_np.shape[0], -1))

    mask = np.isfinite(z_np).all(axis=1)
    if not mask.all():
        dropped = (~mask).sum()
        print(
            f"[WARN] _sanitize_latents: dropped {dropped} samples with NaN/inf for {what}."
        )
        z_np = z_np[mask]
        labels_np = labels_np[mask]

    if z_np.shape[0] < min_points:
        print(
            f"[WARN] _sanitize_latents: only {z_np.shape[0]} valid samples for {what}, "
            f"min_points={min_points}. Skipping embedding."
        )
        return None, None

    return z_np, labels_np


######################## Plot t-SNE of latent space and example reconstructions during training ########################
def plot_training_progress(
    model, dataset, epoch, bottleneck, device=None, n_samples=10000, save_path=None
):
    """
    Plot latent t-SNE + example reconstructions for a given epoch.
    Returns the filename of the saved figure (or None if not saved).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z, labels, xb, recon = _get_latents_and_recons(
        model, dataset, device, n_samples=n_samples
    )

    # NaN-safe latents for t-SNE
    z_np, labels_np = _sanitize_latents(z, labels, what="t-SNE (progress)")
    if z_np is None:
        return None

    # t-SNE projection
    z2d = TSNE(
        n_components=2, init="pca", perplexity=30, learning_rate="auto"
    ).fit_transform(z_np)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc = axes[0].scatter(
        z2d[:, 0], z2d[:, 1], c=labels_np, cmap="tab10", s=6, alpha=0.7
    )
    axes[0].set_title(
        f"{bottleneck.upper()}: Latent space (t-SNE) at epoch {epoch}", fontsize=12
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(sc, ax=axes[0], label="Digit")

    # Reconstructions
    grid = torch.cat([xb, recon])
    grid = vutils.make_grid(grid.view(-1, 1, 28, 28), nrow=8, pad_value=1)
    img = grid.permute(1, 2, 0).cpu().numpy()
    if img.shape[-1] == 1:
        img = img[..., 0]
        axes[1].imshow(img, cmap="gray")
    else:
        axes[1].imshow(img)
    axes[1].axis("off")
    axes[1].set_title(f"{bottleneck.upper()}: Reconstructions (Epoch {epoch})")

    plt.tight_layout()

    fname = None
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fname = os.path.join(
            save_path, f"{bottleneck}_progress_epoch_{epoch:03d}.png"
        )
        plt.savefig(fname, dpi=150)
    plt.close()

    return fname


################ Plot final model results: Training curves, Latent projections (t-SNE, UMAP, MDS) and reconstructions ################
def plot_final_results(
    model, dataset, training_logs, bottleneck, device=None, n_samples=10000, save_path=None
):
    """
    Plot:
      - training curves
      - latent projections (t-SNE, UMAP, MDS)
      - final reconstructions

    Returns:
      (projections_figure_path, reconstructions_figure_path)
      where each can be None if save_path is None.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z, labels, xb, recon = _get_latents_and_recons(
        model, dataset, device, n_samples=n_samples
    )

    # NaN-safe latents for embeddings
    z_np, labels_np = _sanitize_latents(z, labels, what="final embeddings")
    if z_np is None:
        print(
            "[WARN] plot_final_results: skipping latent projections due to insufficient valid points."
        )
        z2d_tsne = z2d_umap = z2d_mds = None
    else:
        print("Computing embeddings: t-SNE, UMAP, MDS...")
        z2d_tsne = TSNE(
            n_components=2, init="pca", perplexity=30, learning_rate="auto"
        ).fit_transform(z_np)
        z2d_umap = umap.UMAP(n_neighbors=30, min_dist=0.2).fit_transform(z_np)
        z2d_mds = MDS(n_components=2).fit_transform(z_np)

    # Training curves + latent projections
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax_curve = axes[0, 0]
    ax_curve.plot(training_logs["loss"], label="Total loss")
    ax_curve.plot(training_logs["recon"], label="Reconstruction")
    ax_curve.plot(training_logs["kl"], label="KL")
    ax_curve.set_title(f"Training curves for {bottleneck.upper()}")
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Loss")
    ax_curve.legend()

    def scatter_latent(ax, emb, title):
        if emb is None:
            ax.set_title(title + " (skipped)")
            ax.set_xticks([])
            ax.set_yticks([])
            return None
        sc = ax.scatter(
            emb[:, 0], emb[:, 1], c=labels_np, cmap="tab10", s=5, alpha=0.7
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return sc

    scatter_latent(axes[0, 1], z2d_tsne, f"{bottleneck.upper()} t-SNE projection")
    scatter_latent(axes[1, 0], z2d_umap, f"{bottleneck.upper()} UMAP projection")
    scatter_latent(axes[1, 1], z2d_mds, f"{bottleneck.upper()} MDS projection")

    plt.tight_layout()

    proj_fname = None
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        proj_fname = os.path.join(save_path, f"{bottleneck}_final_projections.png")
        plt.savefig(proj_fname, dpi=150)
    plt.close()

    # Reconstructions
    grid = torch.cat([xb, recon])
    grid = vutils.make_grid(grid.view(-1, 1, 28, 28), nrow=8, pad_value=1)
    img = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 4))
    if img.shape[-1] == 1:
        plt.imshow(img[..., 0], cmap="gray")
    else:
        plt.imshow(img)
    plt.axis("off")
    plt.title(f"{bottleneck.upper()}: Final reconstructions")
    plt.tight_layout()

    recon_fname = None
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        recon_fname = os.path.join(save_path, f"{bottleneck}_final_recons.png")
        plt.savefig(recon_fname, dpi=150)
    plt.close()

    return proj_fname, recon_fname


#### Normalize the latent vectors to lie on the n-dimensional probability simplex, and then project them to 2D using a regular n-simplex embedding ####
def plot_dirichlet_simplex_nD(
    model, dataset, bottleneck, device=None, n_points=10000, cmap="tab10", save_path=None
):
    """
    Visualize Dirichlet-like latent vectors on an n-simplex projected to 2D.
    Returns the filename of the saved figure (or None if not saved).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z, labels, _, _ = _get_latents_and_recons(
        model, dataset, device, n_samples=n_points
    )

    # Normalize to lie on the simplex (still in torch)
    z = torch.clamp(z, min=1e-8)
    z = z / z.sum(dim=1, keepdim=True)
    n = z.shape[1]

    # Convert to numpy and drop NaN/inf if any
    z_np, labels_np = _sanitize_latents(
        z.detach().cpu(), labels, what="simplex embedding"
    )
    if z_np is None:
        print("[WARN] plot_dirichlet_simplex_nD: skipping simplex plot (no valid points).")
        return None

    # Optional: reduce high-dimensional latent space for visualization
    if n > 10:
        print(f"Latent dim = {n}, reducing to 10D for visualization.")
        z_np = PCA(n_components=10).fit_transform(z_np)
        z_np = np.maximum(z_np, 1e-8)
        z_np = z_np / z_np.sum(axis=1, keepdims=True)
        n = 10

    # Generate vertices of a regular n-simplex in 2D
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vertices = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Map simplex coordinates to 2D
    pos = z_np @ vertices

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(pos[:, 0], pos[:, 1], c=labels_np, cmap=cmap, s=10, alpha=0.7)

    # Draw simplex edges
    for i in range(n):
        for j in range(i + 1, n):
            ax.plot(
                [vertices[i, 0], vertices[j, 0]],
                [vertices[i, 1], vertices[j, 1]],
                "k--",
                lw=0.5,
                alpha=0.3,
            )

    # Draw vertices
    ax.scatter(
        vertices[:, 0], vertices[:, 1], c="black", s=50, marker="x", zorder=5
    )
    for i, (x, y) in enumerate(vertices):
        ax.text(x * 1.1, y * 1.1, f"Comp {i + 1}", ha="center", va="center", fontsize=9)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(scatter, ax=ax, label="Class label")
    ax.set_title(
        f"{bottleneck.upper()}: Latent vectors on simplex (n={n})", fontsize=12
    )
    plt.tight_layout()

    fname = None
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fname = os.path.join(save_path, f"{bottleneck}_dirichlet_simplex.png")
        plt.savefig(fname, dpi=150)
    plt.close()

    return fname


######################## Return latent vectors, labels, and example reconstructions ########################
def _get_latents_and_recons(model, dataset, device, n_samples=5000):
    """
    Helper to:
      - sample up to n_samples examples
      - compute latent z and labels
      - return a small batch of originals + reconstructions for visualization
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    zs, labels = [], []
    xb_vis = None

    for xb, yb in loader:
        xb = xb.to(device)
        xb = xb.view(xb.size(0), -1)
        with torch.no_grad():
            logits, z, _, _ = model(xb)
        zs.append(z.cpu())
        labels.append(yb)
        if xb_vis is None:
            xb_vis = xb[:8]
        if len(torch.cat(zs)) > n_samples:
            break

    z = torch.cat(zs)[:n_samples]
    labels = torch.cat(labels)[:n_samples]
    with torch.no_grad():
        logits, _, _, _ = model(xb_vis)
        recon = torch.sigmoid(logits).cpu()

    return z, labels, xb_vis.cpu(), recon


def visualize_model(model, epoch, loader, device, save_dir,
                    model_name="Model", n_samples=8, tsne_samples=1000):

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # -----------------------------
    # Get batch for reconstructions
    # -----------------------------
    xb, yb = next(iter(loader))
    xb = xb.to(device).view(xb.size(0), -1)

    with torch.no_grad():
        if model_name == "dirichlet":
            logits, z, _, _ = model(xb)
        elif model_name == "gaussian":
            logits, mu, logvar, z = model(xb)
        else:
            raise ValueError("Invalid model_name")

        recon = torch.sigmoid(logits)

    xb_np = xb.cpu().numpy()
    recon_np = recon.cpu().numpy()

    # Remove NaNs
    mask = ~np.isnan(recon_np).any(axis=1)
    xb_np = xb_np[mask]
    recon_np = recon_np[mask]

    # -----------------------------
    # Collect latent samples for t-SNE
    # -----------------------------
    z_all, y_all = [], []

    with torch.no_grad():
        for xb2, yb2 in loader:
            xb2 = xb2.to(device).view(xb2.size(0), -1)

            if model_name == "dirichlet":
                _, z2, _, _ = model(xb2)
            else:
                _, mu2, logvar2, z2 = model(xb2)

            z_all.append(z2.cpu())
            y_all.append(yb2)

            if sum(len(t) for t in z_all) >= tsne_samples:
                break

    z_all = torch.cat(z_all, dim=0)[:tsne_samples].numpy()
    y_all = torch.cat(y_all, dim=0)[:tsne_samples].numpy()

    # Remove NaNs in z_all
    mask = ~np.isnan(z_all).any(axis=1)
    z_all = z_all[mask]
    y_all = y_all[mask]

    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(z_all)

    # =====================================================================
    # ---------------------- SIDE-BY-SIDE PLOTTING -------------------------
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ----- LEFT: Reconstructions -----
    xb_t = torch.from_numpy(xb_np).float()
    recon_t = torch.from_numpy(recon_np).float()

    # Only keep the first n_samples to avoid huge grids
    xb_t = xb_t[:n_samples]
    recon_t = recon_t[:n_samples]

# Concatenate originals + recons

    grid = torch.cat([xb_t, recon_t])
    grid = vutils.make_grid(grid.view(-1, 1, 28, 28), nrow=8, pad_value=1)
    axes[1].imshow(grid.permute(1, 2, 0))
    axes[1].axis("off")
    axes[1].set_title(f"{model_name.upper()} Reconstructions (Epoch {epoch})")

    # ----- RIGHT: t-SNE -----
    sc = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=y_all,
                         cmap="tab10", s=10, alpha=0.7)
    axes[0].set_title(f"{model_name.upper()} Latent Space (t-SNE) (Epoch {epoch})")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(sc, ax=axes[0])

    # Save combined figure
    out_path = os.path.join(save_dir,
                            f"new_{model_name}_vis_epoch_{epoch}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[Saved] {out_path}")

