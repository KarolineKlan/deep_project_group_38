import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE, MDS
import umap
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from sklearn.decomposition import PCA


######################## Plot t-SNE of latent space and example reconstructions during training ########################
def plot_training_progress(model, dataset, epoch, device=None, n_samples=10000, save_path=None):
    """
    model: trained generative model
    dataset: dataset to sample from
    epoch: current epoch number
    device: computation device (CPU or GPU)
    n_samples: number of samples to generate for plotting
    save_path: path to save the generated plot
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z, labels, xb, recon = _get_latents_and_recons(model, dataset, device, n_samples=n_samples)

    # t-SNE projection
    z2d = TSNE(n_components=2, init='pca', perplexity=30, learning_rate='auto').fit_transform(z)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc = axes[0].scatter(z2d[:, 0], z2d[:, 1], c=labels, cmap="tab10", s=6, alpha=0.7)
    axes[0].set_title(f"Latent space (t-SNE) at epoch {epoch}", fontsize=12)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(sc, ax=axes[0], label="Digit")

    # Reconstructions
    grid = torch.cat([xb, recon])
    grid = vutils.make_grid(grid.view(-1, 1, 28, 28), nrow=8, pad_value=1)
    axes[1].imshow(grid.permute(1, 2, 0))
    axes[1].axis("off")
    axes[1].set_title(f"Reconstructions (Epoch {epoch})")

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/progress_epoch_{epoch:03d}.png", dpi=150)
    plt.show()



################ Plot final model results: Training curves, Latent projections (t-SNE, UMAP, MDS) and reconstructions ################
def plot_final_results(model, dataset, training_logs, device=None, n_samples=10000, save_path=None):
    """
    model: trained generative model
    dataset: dataset to sample from
    training_logs: dictionary containing training loss logs
    device: computation device (CPU or GPU)
    n_samples: number of samples to generate for plotting
    save_path: path to save the generated plots
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z, labels, xb, recon = _get_latents_and_recons(model, dataset, device, n_samples=n_samples)

    # t-SNE, UMAP and MDS projections
    print("Computing embeddings: t-SNE, UMAP, MDS...")
    z2d_tsne = TSNE(n_components=2, init='pca', perplexity=30, learning_rate='auto').fit_transform(z)
    z2d_umap = umap.UMAP(n_neighbors=30, min_dist=0.2).fit_transform(z)
    z2d_mds = MDS(n_components=2).fit_transform(z)

    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_curve = axes[0, 0]
    ax_curve.plot(training_logs["loss"], label="Total loss")
    ax_curve.plot(training_logs["recon"], label="Reconstruction")
    ax_curve.plot(training_logs["kl"], label="KL")
    ax_curve.set_title("Training curves")
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Loss")
    ax_curve.legend()

    # Latent projections
    def scatter_latent(ax, emb, title):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return sc

    scatter_latent(axes[0, 1], z2d_tsne, "t-SNE projection")
    scatter_latent(axes[1, 0], z2d_umap, "UMAP projection")
    scatter_latent(axes[1, 1], z2d_mds, "MDS projection")

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/final_projections.png", dpi=150)
    plt.show()

    # Reconstructions
    grid = torch.cat([xb, recon])
    grid = vutils.make_grid(grid.view(-1, 1, 28, 28), nrow=8, pad_value=1)
    plt.figure(figsize=(10, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title("Final reconstructions")
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/final_recons.png", dpi=150)
    plt.show()



#### Normalize the latent vectors to lie on the n-dimensional probability simplex, and then project them to 2D using a regular n-simplex embedding ####
def plot_dirichlet_simplex_nD(model, dataset, device=None, n_points=10000, cmap="tab10"):
    """
    model: trained generative model with Dirichlet latent space
    dataset: dataset to sample from
    device: computation device (CPU or GPU)
    n_points: number of points to sample for plotting
    cmap: colormap for class labels
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z, labels, _, _ = _get_latents_and_recons(model, dataset, device, n_samples=n_points)

    # Normalize to lie on the simplex
    z = torch.clamp(z, min=1e-8)
    z = z / z.sum(dim=1, keepdim=True)
    n = z.shape[1]

    # Optional: reduce high-dimensional latent space for visualization
    if n > 10:
        print(f"Latent dim = {n}, reducing to 10D for visualization.")
        z = PCA(n_components=10).fit_transform(z)
        z = np.maximum(z, 1e-8)
        z = z / z.sum(axis=1, keepdims=True)
        n = 10

    # Generate vertices of a regular n-simplex in 2D
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    vertices = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Map simplex coordinates to 2D
    pos = z @ vertices

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(pos[:, 0], pos[:, 1], c=labels, cmap=cmap, s=10, alpha=0.7)

    # Draw simplex edges
    for i in range(n):
        for j in range(i + 1, n):
            ax.plot([vertices[i, 0], vertices[j, 0]],
                    [vertices[i, 1], vertices[j, 1]],
                    'k--', lw=0.5, alpha=0.3)

    # Draw vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], c="black", s=50, marker="x", zorder=5)
    for i, (x, y) in enumerate(vertices):
        ax.text(x * 1.1, y * 1.1, f"Comp {i+1}", ha="center", va="center", fontsize=9)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(scatter, ax=ax, label="Class label")
    ax.set_title(f"Latent vectors on simplex (n={n})", fontsize=12)
    plt.tight_layout()
    plt.show()


######################## Return latent vectors, labels, and example reconstructions ########################
def _get_latents_and_recons(model, dataset, device, n_samples=5000):
    """
    model: trained generative model
    dataset: dataset to sample from
    device: computation device (CPU or GPU)
    n_samples: number of samples to generate
    """

    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    zs, labels = [], []
    xb_vis = None

    for xb, yb in loader:
        xb = xb.to(device)
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