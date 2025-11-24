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
import matplotlib.image as mpimg

def plot_latent(model, epoch, loader, device, save_dir,
                    model_name, tsne_samples=1000,eval=False):
    
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # -----------------------------
    # Collect latent samples for t-SNE
    # -----------------------------
    z_all, y_all = [], []

    with torch.no_grad():
        for xb2, yb2 in loader:
            xb2 = xb2.to(device).view(xb2.size(0), -1)

            # only denormalize for evaluation plots
            if eval: 
                if model_name == "mnist":
                    xb2 = xb2 * 0.3081 + 0.1307
                elif model_name == "medmnist":
                    xb2 = xb2 * 0.5 + 0.5

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

    # ----- t-SNE -----
    fig = plt.figure(figsize=(6, 6))
    sc = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_all,
                         cmap="tab10", s=10, alpha=0.7)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(sc)
    fname = None
    if eval==False: 
        plt.title(f"{model_name.upper()} Latent Space (t-SNE) (Epoch {epoch})")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(
                save_dir, f"{model_name}_latent_epoch_{epoch:03d}.png"
            )
            plt.savefig(fname, dpi=150)
        plt.close()
    elif eval==True:
        plt.title(f"{model_name.upper()} Test Latent Space (t-SNE)")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(
                save_dir, f"{model_name}_test_latent.png"
            )
            plt.savefig(fname, dpi=150)
        plt.close()
    return fname

def plot_recons(model, epoch, loader, device, save_dir,
                    model_name, n_samples=8, eval=False): 
    
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    xb, yb = next(iter(loader))
    xb = xb.to(device).view(xb.size(0), -1)

    #  only denormalize for evaluation plots
    if eval:
        if model_name == "mnist":
            xb = xb * 0.3081 + 0.1307
        elif model_name == "medmnist":
            xb = xb * 0.5 + 0.5
    
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
    
    # ----- LEFT: Reconstructions -----
    xb_t = torch.from_numpy(xb_np).float()
    recon_t = torch.from_numpy(recon_np).float()

    # Only keep the first n_samples to avoid huge grids
    xb_t = xb_t[:n_samples]
    recon_t = recon_t[:n_samples]


    grid = torch.cat([xb_t, recon_t], dim=0).view(2, n_samples, 1, 28, 28) #torch.cat([xb_t, recon_t])
    grid = vutils.make_grid(grid.view(-1, 1, 28, 28), nrow=n_samples, pad_value=1)
    
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    fname = None
    if eval==False:
        plt.title(f"{model_name.upper()} Reconstructions (Epoch {epoch})")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(
                save_dir, f"{model_name}_recon_epoch_{epoch:03d}.png"
            )
            plt.savefig(fname, dpi=150)
        plt.close()

    elif eval==True: 
        plt.title(f"{model_name.upper()} Test Reconstructions")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(
                save_dir, f"{model_name}_test_recon.png"
            )
            plt.savefig(fname, dpi=150)
        plt.close()
    return fname

def plot_side_by_side(img_path1, img_path2, save_dir, bottleneck, epoch):
    """
    Load two images from disk and display them side by side.
    
    Args:
        img_path1 (str): Path to first image.
        img_path2 (str): Path to second image.
        title1 (str): Optional title for first image.
        title2 (str): Optional title for second image.
    """

    img1 = mpimg.imread(img_path1)
    img2 = mpimg.imread(img_path2)

    plt.figure(figsize=(12, 6))

    # Left image
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.axis("off")

    # Right image
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.axis("off")
    plt.tight_layout()
    
    fname = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(
                save_dir, f"{bottleneck}_progress_{epoch:03d}.png"
            )
        plt.savefig(fname, dpi=150)
    plt.close()
    return fname
    


def plot_training_progress(model, epoch, loader, device, save_dir,
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

    fname = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(
            save_dir, f"{model_name}_progress_epoch_{epoch:03d}.png"
        )
        plt.savefig(fname, dpi=150)
    plt.close()

    return fname
def plot_training_loss(training_logs,bottleneck, save_path,device=None): 
    # Training curves + latent projections
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plt.figure(figsize=(8, 6))  # Single figure
    plt.plot(training_logs["loss"], label="Total train loss")
    plt.plot(training_logs["recon"], label="Train reconstruction")
    plt.plot(training_logs["kl"], label="Train KL")
    plt.plot(training_logs["val_loss"], label="Total val loss")
    plt.plot(training_logs["val_recon"], label="Val reconstruction")
    plt.plot(training_logs["val_kl"], label="Val KL")
    plt.title(f"Training curves for {bottleneck.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    
    curve_fname = None
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        curve_fname = os.path.join(save_path, f"{bottleneck}_training_curves.png")
        plt.savefig(curve_fname, dpi=150)
    plt.close()
    return curve_fname




# def plot_final_results(
#     model, dataset, training_logs, bottleneck, device=None, n_samples=10000, save_path=None
# ):  



#     """
#     Plot:
#       - training curves
#       - latent projections (t-SNE, UMAP, MDS)
#       - final reconstructions

#     Returns:
#       (projections_figure_path, reconstructions_figure_path)
#       where each can be None if save_path is None.
#     """
#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     z, labels, xb, recon = _get_latents_and_recons(
#         model, dataset, device, n_samples=n_samples
#     )

#     # NaN-safe latents for embeddings
#     z_np, labels_np = _sanitize_latents(z, labels, what="final embeddings")
#     if z_np is None:
#         print(
#             "[WARN] plot_final_results: skipping latent projections due to insufficient valid points."
#         )
#         z2d_tsne = z2d_umap = z2d_mds = None
#     else:
#         print("Computing embeddings: t-SNE, UMAP, MDS...")
#         z2d_tsne = TSNE(
#             n_components=2, init="pca", perplexity=30, learning_rate="auto"
#         ).fit_transform(z_np)
#         z2d_umap = umap.UMAP(n_neighbors=30, min_dist=0.2).fit_transform(z_np)
#         z2d_mds = MDS(n_components=2).fit_transform(z_np)

#     # Training curves + latent projections
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))

#     ax_curve = axes[0, 0]
#     ax_curve.plot(training_logs["loss"], label="Total loss")
#     ax_curve.plot(training_logs["recon"], label="Reconstruction")
#     ax_curve.plot(training_logs["kl"], label="KL")
#     ax_curve.set_title(f"Training curves for {bottleneck.upper()}")
#     ax_curve.set_xlabel("Epoch")
#     ax_curve.set_ylabel("Loss")
#     ax_curve.legend()

#     def scatter_latent(ax, emb, title):
#         if emb is None:
#             ax.set_title(title + " (skipped)")
#             ax.set_xticks([])
#             ax.set_yticks([])
#             return None
#         sc = ax.scatter(
#             emb[:, 0], emb[:, 1], c=labels_np, cmap="tab10", s=5, alpha=0.7
#         )
#         ax.set_title(title)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         return sc

#     scatter_latent(axes[0, 1], z2d_tsne, f"{bottleneck.upper()} t-SNE projection")
#     scatter_latent(axes[1, 0], z2d_umap, f"{bottleneck.upper()} UMAP projection")
#     scatter_latent(axes[1, 1], z2d_mds, f"{bottleneck.upper()} MDS projection")

#     plt.tight_layout()

#     proj_fname = None
#     if save_path:
#         os.makedirs(save_path, exist_ok=True)
#         proj_fname = os.path.join(save_path, f"{bottleneck}_final_projections.png")
#         plt.savefig(proj_fname, dpi=150)
#     plt.close()

#     # Reconstructions
#     grid = torch.cat([xb, recon])
#     grid = vutils.make_grid(grid.view(-1, 1, 28, 28), nrow=8, pad_value=1)
#     img = grid.permute(1, 2, 0).cpu().numpy()

#     plt.figure(figsize=(10, 4))
#     if img.shape[-1] == 1:
#         plt.imshow(img[..., 0], cmap="gray")
#     else:
#         plt.imshow(img)
#     plt.axis("off")
#     plt.title(f"{bottleneck.upper()}: Final reconstructions")
#     plt.tight_layout()

#     recon_fname = None
#     if save_path:
#         os.makedirs(save_path, exist_ok=True)
#         recon_fname = os.path.join(save_path, f"{bottleneck}_final_recons.png")
#         plt.savefig(recon_fname, dpi=150)
#     plt.close()

#     return proj_fname, recon_fname