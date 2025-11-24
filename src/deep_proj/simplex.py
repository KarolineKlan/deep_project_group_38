import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def get_polygon_vertices(n_vertices, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n_vertices, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius


def plot_latent_simplex(model, loader, device, model_type="gaussian", n_samples=None, save_dir=None, model_name="Model", map="tab10", point_size=50, alpha=0.7, image_zoom=0.55):
    """
    Plot latent simplex + show MNIST images at the corners.
    Works for any dataset because image is obtained directly from loader.dataset[idx].
    """
    os.makedirs(save_dir, exist_ok=True) if save_dir else None
    model.eval()
    
    z_all, y_all, idx_all = [], [], []

    # --- Collect latent vectors ---
    with torch.no_grad():
        for batch_i, (xb, yb) in enumerate(loader):
            xb_flat = xb.to(device).view(xb.size(0), -1)

            if model_type.lower() == "dirichlet":
                _, z, _, _ = model(xb_flat)
            else:
                _, mu, logvar, z = model(xb_flat)

            z_all.append(z.cpu())
            y_all.append(yb)

            # track dataset indices
            start = batch_i * loader.batch_size
            idx_all.append(torch.arange(start, start + len(xb)))

            if n_samples is not None and sum(len(t) for t in z_all) >= n_samples:
                break

    z_all = torch.cat(z_all)
    y_all = torch.cat(y_all)
    idx_all = torch.cat(idx_all)

    if n_samples is not None:
        z_all = z_all[:n_samples]
        y_all = y_all[:n_samples]
        idx_all = idx_all[:n_samples]

    # --- Normalize latent vectors to simplex ---
    z_all = z_all / (z_all.sum(dim=1, keepdim=True) + 1e-8)
    z_np = z_all.numpy()

    #print(z_np.shape)
    #print(z_np[:5])

    # Compute mask BEFORE filtering
    mask = ~np.isnan(z_np).any(axis=1)

    # Apply mask to *all* arrays
    z_np = z_np[mask]
    y_all = y_all[mask]
    idx_all = idx_all[mask]

    #print(z_np.shape)
    #print(z_np[:5])

    latent_dim = z_np.shape[1]
    vertices = get_polygon_vertices(latent_dim)

    # --- Project latents to 2D ---
    projected = z_np @ vertices

    # --- Plot ---
    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    # Draw edges
    for i, j in itertools.combinations(range(latent_dim), 2):
        ax.plot([vertices[i,0], vertices[j,0]], [vertices[i,1], vertices[j,1]], 'k--', alpha=0.4)

    # plot points
    sc = ax.scatter(projected[:,0], projected[:,1], c=y_all.numpy(), s=point_size, alpha=alpha, edgecolor="k", linewidth=0.3, cmap="tab10")
    fig = plt.gcf()
    fig.colorbar(sc, ax=ax)

    # === Place original images at vertices ===
    for v_idx in range(latent_dim):
        # target corner vector
        corner_vec = np.eye(latent_dim)[v_idx]

        # find closest point
        dists = np.linalg.norm(z_np - corner_vec, axis=1)
        closest = np.argmin(dists)
        img_idx = idx_all[closest].item()

        # get image from dataset
        img_tensor, _ = loader.dataset[img_idx]
        img = img_tensor.squeeze().numpy()

        imagebox = OffsetImage(img, cmap="gray", zoom=image_zoom)
        ab = AnnotationBbox(imagebox, vertices[v_idx], frameon=False, zorder=10)
        ax.add_artist(ab)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(f"{model_name.upper()} Latent Simplex")

    if save_dir:
        fname = os.path.join(save_dir, f"{model_name}_latent_simplex.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        return fname
    else:
        plt.show()
        return None
