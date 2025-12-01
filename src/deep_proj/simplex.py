import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def get_polygon_vertices(n_vertices, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n_vertices, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius


def plot_latent_simplex(model, dataset_name, loader, device, model_type="gaussian", class_labels=None, n_samples=None, save_dir=None, model_name="Model", map="tab10", point_size=50, alpha=0.7, image_zoom=0.55):
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

            # undo normalization just like in training
            if dataset_name.lower() == "mnist":
                xb_flat = xb_flat * 0.3081 + 0.1307
            elif dataset_name.lower() == "medmnist":
                xb_flat = xb_flat * 0.5 + 0.5

            if model_type.lower() == "dirichlet":
                _, _, _, z = model(xb_flat)   # returns 4 values
            elif model_type.lower() == "gaussian":
                _, mu, logvar, z = model(xb_flat)   # returns 4 values
            elif model_type.lower() == "cc":
                _, z, _ = model(xb_flat)   # returns 3 values
            else:
                raise ValueError(f"Unknown model_type: {model_type}")


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

    y_np = y_all.cpu().numpy() if hasattr(y_all, "cpu") else np.array(y_all)

    # --- Determine which classes to include in legend & color mapping ---
    if class_labels is None:
        unique_classes = np.unique(y_np)
        unique_classes.sort()
    else:
        unique_classes = np.array(class_labels)

    # --- Build mapping: class_id -> tab10 color ---
    base_cmap = plt.get_cmap("tab10")
    try:
        palette = list(base_cmap.colors)
    except AttributeError:
        palette = [base_cmap(i) for i in range(base_cmap.N)]

    color_map = {int(cls): palette[int(cls) % len(palette)] for cls in unique_classes}

    # --- Map each point to its corresponding color ---
    point_colors = [color_map[int(lbl)] for lbl in y_np]

    # --- Plot scatter with exact colors ---
    sc = ax.scatter(projected[:, 0], projected[:, 1], c=point_colors, s=point_size, alpha=alpha, edgecolor="k", linewidth=0.3)

    # --- Legend matching exact class colors ---
    legend_handles = [
        plt.Line2D( [0], [0], marker="o", color=color_map[int(cls)], linestyle="", markersize=7, label=str(int(cls)))
        for cls in unique_classes
    ]

    fig = plt.gcf()
    fig.legend(handles=legend_handles, ncol=len(unique_classes), loc="lower center", bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=15)


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
