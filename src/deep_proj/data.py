# src/deep_proj/data.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import hydra
from omegaconf import DictConfig
from medmnist import INFO


# ------------------------------------------------------------
# Dataset builders
# ------------------------------------------------------------
def _build_mnist(cfg: DictConfig):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) #https://stackoverflow.com/questions/63746182/correct-way-of-normalizing-and-scaling-the-mnist-dataset
    ])
    train_dataset = datasets.MNIST(
        cfg.data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        cfg.data_root, train=False, download=True, transform=transform
    )

    # ---------------------------
    # CLASS FILTERING IF USED
    # ---------------------------
    if cfg.get("mnist_classes") is not None:
        selected = set(int(x) for x in cfg.mnist_classes)

        def to_int(l):
            # MNIST labels can be Python ints or 0-dim tensors
            if isinstance(l, torch.Tensor):
                return int(l.item())
            return int(l)

        # ---- Filter train dataset ----
        train_labels = [to_int(l) for l in train_dataset.targets]
        train_mask = torch.tensor([l in selected for l in train_labels])
        train_indices = torch.where(train_mask)[0]
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

        # ---- Filter test dataset ----
        test_labels = [to_int(l) for l in test_dataset.targets]
        test_mask = torch.tensor([l in selected for l in test_labels])
        test_indices = torch.where(test_mask)[0]
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

        # Debug prints (safe to keep)
        print(f"[MNIST FILTER] Selected classes: {selected}")
        print(f"[MNIST FILTER] Train samples: {len(train_dataset)}")
        print(f"[MNIST FILTER] Test samples:  {len(test_dataset)}")

    return train_dataset, test_dataset



def _build_medmnist(cfg: DictConfig):
    """Alternative loader (uses MedMNIST). Requires: pip install medmnist"""
    from medmnist import INFO
    import medmnist as mm

    data_flag = cfg.medmnist_subset  # e.g. "pathmnist"
    info = INFO[data_flag]
    DataClass = getattr(mm, info["python_class"])

    # Simple 0.5/0.5 normalization (works for 1- or 3-channel subsets)
    mean = [0.5] * info["n_channels"]
    std = [0.5] * info["n_channels"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = DataClass(
        split="train", download=True, root=cfg.data_root, transform=transform
    )
    test_dataset = DataClass(
        split="test", download=True, root=cfg.data_root, transform=transform
    )
    return train_dataset, test_dataset


# ------------------------------------------------------------
# Loader entrypoint
# ------------------------------------------------------------
def get_dataloaders(cfg: DictConfig):
    """Generic entry point — chooses dataset based on cfg.dataset (flat key)."""
    name = cfg.dataset.lower()

    if name == "mnist":
        train_dataset, test_dataset = _build_mnist(cfg)
    elif name == "medmnist":
        train_dataset, test_dataset = _build_medmnist(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    val_size = int(len(train_dataset) * cfg.val_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_subset, batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

###### ALL BELOW HERE IS NOT USED UNLESS RUN DIRECTLY ######
# ------------------------------------------------------------
# Hydra entry + visualization only used when running this script directly
# ------------------------------------------------------------
@hydra.main(config_path="../../configs", config_name="base_config", version_base="1.3")
def main(cfg: DictConfig):
    import matplotlib.pyplot as plt
    import torch

    dataset_name = cfg.dataset.lower()

    # ------------------------------------------------------------
    # Build datasets deterministically (no random_split, no shuffle)
    # ------------------------------------------------------------
    if dataset_name == "mnist":
        train_dataset, _ = _build_mnist(cfg)
        # MNIST label names
        label_map = {i: str(i) for i in range(10)}
        # unnormalize params for viewing
        def unnorm(x):
            return (x * 0.3081 + 0.1307).clamp(0, 1)
        cmap = "gray"

    elif dataset_name == "medmnist":
        train_dataset, _ = _build_medmnist(cfg)

        from medmnist import INFO
        label_map = INFO[cfg.medmnist_subset]["label"]
        # normalize keys to int (sometimes INFO uses string keys)
        label_map = {int(k): v for k, v in label_map.items()}

        def unnorm(x):
            return (x * 0.5 + 0.5).clamp(0, 1)
        cmap = None

    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    # ------------------------------------------------------------
    # Collect a non-random, balanced set:
    # first n_per_class samples per label from train_dataset in order
    # ------------------------------------------------------------
    n_per_class = 6  # number of images per label (columns)
    class_ids = sorted(label_map.keys())

    collected = {c: [] for c in class_ids}

    # scan dataset in order (deterministic)
    for idx in range(len(train_dataset)):
        img, lab = train_dataset[idx]

        # lab might be tensor, numpy scalar, or shape (1,)
        if isinstance(lab, torch.Tensor):
            lab_int = int(lab.squeeze().item())
        else:
            # numpy or python int
            lab_int = int(lab)

        if lab_int in collected and len(collected[lab_int]) < n_per_class:
            collected[lab_int].append(img)

        # stop early if we have enough for all classes
        if all(len(collected[c]) >= n_per_class for c in class_ids):
            break

    # ------------------------------------------------------------
    # Plot with a left text column (labels)
    # Always show all labels (one row per class)
    # ------------------------------------------------------------
    label_col_ratio = 2.3
    img_col_ratio   = 1.0
    n_per_class = 6
    rows = len(class_ids)

    total_ratio_units = label_col_ratio + n_per_class * img_col_ratio
    unit_w = 1.1
    fig_w = total_ratio_units * unit_w
    fig_h = rows * 1.1

    fig, axes = plt.subplots(
        rows, n_per_class + 1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [label_col_ratio] + [img_col_ratio] * n_per_class}
    )

    title = f"Sample batch sorted by label ({cfg.dataset}"
    if dataset_name == "medmnist":
        title += f"/{cfg.medmnist_subset}"
    title += ")"
    fig.suptitle(title, fontsize=20, x=0.5) #to put it more to the center of the figure you would write: plt.suptitle(title, x=0.5)

    if rows == 1:
        axes = axes[None, :]

    for r, c_id in enumerate(class_ids):
        # --- left label cell ---
        ax_label = axes[r, 0]
        ax_label.axis("off")
        ax_label.text(
            1, 0.5, label_map[c_id],   # near the right edge of label column
            va="center", ha="right",
            fontsize=16
        )

        # --- image cells ---
        imgs_for_class = collected[c_id]
        for k in range(n_per_class):
            ax = axes[r, k + 1]
            ax.axis("off")

            if k >= len(imgs_for_class):
                continue

            img = unnorm(imgs_for_class[k])
            if img.shape[0] == 1:
                ax.imshow(img[0], cmap="gray")
            else:
                ax.imshow(img.permute(1, 2, 0), cmap=cmap)

    # first let matplotlib pack stuff...
    plt.tight_layout()

    # ...then kill the outer gutter explicitly
    plt.subplots_adjust(left=-0.13, right=0.995, top=0.93, wspace=0.05, hspace=0.15)

    plt.savefig("trash_outputs/sample_batch_sorted.png", dpi=150)
    #plt.show()


if __name__ == "__main__":
    # Hydra will call main(cfg) with your base_config
    main()
