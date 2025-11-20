# src/deep_proj/data.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import hydra
from omegaconf import DictConfig


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


# ------------------------------------------------------------
# Hydra entry + visualization only used when running this script directly
# ------------------------------------------------------------
@hydra.main(config_path="../../configs", config_name="base_config", version_base="1.3")
def main(cfg: DictConfig):
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    loaders = get_dataloaders(cfg)

    x, y = next(iter(loaders["train"]))
    print(f"Batch loaded correctly. Shapes: x={x.shape}, y={y.shape}")

    # Visualization only happens when you run this script directly.
    # When other code imports get_dataloaders, this is never executed.
    if cfg.dataset.lower() == "mnist":
        x_vis = x * 0.3081 + 0.1307
        cmap = "gray"
    else:
        x_vis = x * 0.5 + 0.5
        cmap = None

    x_vis = x_vis.clamp(0, 1)

    grid = vutils.make_grid(x_vis[:64], nrow=8, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    if grid_np.shape[2] == 1:
        grid_np = grid_np[..., 0]

    plt.figure(figsize=(8, 8))
    plt.title(f"Sample batch ({cfg.dataset})")
    plt.imshow(grid_np, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("trash_outputs/sample_batch.png", dpi=150)
    plt.show()

    print("Saved visualization: sample_batch.png")


if __name__ == "__main__":
    # Hydra will call main(cfg) with your base_config
    main()
