import pickle
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

from bw_ml_common.datasets.data_accessor_factory import create_dataset_accessor

from dino_bw.bw_dino_defs import IMAGENET_MEAN, IMAGENET_STD, MODEL_TO_NUM_LAYERS


def setup_patch_quantization_filter(patch_size: int):
    """Create the patch quantization filter for converting pixel masks to patch-level labels."""
    patch_quant_filter = torch.nn.Conv2d(1, 1, patch_size, stride=patch_size, bias=False)
    patch_quant_filter.weight.data.fill_(1.0 / (patch_size * patch_size))
    return patch_quant_filter


def resize_transform(image: Image.Image, image_size: int, patch_size: int) -> torch.Tensor:
    """
    Resize transform to dimensions divisible by patch size while maintaining aspect ratio.

    Args:
        image: PIL Image to resize
        image_size: Target height in pixels
        patch_size: Patch size for alignment

    Returns:
        Resized image as tensor
    """
    w, h = image.size
    h_patches = int(image_size / patch_size)
    w_patches = int(w / h * (image_size / patch_size))
    target_size = (h_patches * patch_size, w_patches * patch_size)
    return TF.to_tensor(TF.resize(image, target_size))


def load_dinov3_model(dinov3_location: Path, model_name: str, checkpoint_path: Path):
    """Load the DINOv3 model."""
    print(f"Loading DINOv3 model: {model_name}")
    model = torch.hub.load(repo_or_dir=str(dinov3_location), model=model_name, source="local",
                           weights=str(checkpoint_path))
    model.cuda()
    model.eval()
    print("Model loaded successfully")
    return model


def get_class_idx_from_name(accessor, class_name: str) -> int:
    """
    Get class index from class name using the dataset accessor.

    Args:
        accessor: Dataset accessor with label mapping
        class_name: Name of the class (e.g., 'tree', 'sky', etc.)

    Returns:
        Class index/ID

    Raises:
        ValueError: If class name not found
    """
    label_config = accessor.label_mapping.labels_by_name.get(class_name, None)
    if label_config is None:
        label_classes = accessor.label_mapping.labels_by_name.keys()
        available_classes = list(label_classes)
        raise ValueError(f"Class '{class_name}' not found. Available classes: {available_classes}")

    return label_config.id


def class_idx_to_binary_mask(class_idx_mask: np.array, class_idx: int) -> Image.Image:
    chosen_cls_mask = np.zeros_like(class_idx_mask)
    chosen_cls_mask[class_idx_mask == class_idx] = 255
    return Image.fromarray(chosen_cls_mask)


def extract_patch_features(
        image: Image.Image,
        model: torch.nn.Module,
        n_layers: int,
        image_size: int,
        patch_size: int,
) -> (torch.Tensor, Tuple):
    """
    Extract patch-level features from an image.

    Args:
        image: RGB image
        model: DINOv3 model
        n_layers: Number of layers in the model

    Returns:
        Features tensor
    """
    # Resize and normalize image
    image_resized = resize_transform(image.convert("RGB"), image_size, patch_size)
    image_normalized = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    image_batch = image_normalized.unsqueeze(0).cuda()

    # Extract features
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            feats = model.get_intermediate_layers(image_batch, n=range(n_layers), reshape=True, norm=True)
            dim = feats[-1].shape[1]
            features = feats[-1].squeeze().view(dim, -1).permute(1, 0).detach().cpu()

    return features, np.array(image_resized.size()[1:])


def extract_patch_labels(
        classes_idx_mask: np.array,
        patch_quant_filter: torch.nn.Module,
        class_idx: int,
        image_size: int,
        patch_size: int,
) -> torch.Tensor:
    """
    Extract patch-level labels from a segmentation mask for a specific class.

    Args:
        mask: RGB segmentation mask
        patch_quant_filter: Patch quantization filter
        class_idx: ID of the target class

    Returns:
        Labels tensor
    """
    # Convert mask to binary mask for specific class
    class_binary_mask = class_idx_to_binary_mask(classes_idx_mask, class_idx)

    # Resize mask and quantize to patch level
    mask_resized = resize_transform(class_binary_mask, image_size, patch_size)
    with torch.no_grad():
        mask_quantized = patch_quant_filter(mask_resized.unsqueeze(0)).squeeze().view(-1).detach().cpu()

    return mask_quantized


def extract_raw_features(
        class_name: str,
        patch_size: int,
        image_size: int,
        dataset_path: Path,
        dinov3_location: Path,
        model_name: str,
        checkpoint_path: Path,
        cache_path: Path,
):
    """Extract raw features and labels from dataset and save to cache."""
    print("=== EXTRACTING RAW FEATURES ===")
    print(f"Class: {class_name}")
    print(f"Dataset path: {dataset_path}")
    print(f"Model: {model_name}")
    print(f"Patch size: {patch_size}, Image size: {image_size}")

    sample_info_keys = ['image_path', 'labels_mask_path']

    # Setup
    print("Setting up patch quantization filter...")
    patch_quant_filter = setup_patch_quantization_filter(patch_size)

    print("Loading DINOv3 model...")
    model = load_dinov3_model(dinov3_location, model_name, checkpoint_path)
    n_layers = MODEL_TO_NUM_LAYERS[model_name]

    # Load dataset
    print(f"Loading {dataset_path.stem} dataset from {dataset_path.parent}")
    accessor = create_dataset_accessor(dataset_name=dataset_path.stem, data_root=str(dataset_path.parent),
                                       split_name="train")
    accessor.update_lookup_tables_with_task()
    samples = accessor.populate(stable=True)

    print(f"Found {len(samples)} training samples")

    # Get class index from class name
    try:
        class_idx = get_class_idx_from_name(accessor, class_name)
        print(f"Class '{class_name}' has index {class_idx}")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Extract features and labels
    all_features = []
    all_labels = []
    all_image_indices = []
    sample_info = []

    print("Extracting features and labels...")
    image_path = ""
    for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
        try:
            # Load image and mask
            image_path = sample["image_path"]
            mask_path = sample["labels_mask_path"]

            if not mask_path:  # Skip samples without masks
                continue

            image = accessor.read_image(image_path)
            rgb_mask_img = accessor.read_labels_mask(mask_path)
            class_labels = accessor.convert_rgb_mask_to_labels(rgb_mask_img)

            # Extract features and labels separately
            features = extract_patch_features(image, model, n_layers, image_size, patch_size)
            chosen_idx_mask = extract_patch_labels(class_labels, patch_quant_filter, class_idx, image_size, patch_size)

            all_features.append(features)
            all_labels.append(chosen_idx_mask)
            all_image_indices.append(i * torch.ones(chosen_idx_mask.shape))
            sample_info.append({'image_path': str(Path(image_path).relative_to(dataset_path)),
                                'label_path': str(Path(mask_path).relative_to(dataset_path))})

        except Exception as e:
            print(f"Error processing sample {i} ({image_path}): {e}")
            continue

    if not all_features:
        print("No valid samples found!")
        return

    # Save raw data to cache
    cache_data = {
        "features": all_features,
        "labels": all_labels,
        "image_indices": all_image_indices,
        "sample_info": sample_info,
        "class_name": class_name,
        "class_idx": class_idx,
        "config": {
            "patch_size": patch_size,
            "image_size": image_size,
            "model_name": model_name,
            "dataset_path": str(dataset_path),
        },
    }

    print(f"Saving raw features cache to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)

    print(f"Raw feature extraction completed! Processed {len(sample_info)} samples")
    return cache_data


def extract_all_features(cache_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Load cached features and convert them to features tensor.

    Args:
        cache_path: Path to the cached features file

    Returns:
        Tuple of (features, labels, image_indices, cache_metadata)
    """
    print("Loading cached features...")
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)

    all_features = cache_data["features"]
    all_labels = cache_data["labels"]
    all_image_indices = cache_data["image_indices"]

    # Concatenate all data
    print("Concatenating extracted data...")
    features_tensor = torch.cat(all_features)
    labels_tensor = torch.cat(all_labels)
    indices_tensor = torch.cat(all_image_indices)

    # Filter patches with clear positive or negative labels
    print("Filtering patches with clear labels...")
    clear_labels_mask = (labels_tensor < 0.01) | (labels_tensor > 0.99)
    filtered_features = features_tensor[clear_labels_mask]
    filtered_labels = labels_tensor[clear_labels_mask]
    filtered_indices = indices_tensor[clear_labels_mask]

    print(f"Original patches: {len(features_tensor)}")
    print(f"Filtered patches: {len(filtered_features)}")
    print(f"Features shape: {filtered_features.shape}")
    print(f"Labels shape: {filtered_labels.shape}")

    return filtered_features, filtered_labels, filtered_indices, cache_data
