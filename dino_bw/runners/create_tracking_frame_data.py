#!/usr/bin/env python3
"""
Script to extract DINO embeddings and process corresponding bounding box and segmentation data
for tracking between frames.

This script processes data from the greeting_dev/bags_parsed/turn8_0/ folder structure:
- images/ - raw sequential frames  
- od_detections/json/ - object detection results
- segmentation/labels_classes/ - segmentation masks with class pixel values

The processing is done in stages:
1. Extract DINO embeddings from all images using dino_embeddings_utils
2. Process corresponding bounding box detections 
3. Process segmentation masks
4. Create unified cache structure for easy loading and processing
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import argparse
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dino_bw.bw_defs import MODEL_TO_NUM_LAYERS, DINO_V3_REPO
from dino_bw.utils.dino_embeddings_utils import (
    load_dinov3_model,
    resize_transform,
    extract_patch_features
)


def get_frame_number_from_filename(filename: str) -> int:
    """Extract frame number from filename like '0042_20250710190621145.jpg'"""
    return int(filename.split('_')[0])


def load_detection_data(detections_json_path: Path) -> List[Dict]:
    """Load object detection data from JSON file."""
    if not detections_json_path.exists():
        return []

    with open(detections_json_path, 'r') as f:
        detections = json.load(f)

    return detections


def process_segmentation_mask(mask_path: Path, original_image_size: Tuple[int, int], target_image_size: Tuple[int, int],
                              patch_size: int, tree_class_id: int = 2,
                              tree_threshold: float = 0.25) -> Union[np.ndarray, None]:
    """
    Process segmentation mask to match the DINO patch grid.
    
    Args:
        mask_path: Path to segmentation mask (PNG with class values as pixel intensities)
        original_image_size: (width, height) of the original image
        target_image_size: Target image size  (width, height) for DINO processing
        patch_size: DINO patch size
        enhanced_tree_detection: If True, use threshold-based tree detection for narrow objects
        tree_class_id: Class ID for trees (default: 2 for bw2508)
        tree_threshold: Minimum fraction of tree pixels to label patch as tree (default: 0.25). If set to zero, will check for maximum likelyhood class
        
    Returns:
        Segmentation data resized to match DINO patch grid (1D array of patch-level class indices)
    """
    if not mask_path.exists():
        return None

    # Load segmentation mask (grayscale with class indices as pixel values)
    mask_pil = Image.open(mask_path)
    if mask_pil.mode != 'L':  # Convert to grayscale if needed
        mask_image = mask_pil.convert('L')  # type: ignore
    else:
        mask_image = mask_pil  # type: ignore

    # First resize mask to match original image dimensions if needed
    from PIL.Image import Resampling
    # Resize to match DINO processing dimensions while preserving integer class values
    mask_resized = mask_image.resize(target_image_size, Resampling.NEAREST)

    # Convert to patch-level segmentation by taking the mode in each patch
    h_patches = target_image_size[1] // patch_size
    w_patches = target_image_size[0] // patch_size

    # Reshape to patches and get class labels
    mask_resized_np = np.array(mask_resized)
    patch_classes = np.empty((h_patches, w_patches))
    patch_classes.fill(255)

    for i in range(h_patches):
        for j in range(w_patches):
            patch_start_h = i * patch_size
            patch_end_h = (i + 1) * patch_size
            patch_start_w = j * patch_size
            patch_end_w = (j + 1) * patch_size

            patch_values = mask_resized_np[patch_start_h:patch_end_h, patch_start_w:patch_end_w].flatten()
            # Enhanced tree detection: use threshold-based approach for narrow objects
            if tree_threshold != 0:
                tree_pixels = np.sum(patch_values == tree_class_id)
                total_pixels = len(patch_values)
                tree_fraction = tree_pixels / total_pixels
                if tree_fraction >= tree_threshold:
                    patch_classes[i, j] = tree_class_id
            else:
                # Use most common class for non-tree patches
                unique, counts = np.unique(patch_values, return_counts=True)
                most_common_class = unique[np.argmax(counts)]
                patch_classes[i, j] = most_common_class
    return np.array(patch_classes, dtype=np.int32)  # Ensure integer type


def extract_dino_embeddings_from_folder(
        images_folder: Path,
        dinov3_location: Path,
        model_name: str,
        checkpoint_path: Union[Path, None],
        image_size: int = 768,  # Updated default
        patch_size: int = 16,  # Updated default patch size for DINOv3
) -> Tuple[List[torch.Tensor], List[str], List[Tuple[int, int]], torch.nn.Module]:
    """
    Extract DINO embeddings from all images in a folder.
    
    Returns:
        List of feature tensors, list of image filenames, list of original image sizes, loaded model
    """
    print("=== EXTRACTING DINO EMBEDDINGS ===")
    print(f"Images folder: {images_folder}")
    print(f"Model: {model_name}")
    print(f"Patch size: {patch_size}, Image size: {image_size}")

    # Load DINOv3 model
    print("Loading DINOv3 model...")
    if checkpoint_path is None:
        # Load pretrained model using torch.hub
        print(f"Loading pretrained DINOv3 model: {model_name}")
        model = torch.hub.load(repo_or_dir=str(dinov3_location), model=model_name, source="local")
    else:
        # Load model from checkpoint
        model = load_dinov3_model(dinov3_location, model_name, checkpoint_path)
    model.cuda()
    model.eval()
    print("Model loaded successfully")
    n_layers = MODEL_TO_NUM_LAYERS[model_name]

    # Get sorted list of image files
    image_files = sorted([f for f in images_folder.glob("*.jpg")])
    print(f"Found {len(image_files)} images")

    all_features = []
    image_filenames = []
    original_image_sizes = []

    print("Extracting features from images...")
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            original_size = image.size  # (width, height)

            # Extract features using existing function
            features, image_dims = extract_patch_features(
                image, model, n_layers, image_size, patch_size
            )

            all_features.append(features)
            image_filenames.append(image_path.name)
            original_image_sizes.append(original_size)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    print(f"Successfully extracted features from {len(all_features)} images")
    return all_features, image_filenames, original_image_sizes, model


def scale_bounding_boxes(detections: List[Dict], original_size: Tuple[int, int], target_size: Tuple[int, int]) -> List[
    Dict]:
    """
    Scale bounding boxes from original image coordinates to target image coordinates.
    
    Args:
        detections: List of detection dictionaries with 'box' containing x1,y1,x2,y2
        original_size: (width, height) of original image
        target_size: (width, height) of target image
    
    Returns:
        List of detections with scaled bounding boxes
    """
    if not detections:
        return []

    orig_w, orig_h = original_size
    target_w, target_h = target_size

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    scaled_detections = []
    for detection in detections:
        scaled_detection = detection.copy()
        if 'box' in detection:
            box = detection['box']
            scaled_detection['box'] = {
                'x1': box['x1'] * scale_x,
                'y1': box['y1'] * scale_y,
                'x2': box['x2'] * scale_x,
                'y2': box['y2'] * scale_y
            }
            # Store original box for reference
            scaled_detection['original_box'] = box.copy()
        scaled_detections.append(scaled_detection)

    return scaled_detections


def process_tracking_dataset(
        data_folder: Path,
        cache_path: Path,
        model_name: str = "dinov3_vits16",
        checkpoint_path: Optional[Path] = None,
        dino_size: int = 768,
        patch_size: int = 16,
):
    """
    Main function to process the tracking dataset.
    
    Args:
        data_folder: Path to turn8_0 folder containing images/, od_detections/, segmentation/
        cache_path: Path where to save the processed cache
        model_name: DINOv3 model name to use
        checkpoint_path: Path to model checkpoint (if None, uses pretrained)
        dino_size: Target image size for DINO processing
        patch_size: DINO patch size
    """
    print("=== PROCESSING TRACKING DATASET ===")
    print(f"Data folder: {data_folder}")
    print(f"Cache path: {cache_path}")

    # Define folder paths
    images_folder = data_folder / "images"
    detections_folder = data_folder / "od_detections" / "json"
    segmentation_folder = data_folder / "segmentation" / "labels_classes"

    # Verify folders exist
    if not images_folder.exists():
        raise FileNotFoundError(f"Images folder not found: {images_folder}")
    if not detections_folder.exists():
        print(f"Warning: Detections folder not found: {detections_folder}")
    if not segmentation_folder.exists():
        print(f"Warning: Segmentation folder not found: {segmentation_folder}")

    # Stage 1: Extract DINO embeddings
    print("\n--- STAGE 1: EXTRACTING DINO EMBEDDINGS ---")
    features_list, image_filenames, original_image_sizes, model = extract_dino_embeddings_from_folder(
        images_folder, DINO_V3_REPO, model_name, checkpoint_path, dino_size, patch_size
    )

    # Calculate target image dimensions based on DINO processing
    # The resize_transform function determines the actual target size
    sample_image = Image.open(images_folder / image_filenames[0])
    sample_resized = resize_transform(sample_image, dino_size, patch_size)
    target_image_size = (sample_resized.shape[2], sample_resized.shape[1])  # (width, height)

    # Stage 2: Process detections and segmentation data
    print("\n--- STAGE 2: PROCESSING DETECTIONS AND SEGMENTATION ---")

    frame_data = []

    for i, filename in enumerate(tqdm(image_filenames, desc="Processing frame data")):
        original_size = original_image_sizes[i]

        frame_info = {
            'frame_id': i,
            'filename': filename,
            'frame_number': get_frame_number_from_filename(filename),
            'features': features_list[i].cpu().numpy(),
            'original_image_size': original_size,
            'target_image_size': target_image_size,
            'detections': [],
            'detections_scaled': [],  # Detections scaled to DINO image coordinates
            'segmentation': None
        }

        # Process detections
        base_name = Path(filename).stem  # Remove .jpg extension
        detection_file = detections_folder / f"{base_name}.json"
        if detection_file.exists():
            original_detections = load_detection_data(detection_file)
            frame_info['detections'] = original_detections  # Keep original coordinates
            # Scale detections to match DINO processing coordinates
            frame_info['detections_scaled'] = scale_bounding_boxes(
                original_detections, original_size, target_image_size
            )

        # Process segmentation
        segmentation_file = segmentation_folder / f"{base_name}.png"
        if segmentation_file.exists():
            frame_info['segmentation'] = process_segmentation_mask(
                segmentation_file, original_size, target_image_size, patch_size)

        frame_data.append(frame_info)

    # Stage 3: Create unified cache
    print("\n--- STAGE 3: CREATING UNIFIED CACHE ---")

    cache_data = {
        'metadata': {
            'model_name': model_name,
            'image_size': dino_size,
            'patch_size': patch_size,
            'data_folder': str(data_folder),
            'num_frames': len(frame_data),
            'processing_timestamp': datetime.now().isoformat()
        },
        'frames': frame_data,
        'config': {
            'patch_size': patch_size,
            'image_size': dino_size,
            'model_name': model_name,
            'data_folder': str(data_folder)
        }
    }

    # Save cache
    cache_path = cache_path.parent / f'{cache_path.stem}_img_{dino_size}_patch_{patch_size}.pkl'
    print(f"Saving unified cache to {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"✓ Successfully processed {len(frame_data)} frames")
    print(f"✓ Cache saved to {cache_path}")

    # Print summary statistics
    num_frames_with_detections = sum(1 for frame in frame_data if frame['detections'])
    num_frames_with_segmentation = sum(1 for frame in frame_data if frame['segmentation'] is not None)

    print(f"\nSUMMARY:")
    print(f"  Total frames: {len(frame_data)}")
    print(f"  Frames with detections: {num_frames_with_detections}")
    print(f"  Frames with segmentation: {num_frames_with_segmentation}")
    print(f"  Feature dimensions: {features_list[0].shape if features_list else 'N/A'}")

    return cache_data


def load_tracking_cache(cache_path: Path, patch_size: Optional[int] = None, image_size: Optional[int] = None) -> Union[Dict[str, Any], Tuple[bool, Optional[Dict[str, Any]]]]:
    """
    Load the processed tracking cache and optionally validate parameters.
    
    Args:
        cache_path: Path to the cache file
        patch_size: Expected patch size for validation (optional)
        image_size: Expected image size for validation (optional)
    
    Returns:
        If patch_size and image_size are provided: tuple of (is_valid, cache_data)
        Otherwise: the loaded cache data dictionary
    """
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # If validation parameters are provided, perform validation
        if patch_size is not None and image_size is not None:
            metadata = cache_data.get('metadata')
            if metadata is None:
                return False, None
            
            cache_patch_size = metadata.get('patch_size', None)
            cache_image_size = metadata.get('image_size', None)
            
            if cache_patch_size != patch_size or cache_image_size != image_size:
                return False, None
            
            return True, cache_data
        
        # If no validation parameters, return the cache data
        return cache_data
        
    except Exception:
        return (False, None) if (patch_size is not None and image_size is not None) else {}


def main():
    parser = argparse.ArgumentParser(description="Process tracking dataset and extract DINO embeddings")
    parser.add_argument("--data_folder", type=Path, required=True,
                        help="Path to folder containing images/, od_detections/, segmentation/")
    parser.add_argument("--cache_path", type=Path, required=True,
                        help="Output path for the processed cache file")
    parser.add_argument("--model_name", type=str, default="dinov3_vits16",
                        choices=list(MODEL_TO_NUM_LAYERS.keys()),
                        help="DINOv3 model name to use")
    parser.add_argument("--checkpoint_path", type=Path,
                        default="/home/nati/source/dinov3/checkpoints/dinov3_vits16_pretrain.pth",
                        help="Path to model checkpoint (if None, uses pretrained)")
    parser.add_argument("--image_size", type=int, default=768,
                        help="Target image size for DINO processing")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="DINO patch size")

    args = parser.parse_args()
    process_tracking_dataset(
        data_folder=args.data_folder,
        cache_path=args.cache_path,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        dino_size=args.image_size,
        patch_size=args.patch_size,
    )


if __name__ == "__main__":
    main()
