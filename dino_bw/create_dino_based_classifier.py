#!/usr/bin/env python3
"""
Script to extract DINO embeddings from bw2508 dataset for foreground segmentation training.

Based on the foreground_segmentation.ipynb notebook but adapted for the bw2508 dataset.
"""

import pickle
import argparse
from pathlib import Path

import numpy as np

from dino_bw.bw_dino_defs import MODEL_TO_NUM_LAYERS
from dino_bw.dino_classifier_training import split_to_folds, calculate_fold_scores, accumulate_cv_statistics, \
    retrain_optimal_classifier
from dino_bw.dino_embeddings_utils import extract_raw_features, extract_all_features


def process_cached_features(cache_path: Path, class_name: str, output_dir: Path):
    """Load cached features and process them with cross-validation to find optimal classifier."""
    print("=== PROCESSING CACHED FEATURES WITH CROSS-VALIDATION ===")
    # Stage 1: Extract all features
    print("\n--- STAGE 1: EXTRACTING FEATURES ---")
    features, labels, image_indices, cache_metadata = extract_all_features(cache_path)

    # Stage 2: Split to six folds
    print("\n--- STAGE 2: SPLITTING TO FOLDS ---")
    folds = split_to_folds(image_indices, n_folds=6)

    # Stage 3: Calculate scores for each fold across all C values
    print("\n--- STAGE 3: CROSS-VALIDATION ---")
    c_values = np.logspace(-7, 0, 8)  # Same as in the notebook
    all_fold_scores = []
    all_detailed_results = []

    for fold_idx, (train_mask, val_mask) in enumerate(folds):
        fold_scores, detailed_results = calculate_fold_scores(
            features, labels, train_mask, val_mask, c_values, fold_idx
        )
        all_fold_scores.append(fold_scores)
        all_detailed_results.extend(detailed_results)

    # Stage 4: Accumulate statistics and find optimal C
    print("\n--- STAGE 4: FINDING OPTIMAL C ---")
    optimal_c, optimal_c_idx = accumulate_cv_statistics(all_fold_scores, c_values, output_dir)

    # Stage 5: Retrain optimal classifier and save
    print("\n--- STAGE 5: RETRAINING OPTIMAL CLASSIFIER ---")
    final_classifier = retrain_optimal_classifier(features, labels, optimal_c, class_name, output_dir)

    # Save processed results (existing functionality)
    print("\n--- SAVING PROCESSED FEATURES ---")
    features_path = output_dir / f"bw2508_dino_features_{class_name}.pkl"
    labels_path = output_dir / f"bw2508_dino_labels_{class_name}.pkl"
    indices_path = output_dir / f"bw2508_image_indices_{class_name}.pkl"

    print(f"Saving processed features to {features_path}")
    with open(features_path, "wb") as f:
        pickle.dump(features, f)

    print(f"Saving processed labels to {labels_path}")
    with open(labels_path, "wb") as f:
        pickle.dump(labels, f)

    print(f"Saving processed image indices to {indices_path}")
    with open(indices_path, "wb") as f:
        pickle.dump(image_indices, f)

    # Save dataset samples for reference
    sample_info = cache_metadata["sample_info"]
    samples_path = output_dir / "bw2508_samples.pkl"
    print(f"Saving samples metadata to {samples_path}")
    with open(samples_path, "wb") as f:
        pickle.dump(sample_info, f)

    print("\n=== PROCESSING COMPLETED SUCCESSFULLY! ===")
    print(f"Processed {len(sample_info)} samples")
    print(f"Extracted {len(features)} patch features")
    print(f"Optimal C found: {optimal_c:.2e}")
    print(f"Final classifier saved to: {output_dir / f'fg_classifier_{class_name}_c_{optimal_c}.pkl'}")
    print(f"Plots saved to: {output_dir / 'optimal_clf_debugging'}")

    return {
        'features': features,
        'labels': labels,
        'image_indices': image_indices,
        'optimal_c': optimal_c,
        'classifier': final_classifier,
        'cv_scores': all_fold_scores,
        'detailed_results': all_detailed_results
    }


def main(args):
    """Main function to extract and save DINO embeddings with caching."""
    print(
        f"Starting DINO embeddings extraction for {args.dataset_path.stem} dataset - class: {args.segmentation_class_name}")

    # Define cache path
    args.output_dir.mkdir(exist_ok=True, parents=True)
    cache_filename = f"raw_features_cache_{args.segmentation_class_name}_{args.model_name}_{args.dino_patch_size}_{args.image_size}.pkl"
    cache_path = args.output_dir / cache_filename

    # Check if cache exists
    if cache_path.exists():
        print(f"Found existing cache at {cache_path}")
        print("Skipping feature extraction, loading from cache...")
    else:
        print("No cache found, extracting raw features...")
        extract_raw_features(
            args.segmentation_class_name, args.dino_patch_size, args.image_size, args.dataset_path,
            args.dinov3_location, args.model_name,
            args.checkpoint_path, cache_path
        )

    # Process cached features
    process_cached_features(cache_path, args.segmentation_class_name, args.output_dir)

    print("DINO embeddings extraction completed successfully!")


if __name__ == "__main__":
    # Default constants (can be overridden via command line)
    DEFAULT_PATCH_SIZE = 16
    DEFAULT_IMAGE_SIZE = 768
    DEFAULT_DATASET_PATH = Path("/home/nati/source/datasets/bw2508")
    DEFAULT_DINOV3_LOCATION = Path("/home/nati/source/dinov3")
    DEFAULT_MODEL_NAME = "dinov3_vits16"
    DEFAULT_CHECKPOINT_PATH = Path("/home/nati/source/dinov3/checkpoints/dinov3_vits16_pretrain.pth")

    parser = argparse.ArgumentParser(description="Extract DINO embeddings for bw2508 dataset")
    parser.add_argument("--segmenation_class_name", default="tree", help="Class name to extract embeddings for")
    parser.add_argument("--dino_patch_size", type=int, default=DEFAULT_PATCH_SIZE, help="Patch size for DINOv3 model")
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE, help="Target image height in pixels")
    parser.add_argument("--dataset_path", type=Path, default=DEFAULT_DATASET_PATH, help="Path to bw2508 dataset")
    parser.add_argument("--dinov3_location", type=Path, default=DEFAULT_DINOV3_LOCATION,
                        help="Path to DINOv3 repository")
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        choices=list(MODEL_TO_NUM_LAYERS.keys()),
        help="DINOv3 model name",
    )
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT_PATH,
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=Path, default="/home/nati/source/dataOut/dinov3/dino_bw",
                        help="Path to model checkpoint")

    args = parser.parse_args()
    main(args)
