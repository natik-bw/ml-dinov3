#!/usr/bin/env python3
"""
Enhanced tracking processor for tree object correspondence using DINOv3 features.

This script implements multi-frame tree object tracking by:
1. Processing multiple frame indices from the tracking dataset
2. Extracting tree objects from each frame using bounding boxes and segmentation masks
3. Computing unified PCA across all frames for consistent coloring
4. Creating comprehensive visualizations for individual frames and correspondences
5. Finding correspondences between the first two frames using Hungarian matching
"""
import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from dino_bw.utils.dino_embeddings_utils import get_class_idx_from_name
from bw_ml_common.datasets.data_accessor_factory import create_dataset_accessor

from dino_bw.utils.feature_extraction_utils import extract_tree_objects_from_frame, compute_unified_pca, \
    apply_pca_coloring_to_patches
from dino_bw.utils.tracking_visualizations import visualize_tree_objects, visualize_patch_colors, \
    visualize_united_pca_overlay


def main(
        data_folder: Path,
        cache_path: Path,
        tree_class_id: int,
        debugging_folder: Optional[Path] = None,
        frame_indices: Optional[Union[List[int], np.ndarray]] = None,
        patch_size: int = 16,
        image_size: int = 1024
):
    """
    Multi-frame tree tracking processor with unified PCA visualization.
    
    Args:
        data_folder: Path to dataset folder
        cache_path: Path to cached DINO features and segmentation
        debugging_folder: Optional folder for saving debug visualizations
        frame_indices: List of frame indices to process (default: auto-select)
        patch_size: DINO patch size (should match cache)
        image_size: Target image size (should match cache)
    """
    print("=" * 80)
    print("MULTI-FRAME TREE TRACKING PROCESSOR")
    print("=" * 80)

    # Check cache validity and parameters
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    metadata = cache_data.get('metadata')
    if metadata is None:
        print("unable to fetch cache metadata")
        return

    cache_patch_size = metadata.get('patch_size', None)
    cache_image_size = metadata.get('image_size', None)

    if cache_patch_size != patch_size or cache_image_size != image_size:
        print(f"Cache parameters mismatch:")
        print(f"  Cache: patch_size={cache_patch_size}, image_size={cache_image_size}")
        print(f"  Requested: patch_size={patch_size}, image_size={image_size}")
        print(f"Please run tracking_data_processor.py with correct parameters first")
        return

    # Select frames to process
    if frame_indices is None:
        # Auto-select frames with valid data
        valid_frames = []
        for i, frame in enumerate(cache_data['frames']):
            has_detections = len(frame.get('detections', [])) > 0
            has_segmentation = frame.get('segmentation') is not None
            if has_detections and has_segmentation:
                valid_frames.append(i)

        if len(valid_frames) < 2:
            raise ValueError(f"Need at least 2 frames with valid data, found {len(valid_frames)}")

        # Select 3-4 frames spread across the dataset
        frame_indices = [
                            valid_frames[0],  # First valid frame
                            valid_frames[len(valid_frames) // 3],  # Early frame
                            valid_frames[2 * len(valid_frames) // 3],  # Mid frame
                            valid_frames[-1]  # Last valid frame
                        ][:4]  # Limit to 4 frames max

    print(f"Processing {len(frame_indices)} frames: {frame_indices}")

    # STAGE 1: Extract tree objects from each frame
    print("\n=== STAGE 1: EXTRACTING TREE OBJECTS FROM FRAMES ===")
    all_trees = []
    frames_data = []

    for i, frame_idx in enumerate(frame_indices):
        print(f"\nProcessing Frame {frame_idx} ({i + 1}/{len(frame_indices)})...")
        frame_data = cache_data['frames'][int(frame_idx)]

        # Extract tree objects
        trees = extract_tree_objects_from_frame(
            frame_data, tree_class_id, patch_size, min_tree_patches=3
        )

        all_trees.append(trees)
        frames_data.append(frame_data)

        # print(f"  Extracted {len(trees)} tree objects")
        # for j, tree in enumerate(trees):
        #     print(f"    Tree {j}: {len(tree.patch_indices)} patches, confidence: {tree.confidence:.3f}")

    # STAGE 2: Visualize individual tree objects
    print("\n=== STAGE 2: VISUALIZING INDIVIDUAL TREE OBJECTS ===")
    for i, (frame_data, trees) in enumerate(zip(frames_data, all_trees)):
        frame_idx = frame_indices[i]
        save_path = debugging_folder / f"tree_objects_frame_{int(frame_idx)}.png" if debugging_folder else None
        visualize_tree_objects(frame_data, trees, data_folder, save_path=save_path)

    # STAGE 3: Compute unified PCA across all frames
    print("\n=== STAGE 3: COMPUTING UNIFIED PCA ===")
    try:
        unified_pca = compute_unified_pca(*all_trees)
        print("✓ Unified PCA computed successfully")
    except ValueError as e:
        print(f"Warning: Could not compute unified PCA: {e}")
        unified_pca = None

    # STAGE 4: Create patch-level PCA visualizations for each frame
    if unified_pca is not None:
        print("\n=== STAGE 4: CREATING PATCH-LEVEL PCA VISUALIZATIONS ===")
        patch_color_data_list = []

        for i, (frame_data, trees) in enumerate(zip(frames_data, all_trees)):
            frame_idx = frame_indices[i]
            print(f"Creating patch colors for frame {frame_idx}...")

            # Apply PCA coloring to patches
            patch_color_data = apply_pca_coloring_to_patches(
                trees, unified_pca, patch_size, frame_data['target_image_size']
            )
            patch_color_data_list.append(patch_color_data)

            # Visualize individual frame
            save_path = debugging_folder / f"patch_colors_frame_{frame_idx}.png" if debugging_folder else None
            visualize_patch_colors(frame_data, patch_color_data, data_folder, patch_size,
                                   alpha=0.4, save_path=save_path)

        # STAGE 5: Create united PCA visualization
        print("\n=== STAGE 5: CREATING UNITED PCA VISUALIZATION ===")
        save_path = debugging_folder / "united_pca_visualization.png" if debugging_folder else None
        visualize_united_pca_overlay(frames_data, patch_color_data_list, data_folder,
                                     patch_size, alpha=0.4, save_path=save_path)

    # STAGE 6: Compute correspondences between first two frames
    # if len(all_trees) >= 2:
    #     print("\n=== STAGE 6: COMPUTING CORRESPONDENCES (FIRST TWO FRAMES) ===")
    #     trees_frame1, trees_frame2 = all_trees[0], all_trees[1]
    #     frame1_data, frame2_data = frames_data[0], frames_data[1]
    #
    #     correspondences = compute_tree_correspondences(trees_frame1, trees_frame2, similarity_threshold=0.6)
    #
    #     print(f"Found {len(correspondences)} correspondences:")
    #     for tree1_idx, tree2_idx, similarity in correspondences:
    #         print(f"  Tree {tree1_idx} → Tree {tree2_idx}: {similarity:.3f}")
    #
    #     # Visualize correspondences
    #     save_path = debugging_folder / "tree_correspondences.png" if debugging_folder else None
    #     visualize_tree_correspondences(frame1_data, frame2_data, trees_frame1, trees_frame2,
    #                                    correspondences, data_folder, save_path=save_path)

    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Processed {len(frame_indices)} frames: {frame_indices}")
    print(f"Total tree objects: {sum(len(trees) for trees in all_trees)}")
    if unified_pca:
        print(f"PCA explained variance: {unified_pca.explained_variance_ratio_.sum():.3f}")
    # if len(all_trees) >= 2:
    #     print(f"Correspondences found: {len(correspondences) if 'correspondences' in locals() else 0}")

    if debugging_folder:
        print(f"\nVisualizations saved to: {debugging_folder}")
        saved_files = list(debugging_folder.glob("*.png"))
        for file in saved_files:
            print(f"  - {file.name}")


if __name__ == "__main__":
    # Example usage - update these paths for your setup
    data_folder = Path("/home/nati/source/data/greeting_dev/bags_parsed/turn8_0")
    cache_path = Path("/home/nati/source/data/greeting_dev/bags_parsed/turn8_0/tracking_cache_img_1024_patch_16.pkl")
    debugging_folder = Path("/home/nati/source/data/greeting_dev/bags_parsed/turn8_0/tracking_debug")

    datasets_folder = "/home/nati/source/datasets"
    data_accessor = create_dataset_accessor("bw2508", datasets_folder)
    tree_class_id = get_class_idx_from_name(data_accessor, "tree")

    # Create debugging folder if it doesn't exist
    debugging_folder.mkdir(exist_ok=True)

    # Run with custom frame indices
    frame_indices = np.linspace(200, 249, 9).astype('uint8')

    main(
        data_folder=data_folder,
        cache_path=cache_path,
        tree_class_id=tree_class_id,
        debugging_folder=debugging_folder,
        frame_indices=frame_indices,
        patch_size=16,
        image_size=1024
    )
