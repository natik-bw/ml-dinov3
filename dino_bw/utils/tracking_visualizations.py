#!/usr/bin/env python3
"""
Visualization utilities for tree tracking using DINOv3 features.

This module contains functions for:
- Tree object visualization
- Patch-level PCA color visualization  
- Correspondence visualization between frames
- Multi-frame unified visualizations
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from feature_extraction import TreeObject


def visualize_tree_objects(
        frame_data: Dict[str, Any],
        tree_objects: List[TreeObject],
        data_folder: Path,
        save_path: Optional[Path] = None,
        auto_close_fig: bool = True,
) -> None:
    """
    Visualize tree objects on the original image with concise labels.
    """
    # Load original image
    image_path = data_folder / "images" / frame_data['filename']
    if not image_path.exists():
        print(f"Warning: Image not found at {image_path}")
        return

    image = Image.open(image_path).convert('RGB')

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(f"Frame {frame_data['frame_id']}: {len(tree_objects)} Tree Objects")

    # Draw bounding boxes
    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    colors = [color_list[i % len(color_list)] for i in range(len(tree_objects))]

    for i, tree_obj in enumerate(tree_objects):
        bbox = tree_obj.original_bbox
        rect = patches.Rectangle(
            (bbox['x1'], bbox['y1']),
            bbox['x2'] - bbox['x1'],
            bbox['y2'] - bbox['y1'],
            linewidth=2,
            edgecolor=colors[i],
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add concise label in top-right corner of bounding box
        ax.text(
            bbox['x2'] - 5, bbox['y1'] + 15,  # Top-right corner with small offset
            f"{tree_obj.object_id}",
            color=colors[i],
            fontsize=8,  # Smaller font
            weight='normal',  # Not bold
            ha='right',  # Right-aligned to stay within bbox
            va='top',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor=colors[i])
        )

    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    if auto_close_fig:
        plt.close()
    else:
        plt.show()


def visualize_patch_colors(
        frame_data: Dict[str, Any],
        patch_color_data: Dict[str, np.ndarray],
        data_folder: Path,
        patch_size: int,
        alpha: float = 0.5,
        save_path: Optional[Path] = None,
        auto_close_fig: bool = True,
) -> None:
    """
    Visualize patch-level PCA colors overlaid on the original image.
    
    Args:
        frame_data: Frame data containing image filename
        patch_color_data: Output from apply_pca_coloring_to_patches
        data_folder: Path to data folder containing images
        patch_size: Size of each patch in pixels
        alpha: Transparency of color overlay (0=transparent, 1=opaque)
        save_path: Optional save path for the visualization
    """
    if not patch_color_data:
        print("No patch color data provided")
        return

    # Load original image
    image_path = data_folder / "images" / frame_data['filename']
    if not image_path.exists():
        print(f"Warning: Image not found at {image_path}")
        return

    original_image = Image.open(image_path).convert('RGB')
    target_image_size = frame_data['target_image_size']  # (width, height)

    # Resize original image to match target size
    resized_image = original_image.resize(target_image_size)

    # Get patch data
    patch_colors = patch_color_data['patch_colors']
    patch_mask = patch_color_data['patch_mask']
    h_patches, w_patches = patch_colors.shape[:2]

    # Create color overlay image
    overlay = np.zeros((target_image_size[1], target_image_size[0], 3))
    overlay_mask = np.zeros((target_image_size[1], target_image_size[0]), dtype=bool)

    # Fill overlay with patch colors
    for i in range(h_patches):
        for j in range(w_patches):
            if patch_mask[i, j]:  # Only show tree patches
                # Calculate pixel coordinates for this patch
                y_start = i * patch_size
                y_end = min((i + 1) * patch_size, target_image_size[1])
                x_start = j * patch_size
                x_end = min((j + 1) * patch_size, target_image_size[0])

                # Set color for this patch
                overlay[y_start:y_end, x_start:x_end] = patch_colors[i, j]
                overlay_mask[y_start:y_end, x_start:x_end] = True

    # Create matplotlib figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    ax1.imshow(resized_image)
    ax1.set_title(f"Original Image - Frame {frame_data['frame_id']}")
    ax1.axis('off')

    # PCA color overlay only
    ax2.imshow(overlay)
    ax2.set_title("PCA Colors (Tree Patches Only)")
    ax2.axis('off')

    # Combined overlay
    combined = np.array(resized_image, dtype=float) / 255.0
    # Apply overlay where tree patches exist
    combined[overlay_mask] = (1 - alpha) * combined[overlay_mask] + alpha * overlay[overlay_mask]

    ax3.imshow(combined)
    ax3.set_title(f"Combined (α={alpha})")
    ax3.axis('off')

    # Add patch grid visualization for reference
    for ax in [ax2, ax3]:
        # Draw patch grid lines (light gray, subtle)
        for i in range(0, target_image_size[1], patch_size):
            ax.axhline(y=i, color='lightgray', linewidth=0.5, alpha=0.3)
        for j in range(0, target_image_size[0], patch_size):
            ax.axvline(x=j, color='lightgray', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Patch color visualization saved to {save_path}")

    if auto_close_fig:
        plt.close()
    else:
        plt.show()

    # Print statistics
    tree_patches_count = np.sum(patch_mask)
    print(f"Visualized {tree_patches_count} tree patches")


def visualize_tree_correspondences(
        frame1_data: Dict[str, Any],
        frame2_data: Dict[str, Any],
        trees_frame1: List[TreeObject],
        trees_frame2: List[TreeObject],
        correspondences: List[Tuple[int, int, float]],
        data_folder: Path,
        save_path: Optional[Path] = None
) -> None:
    """
    Visualize tree correspondences between two frames with color-coded legend.
    
    Args:
        frame1_data: First frame data
        frame2_data: Second frame data
        trees_frame1: Tree objects from first frame
        trees_frame2: Tree objects from second frame
        correspondences: List of (tree1_idx, tree2_idx, similarity_score) tuples
        data_folder: Path to data folder containing images
        save_path: Optional save path for the visualization
    """
    # Load both images
    image1_path = data_folder / "images" / frame1_data['filename']
    image2_path = data_folder / "images" / frame2_data['filename']

    if not image1_path.exists() or not image2_path.exists():
        print(f"Warning: Images not found at {image1_path} or {image2_path}")
        return

    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')

    # Make images the same width for vertical stacking
    width = max(image1.width, image2.width)
    image1_resized = image1.resize((width, int(image1.height * width / image1.width)))
    image2_resized = image2.resize((width, int(image2.height * width / image2.width)))

    # Create vertically stacked image using PIL
    total_height = image1_resized.height + image2_resized.height
    combined_image = Image.new('RGB', (width, total_height))
    combined_image.paste(image1_resized, (0, 0))
    combined_image.paste(image2_resized, (0, image1_resized.height))

    # Create matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax.imshow(combined_image)
    ax.set_title(f"Tree Correspondences: Frame {frame1_data['frame_id']} → Frame {frame2_data['frame_id']}")

    # Color map for correspondences
    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # Calculate scaling factors for bounding box coordinates
    scale1_x = image1_resized.width / image1.width
    scale1_y = image1_resized.height / image1.height
    scale2_x = image2_resized.width / image2.width
    scale2_y = image2_resized.height / image2.height

    # Create correspondence color mapping
    correspondence_colors: Dict[Tuple[int, int], Tuple[str, float]] = {}
    for tree1_idx, tree2_idx, similarity in correspondences:
        color = color_list[len(correspondence_colors) % len(color_list)]
        correspondence_colors[(tree1_idx, tree2_idx)] = (color, similarity)

    # Draw tree objects with correspondence colors only
    for tree1_idx, tree2_idx, similarity in correspondences:
        color, _ = correspondence_colors[(tree1_idx, tree2_idx)]
        line_width = max(2, int(similarity * 6))  # Thicker lines for better visibility

        # Draw first frame tree
        tree1 = trees_frame1[tree1_idx]
        bbox1 = tree1.original_bbox
        scaled_bbox1 = {
            'x1': bbox1['x1'] * scale1_x,
            'y1': bbox1['y1'] * scale1_y,
            'x2': bbox1['x2'] * scale1_x,
            'y2': bbox1['y2'] * scale1_y
        }

        rect1 = patches.Rectangle(
            (scaled_bbox1['x1'], scaled_bbox1['y1']),
            scaled_bbox1['x2'] - scaled_bbox1['x1'],
            scaled_bbox1['y2'] - scaled_bbox1['y1'],
            linewidth=line_width,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect1)

        # Draw second frame tree
        tree2 = trees_frame2[tree2_idx]
        bbox2 = tree2.original_bbox
        scaled_bbox2 = {
            'x1': bbox2['x1'] * scale2_x,
            'y1': bbox2['y1'] * scale2_y + image1_resized.height,
            'x2': bbox2['x2'] * scale2_x,
            'y2': bbox2['y2'] * scale2_y + image1_resized.height
        }

        rect2 = patches.Rectangle(
            (scaled_bbox2['x1'], scaled_bbox2['y1']),
            scaled_bbox2['x2'] - scaled_bbox2['x1'],
            scaled_bbox2['y2'] - scaled_bbox2['y1'],
            linewidth=line_width,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect2)

    # Create legend in top-right corner
    legend_x = width - 200  # Position from right edge
    legend_y = 30  # Position from top
    legend_line_height = 25

    # Legend title
    ax.text(legend_x, legend_y, "Correspondences:", fontsize=12, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))

    # Legend entries
    for i, (tree1_idx, tree2_idx, similarity) in enumerate(correspondences):
        color, _ = correspondence_colors[(tree1_idx, tree2_idx)]
        y_pos = legend_y + (i + 1) * legend_line_height

        # Draw colored square indicator
        legend_rect = patches.Rectangle(
            (legend_x, y_pos - 8), 16, 16,
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.3
        )
        ax.add_patch(legend_rect)

        # Add text
        ax.text(legend_x + 25, y_pos, f"{tree1_idx} → {tree2_idx}: {similarity:.2f}",
                fontsize=10, va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Correspondence visualization saved to {save_path}")

    plt.show()


def visualize_united_pca_overlay(
        frames_data: List[Dict[str, Any]],
        patch_color_data_list: List[Dict[str, np.ndarray]],
        data_folder: Path,
        patch_size: int,
        alpha: float = 0.6,
        save_path: Optional[Path] = None
) -> None:
    """
    Create a united visualization showing PCA overlays from all frames.
    
    Args:
        frames_data: List of frame data dictionaries
        patch_color_data_list: List of patch color data for each frame
        data_folder: Path to data folder containing images
        patch_size: Size of each patch in pixels
        alpha: Transparency of color overlays
        save_path: Optional save path for the visualization
    """
    if not frames_data or not patch_color_data_list:
        print("No frame data or patch color data provided")
        return

    num_frames = len(frames_data)

    # Calculate grid layout for subplots
    cols = min(3, num_frames)  # Max 3 columns
    rows = (num_frames + cols - 1) // cols  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if num_frames == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()

    fig.suptitle("United PCA Color Visualization - All Frames", fontsize=16)

    for i, (frame_data, patch_color_data) in enumerate(zip(frames_data, patch_color_data_list)):
        if i >= len(axes):
            break

        ax = axes[i]

        # Load and resize image
        image_path = data_folder / "images" / frame_data['filename']
        if not image_path.exists():
            print(f"Warning: Image not found at {image_path}")
            continue

        original_image = Image.open(image_path).convert('RGB')
        target_image_size = frame_data['target_image_size']
        resized_image = original_image.resize(target_image_size)

        # Create overlay
        if patch_color_data:
            patch_colors = patch_color_data['patch_colors']
            patch_mask = patch_color_data['patch_mask']
            h_patches, w_patches = patch_colors.shape[:2]

            # Create color overlay
            overlay = np.zeros((target_image_size[1], target_image_size[0], 3))
            overlay_mask = np.zeros((target_image_size[1], target_image_size[0]), dtype=bool)

            # Fill overlay with patch colors
            for row in range(h_patches):
                for col in range(w_patches):
                    if patch_mask[row, col]:
                        y_start = row * patch_size
                        y_end = min((row + 1) * patch_size, target_image_size[1])
                        x_start = col * patch_size
                        x_end = min((col + 1) * patch_size, target_image_size[0])

                        overlay[y_start:y_end, x_start:x_end] = patch_colors[row, col]
                        overlay_mask[y_start:y_end, x_start:x_end] = True

            # Combine with original image
            combined = np.array(resized_image, dtype=float) / 255.0
            combined[overlay_mask] = (1 - alpha) * combined[overlay_mask] + alpha * overlay[overlay_mask]
            ax.imshow(combined)
        else:
            # Show original image if no patch data
            ax.imshow(resized_image)

        ax.set_title(f"Frame {frame_data['frame_id']}")
        ax.axis('off')

    # Hide unused subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"United PCA visualization saved to {save_path}")

    plt.show()

    # Print statistics
    total_patches = sum(np.sum(data.get('patch_mask', [])) for data in patch_color_data_list if data)
    print(f"United visualization: {num_frames} frames, {total_patches} total tree patches")
