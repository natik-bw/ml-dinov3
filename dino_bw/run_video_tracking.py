#!/usr/bin/env python3
"""
Video-based segmentation tracking using DINOv3 features.

This script implements temporal segmentation tracking by:
1. Loading frames from the tracking cache (similar to run_tracking_processor.py)
2. Setting up the first frame as reference segmentation
3. Tracking segmentation masks through subsequent frames using DINOv3 features
4. Creating visualizations of the tracking results

Based on the segmentation tracking approach from segmentation_tracking.ipynb
"""

import warnings
import functools
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
import matplotlib.pyplot as plt

from tracking_data_processor import load_tracking_cache
from dino_bw.dino_embeddings_utils import get_class_idx_from_name, load_dinov3_model
from bw_ml_common.datasets.data_accessor_factory import create_dataset_accessor


class VideoFrameData:
    """Container for video frame data extracted from cache."""

    def __init__(self, frame_data: Dict[str, Any], frame_idx: int):
        self.frame_idx = frame_idx
        # self.frame_data = frame_data
        self.filename = frame_data['filename']
        self.frame_number = frame_data['frame_number']
        self.segmentation = frame_data.get('segmentation')  # [H_patches, W_patches] or None

        # Reshape features to [H, W, D] format
        if self.segmentation is not None:
            seg_array = np.array(self.segmentation)
            h_patches, w_patches = seg_array.shape
            self.features = torch.from_numpy(np.reshape(frame_data['features'], (h_patches, w_patches, -1))).cuda()
        else:
            # Fallback: assume square patch grid or use original feature shape
            features_flat = frame_data['features']
            total_patches = len(features_flat)
            patch_dim = int(np.sqrt(total_patches))  # Assume square grid
            feature_dim = features_flat.shape[-1] if len(features_flat.shape) > 1 else features_flat.shape[
                                                                                           0] // total_patches
            self.features = torch.from_numpy(np.reshape(features_flat, (patch_dim, patch_dim, feature_dim))).cuda()
        self.original_image_size = frame_data['original_image_size']  # (width, height)
        self.target_image_size = frame_data['target_image_size']  # (width, height)
        self.detections = frame_data.get('detections', [])
        self.detections_scaled = frame_data.get('detections_scaled', [])

    @property
    def has_segmentation(self) -> bool:
        return self.segmentation is not None

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0

    @property
    def feature_height(self) -> int:
        return self.features.shape[0]

    @property
    def feature_width(self) -> int:
        return self.features.shape[1]

    @property
    def feature_dim(self) -> int:
        return self.features.shape[2]


# ============================================================================
# TREE INSTANCE TRACKING DATA STRUCTURE
# ============================================================================

class TreesTracker:
    """
    Manages tree instance allocation and semantic-to-instance conversion.
    
    Handles:
    - Assignment of unique tree instance IDs
    - Conversion from semantic segmentation to instance segmentation
    - Tracking tree instances across frames using detection boxes
    """

    def __init__(self, tree_class_id: int = 2, background_class_id: int = 0):
        self.tree_class_id = tree_class_id
        self.background_class_id = background_class_id  # Use 0 for background
        self.next_instance_id = 1  # Start from 1, background is 0
        self.active_instances: Dict[int, Dict[str, Any]] = {}  # instance_id -> metadata
        self.frame_count = 0

    def get_next_instance_id(self) -> int:
        """Get the next available instance ID."""
        instance_id = self.next_instance_id
        self.next_instance_id += 1
        return instance_id

    def create_initial_instances(
            self,
            segmentation_mask: torch.Tensor,
            detections: List[Dict],
            patch_size: int,
            target_image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        Create initial tree instances from semantic segmentation and detections.
        
        Args:
            segmentation_mask: [H_patches, W_patches] semantic segmentation
            detections: List of detection boxes
            patch_size: Size of patches
            target_image_size: (width, height) of target image
            
        Returns:
            Tuple of (instance_mask, class_mapping)
        """
        print(f"=== CREATING INITIAL TREE INSTANCES (Frame {self.frame_count}) ===")

        h_patches, w_patches = segmentation_mask.shape

        # Initialize instance mask with all background (0)
        instance_mask = torch.full_like(segmentation_mask, self.background_class_id)

        # Find tree patches
        tree_patches = (segmentation_mask == self.tree_class_id)
        print(f"Found {tree_patches.sum().item()} tree patches")

        # Debug: Print original segmentation unique classes
        orig_classes = segmentation_mask.unique()
        print(f"Original segmentation classes: {orig_classes.cpu().numpy()}")

        # Process detections to create instances
        instance_count = 0
        for detection in detections:
            if detection.get('name') == 'trunk':  # Filter for tree detections
                box = detection['box']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

                # Convert to patch coordinates
                patch_x1 = max(0, min(int(x1 / patch_size), w_patches - 1))
                patch_y1 = max(0, min(int(y1 / patch_size), h_patches - 1))
                patch_x2 = max(0, min(int(x2 / patch_size), w_patches - 1))
                patch_y2 = max(0, min(int(y2 / patch_size), h_patches - 1))

                # Find tree patches in this box
                box_tree_patches = tree_patches[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1]

                if box_tree_patches.sum() > 0:
                    instance_id = self.get_next_instance_id()
                    instance_mask[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1][box_tree_patches] = instance_id

                    # Store instance metadata
                    self.active_instances[instance_id] = {
                        'bbox': (patch_x1, patch_y1, patch_x2, patch_y2),
                        'patch_count': box_tree_patches.sum().item(),
                        'first_seen': self.frame_count
                    }

                    print(
                        f"  Created tree instance {instance_id}: bbox=({patch_x1},{patch_y1})->({patch_x2},{patch_y2}), "
                        f"{box_tree_patches.sum().item()} patches")
                    instance_count += 1

        # Tree patches not assigned to any instance remain as background (already 0)

        # Since we start with background=0 and instances=1,2,3..., classes should be contiguous
        unique_classes = instance_mask.unique()

        # Create identity mapping since classes should already be contiguous
        class_mapping = {k.item(): k.item() for k in unique_classes}

        print(f"Created {instance_count} tree instances")
        print(f"Active instances: {list(self.active_instances.keys())}")
        print(f"Final unique classes: {unique_classes.cpu().numpy()}")
        print(f"Class mapping: {class_mapping}")

        # Debug: Check if classes are contiguous starting from 0
        sorted_classes = sorted(unique_classes.cpu().numpy())
        expected_classes = list(range(len(sorted_classes)))
        print(f"Checking contiguity: expected {expected_classes}, got {sorted_classes}")

        if sorted_classes != expected_classes:
            print(f"ERROR: Classes are not contiguous! This will cause F.one_hot to fail.")
            print(f"Expected: {expected_classes}")
            print(f"Got: {sorted_classes}")

            # Create proper contiguous mapping 
            class_mapping = {k.item(): i for i, k in enumerate(unique_classes)}

            # Apply remapping to ensure contiguous classes 0, 1, 2, ...
            remapped_mask = torch.zeros_like(instance_mask)
            for orig_id, new_id in class_mapping.items():
                remapped_mask[instance_mask == orig_id] = new_id

            # Update active_instances keys to use remapped IDs
            new_active_instances = {}
            for orig_id, metadata in self.active_instances.items():
                if orig_id in class_mapping:
                    new_id = class_mapping[orig_id]
                    new_active_instances[new_id] = metadata
                    print(f"  Remapped active instance {orig_id} -> {new_id}")
            self.active_instances = new_active_instances

            print(f"Applied remapping: {class_mapping}")
            print(f"Updated active instances: {list(self.active_instances.keys())}")
            return remapped_mask, class_mapping

        self.frame_count += 1
        return instance_mask, class_mapping

    def assign_instances_from_predictions(
            self,
            semantic_mask: torch.Tensor,
            predicted_probs: torch.Tensor,
            detections: List[Dict],
            patch_size: int,
            target_image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        Convert semantic segmentation to instance segmentation using predicted probabilities.
        
        Process:
        1. For each detection box, find tree patches
        2. Check most common predicted instance in those patches
        3. If background -> create new instance
        4. If existing instance -> assign that instance
        
        Args:
            semantic_mask: [H_patches, W_patches] semantic segmentation 
            predicted_probs: [H_patches, W_patches, M] predicted probabilities
            detections: List of detection boxes
            patch_size: Size of patches
            target_image_size: (width, height) of target image
            
        Returns:
            Tuple of (instance_mask, class_mapping)
        """
        print(f"=== ASSIGNING TREE INSTANCES (Frame {self.frame_count}) ===")

        h_patches, w_patches = semantic_mask.shape
        instance_mask = torch.full_like(semantic_mask, self.background_class_id)

        # Find tree patches in semantic segmentation
        tree_patches = (semantic_mask == self.tree_class_id)
        print(f"Found {tree_patches.sum().item()} tree patches in semantic mask")

        # Get predicted instance labels (argmax of probabilities)
        predicted_instances = predicted_probs.argmax(dim=-1)  # [H_patches, W_patches]

        # Process each detection box
        assigned_count = 0
        new_instances = []

        for detection in detections:
            if detection.get('name') == 'trunk':  # Filter for tree detections
                box = detection['box']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

                # Convert to patch coordinates
                patch_x1 = max(0, min(int(x1 / patch_size), w_patches - 1))
                patch_y1 = max(0, min(int(y1 / patch_size), h_patches - 1))
                patch_x2 = max(0, min(int(x2 / patch_size), w_patches - 1))
                patch_y2 = max(0, min(int(y2 / patch_size), h_patches - 1))

                # Find tree patches in this box
                box_tree_patches = tree_patches[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1]

                if box_tree_patches.sum() > 0:
                    # Get predicted instances for tree patches in this box
                    box_predictions = predicted_instances[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1]
                    tree_predictions = box_predictions[box_tree_patches]

                    # Find unique predictions and their counts
                    unique_preds, counts = torch.unique(tree_predictions, return_counts=True)

                    # Filter out background predictions
                    non_bg_mask = unique_preds != self.background_class_id
                    non_bg_preds = unique_preds[non_bg_mask]
                    non_bg_counts = counts[non_bg_mask]

                    total_non_bg_votes = non_bg_counts.sum().item()

                    print(
                        f"  Voting analysis: {total_non_bg_votes} non-background votes out of {len(tree_predictions)} tree patches")
                    for pred, count in zip(non_bg_preds, non_bg_counts):
                        percentage = (count.item() / total_non_bg_votes) * 100 if total_non_bg_votes > 0 else 0
                        print(f"    Class {pred.item()}: {count.item()} votes ({percentage:.1f}%)")

                    # Check if any non-background prediction has >50% of non-background votes
                    assigned_instance = None
                    if total_non_bg_votes > 0:
                        for pred, count in zip(non_bg_preds, non_bg_counts):
                            percentage = count.item() / total_non_bg_votes
                            if percentage > 0.5 and pred.item() in self.active_instances:
                                assigned_instance = pred.item()
                                print(
                                    f"  Assigning to existing instance {assigned_instance} ({percentage:.1%} of non-bg votes)")
                                break

                    if assigned_instance is None:
                        # Create new instance (no strong majority or non-existent instance)
                        assigned_instance = self.get_next_instance_id()
                        self.active_instances[assigned_instance] = {
                            'bbox': (patch_x1, patch_y1, patch_x2, patch_y2),
                            'patch_count': box_tree_patches.sum().item(),
                            'first_seen': self.frame_count
                        }
                        new_instances.append(assigned_instance)
                        if total_non_bg_votes == 0:
                            print(f"  Created new tree instance {assigned_instance} (no non-background votes)")
                        else:
                            print(
                                f"  Created new tree instance {assigned_instance} (no class >50% or not in active instances)")

                    # Assign instance to tree patches in this box
                    instance_mask[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1][box_tree_patches] = assigned_instance
                    assigned_count += 1

        # No need for complex remapping - instance IDs are already clean
        # Just create a simple identity mapping for compatibility with existing code
        unique_classes = instance_mask.unique()
        class_mapping = {k.item(): k.item() for k in unique_classes}

        print(f"Assigned {assigned_count} trees to instances")
        print(f"New instances created: {new_instances}")
        print(f"Active instances: {list(self.active_instances.keys())}")
        print(f"Final unique classes: {unique_classes.cpu().numpy()}")

        self.frame_count += 1
        return instance_mask, class_mapping


# ============================================================================
# CORE TRACKING FUNCTIONS (adapted from segmentation_tracking.ipynb)
# ============================================================================

@torch.compile(disable=True)
def propagate(
        current_features: Tensor,  # [h", w", D], where h=h", w=w", and " stands for current
        context_features: Tensor,  # [t, h, w, D]
        context_probs: Tensor,  # [t, h, w, M]
        neighborhood_mask: Tensor,  # [h", w", h, w]
        topk: int,
        temperature: float,
) -> Tensor:
    """
    Core label propagation function adapted from segmentation_tracking.ipynb.
    
    For each patch of the current frame:
    - Compute cosine similarity with all context patches
    - Restrict focus to local neighborhood and select top-k most similar patches
    - Compute weighted average of mask probabilities to get prediction
    """
    t, h, w, M = context_probs.shape

    # Compute similarity current -> context
    dot = torch.einsum(
        "ijd, tuvd -> ijtuv",
        current_features,  # [h", w", D]
        context_features,  # [t, h, w, D]
    )  # [h", w", t, h, w]

    # Restrict focus to local neighborhood
    dot = torch.where(
        neighborhood_mask[:, :, None, :, :],  # [h", w", 1, h, w]
        dot,  # [h", w", t, h, w]
        -torch.inf,
    )

    # Select top-k patches inside the neighborhood
    dot = dot.flatten(2, -1).flatten(0, 1)  # [h"w", thw]
    k_th_largest = torch.topk(dot, dim=1, k=topk).values  # [h"w", k]
    dot = torch.where(
        dot >= k_th_largest[:, -1:],  # [h"w", thw]
        dot,  # [h"w", thw]
        -torch.inf,
    )

    # Propagate probabilities from context to current frame
    weights = F.softmax(dot / temperature, dim=1)  # [h"w", thw]
    current_probs = torch.mm(
        weights,  # [h"w", thw]
        context_probs.flatten(0, 2),  # [thw, M]
    )  # [h"w", M]

    # Propagated probs should already sum to 1, but just in case
    current_probs = current_probs / current_probs.sum(dim=1, keepdim=True)  # [h"w", M]

    return current_probs.unflatten(0, (h, w))  # [h", w", M]


@functools.lru_cache()
def make_neighborhood_mask(h: int, w: int, size: float, shape: str) -> Tensor:
    """
    Create neighborhood mask for spatial locality constraints.
    Adapted from segmentation_tracking.ipynb.
    """
    ij = torch.stack(
        torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device="cuda"),
            torch.arange(w, dtype=torch.float32, device="cuda"),
            indexing="ij",
        ),
        dim=-1,
    )  # [h, w, 2]
    if shape == "circle":
        norm = torch.linalg.vector_norm(
            ij[:, :, None, None, :] - ij[None, None, :, :, :],  # [h", w", h, w, 2]
            ord=2,
            dim=-1,
        )  # [h", w", h, w]
    elif shape == "square":
        norm = torch.linalg.vector_norm(
            ij[:, :, None, None, :] - ij[None, None, :, :, :],  # [h", w", h, w, 2]
            ord=torch.inf,
            dim=-1,
        )  # [h", w", h, w]
    else:
        raise ValueError(f"Invalid {shape=}")
    mask = norm <= size  # [h", w", h, w] bool, True inside, False outside
    return mask


def postprocess_probs(
        probs: Tensor,  # [B, M, H', W']
) -> Tensor:
    """
    Normalize and clean probability distributions.
    Adapted from segmentation_tracking.ipynb.
    """
    vmin = probs.flatten(2, 3).min(dim=2).values  # [B, M]
    vmax = probs.flatten(2, 3).max(dim=2).values  # [B, M]
    probs = (probs - vmin[:, :, None, None]) / (vmax[:, :, None, None] - vmin[:, :, None, None])
    probs = torch.nan_to_num(probs, nan=0)
    return probs  # [B, M, H', W']


def track_first_to_second_frame(
        first_frame: VideoFrameData,
        second_frame: VideoFrameData,
        reference_probs: Tensor,  # [h, w, M]
        num_classes: int,
        neighborhood_size: float = 12.0,
        neighborhood_shape: str = "circle",
        topk: int = 5,
        temperature: float = 0.2
) -> Tuple[Tensor, np.ndarray, np.ndarray]:
    """
    Track tree instances from first frame to second frame.
    
    Args:
        first_frame: Reference frame with tree instances
        second_frame: Target frame to track to
        reference_probs: Reference probabilities [h, w, M]
        num_classes: Number of classes/instances
        neighborhood_size: Size of spatial neighborhood
        neighborhood_shape: Shape of neighborhood ("circle" or "square") 
        topk: Number of top similar patches to consider
        temperature: Temperature for softmax weighting
        
    Returns:
        Tuple of (current_probs, current_pred_patches, current_probs_np)
        - current_probs: Predicted probabilities [h, w, M] (patch resolution)
        - current_pred_patches: Predicted mask as numpy array [h, w] (patch resolution)
        - current_probs_np: Probabilities as numpy array [M, H', W'] (target resolution)
    """
    print("=== TRACKING FIRST TO SECOND FRAME ===")

    # Get features (already normalized)
    first_feats = first_frame.features  # [h, w, D]
    second_feats = second_frame.features  # [h, w, D]

    print(f"First frame features: {first_feats.shape}")
    print(f"Second frame features: {second_feats.shape}")
    print(f"Reference probs: {reference_probs.shape}")

    # Create neighborhood mask
    h, w = first_feats.shape[:2]
    neighborhood_mask = make_neighborhood_mask(h, w, neighborhood_size, neighborhood_shape)
    print(f"Neighborhood mask: {neighborhood_mask.shape}")

    # Ensure features have proper shape and mark as dynamic for torch.compile
    torch._dynamo.maybe_mark_dynamic(first_feats, (0, 1))
    torch._dynamo.maybe_mark_dynamic(second_feats, (0, 1))

    # Run propagation
    current_probs = propagate(
        second_feats,  # [h", w", D] - current frame
        context_features=first_feats.unsqueeze(0),  # [1, h, w, D] - context
        context_probs=reference_probs.unsqueeze(0),  # [1, h, w, M] - context probs
        neighborhood_mask=neighborhood_mask,  # [h", w", h, w]
        topk=topk,
        temperature=temperature,
    )  # [h", w", M]

    print(f"Propagated probs: {current_probs.shape}")

    # Upsample to original image resolution for visualization
    # Convert from [h, w, M] to [1, M, h, w] for interpolation
    p = current_probs.movedim(-1, -3).unsqueeze(0)  # [1, M, h, w]

    # Get target size from second frame
    target_h, target_w = second_frame.target_image_size[1], second_frame.target_image_size[0]  # (height, width)
    p = F.interpolate(p, size=(target_h, target_w), mode="nearest")  # [1, M, H', W']

    # Postprocess probabilities
    p = postprocess_probs(p).squeeze(0)  # [M, H', W']

    # Get predictions at both resolutions
    # Patch resolution predictions for consistent mask handling
    current_pred_patches = current_probs.argmax(-1).cpu().numpy()  # [h, w]

    # Target resolution predictions and probabilities for detailed visualization
    current_pred_np = p.argmax(0).cpu().numpy()  # [H', W']
    current_probs_np = p.cpu().numpy()  # [M, H', W']

    print(f"Patch resolution prediction shape: {current_pred_patches.shape}")
    print(f"Target resolution prediction shape: {current_pred_np.shape}")
    print(f"Final probabilities shape: {current_probs_np.shape}")

    return current_probs, current_pred_patches, current_probs_np


def process_consecutive_frames(
        video_frames: List[VideoFrameData],
        tree_class_id: int,
        patch_size: int = 16,
        max_context_length: int = 4,
        neighborhood_size: float = 12.0,
        neighborhood_shape: str = "circle",
        topk: int = 5,
        temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Process consecutive frames with memory-based tree instance tracking.
    
    Args:
        video_frames: List of video frames to process
        tree_class_id: Class ID for trees 
        patch_size: DINO patch size
        max_context_length: Maximum number of context frames to keep
        neighborhood_size: Spatial neighborhood size for propagation
        neighborhood_shape: Shape of neighborhood ("circle" or "square")
        topk: Number of top similar patches
        temperature: Temperature for softmax
        
    Returns:
        Dictionary containing tracking results for all frames
    """
    print("=== PROCESSING CONSECUTIVE FRAMES ===")
    print(f"Total frames: {len(video_frames)}")

    # Initialize tracker
    tracker = TreesTracker(tree_class_id=tree_class_id)

    # Storage for results
    results: Dict[str, Any] = {
        'frames': [],
        'tracker': tracker,
        'metadata': {
            'max_context_length': max_context_length,
            'neighborhood_size': neighborhood_size,
            'topk': topk,
            'temperature': temperature
        }
    }

    # Context queues (similar to segmentation_tracking.ipynb)
    features_queue: List[Tensor] = []
    probs_queue: List[Tensor] = []

    # Process first frame (reference)
    first_frame = video_frames[0]
    if not first_frame.has_segmentation or not first_frame.has_detections:
        raise ValueError("First frame must have both segmentation and detections")

    print(f"\n--- PROCESSING FRAME 0 (REFERENCE) ---")

    # Create initial instances using TreesTracker
    first_seg_mask = torch.from_numpy(first_frame.segmentation).long().cuda()
    first_instance_mask, first_class_mapping = tracker.create_initial_instances(
        segmentation_mask=first_seg_mask,
        detections=first_frame.detections_scaled,
        patch_size=patch_size,
        target_image_size=first_frame.target_image_size
    )

    # Convert to one-hot probabilities (mask is already remapped)
    num_classes = len(first_class_mapping)
    first_probs = F.one_hot(first_instance_mask, num_classes).float()

    # Store first frame results
    frame_result = {
        'frame_idx': 0,
        'instance_mask': first_instance_mask.cpu().numpy(),
        'class_mapping': first_class_mapping,
        'probs': first_probs.cpu().numpy(),
        'num_classes': num_classes
    }
    results['frames'].append(frame_result)

    # Set up neighborhood mask
    h, w = first_frame.features.shape[:2]
    neighborhood_mask = make_neighborhood_mask(h, w, neighborhood_size, neighborhood_shape)

    # Process remaining frames
    for frame_idx in range(1, len(video_frames)):
        current_frame = video_frames[frame_idx]
        print(f"\n--- PROCESSING FRAME {frame_idx} ---")

        # Prepare context
        if frame_idx == 1:
            # First tracking frame: use only reference frame as context
            context_features = first_frame.features.unsqueeze(0)  # [1, h, w, D]
            context_probs = first_probs.unsqueeze(0)  # [1, h, w, M]
        else:
            # Use reference + recent frames as context
            context_features = torch.stack([first_frame.features] + features_queue, dim=0)
            context_probs = torch.stack([first_probs] + probs_queue, dim=0)

        print(f"Context shape: {context_features.shape}")

        # Mark features as dynamic for torch.compile
        torch._dynamo.maybe_mark_dynamic(current_frame.features, (0, 1))
        torch._dynamo.maybe_mark_dynamic(context_features, 0)
        torch._dynamo.maybe_mark_dynamic(context_probs, (0, 3))

        # Run propagation to get predicted probabilities
        predicted_probs = propagate(
            current_frame.features,  # [h, w, D]
            context_features,  # [t, h, w, D]
            context_probs,  # [t, h, w, M]
            neighborhood_mask,  # [h, w, h, w]
            topk=topk,
            temperature=temperature
        )  # [h, w, M]

        print(f"Predicted probs shape: {predicted_probs.shape}")

        # Convert semantic segmentation to instance segmentation using tracker
        if current_frame.has_segmentation and current_frame.has_detections:
            current_seg_mask = torch.from_numpy(current_frame.segmentation).long().cuda()

            current_instance_mask, current_class_mapping = tracker.assign_instances_from_predictions(
                semantic_mask=current_seg_mask,
                predicted_probs=predicted_probs,
                detections=current_frame.detections_scaled,
                patch_size=patch_size,
                target_image_size=current_frame.target_image_size
            )

            # Update class mapping and probabilities for new instances
            current_num_classes = len(current_class_mapping)

            # Convert to one-hot probabilities (mask is already remapped)
            current_probs = F.one_hot(current_instance_mask, current_num_classes).float()
        else:
            # No segmentation/detections: use predicted probabilities directly
            print("Warning: No segmentation/detections found, using predictions directly")
            current_instance_mask = predicted_probs.argmax(dim=-1)
            current_class_mapping = {i: i for i in range(predicted_probs.shape[-1])}
            current_num_classes = predicted_probs.shape[-1]
            current_probs = predicted_probs

        # Store frame results
        frame_result = {
            'frame_idx': frame_idx,
            'instance_mask': current_instance_mask.cpu().numpy(),
            'class_mapping': current_class_mapping,
            'probs': current_probs.cpu().numpy(),
            'num_classes': current_num_classes,
            'predicted_probs': predicted_probs.cpu().numpy()
        }
        results['frames'].append(frame_result)

        # Update context queues
        features_queue.append(current_frame.features)
        probs_queue.append(current_probs)

        # Maintain queue size
        if len(features_queue) > max_context_length:
            features_queue.pop(0)
        if len(probs_queue) > max_context_length:
            probs_queue.pop(0)

    print(f"\n✓ Processed {len(video_frames)} frames successfully")
    print(f"Final active instances: {list(tracker.active_instances.keys())}")

    return results


def visualize_tracking_results(
        first_frame: VideoFrameData,
        second_frame: VideoFrameData,
        reference_mask: np.ndarray,
        predicted_mask: np.ndarray,
        predicted_probs: np.ndarray,
        num_classes: int,
        data_folder: Path,
        save_path: Optional[Path] = None,
        ignore_class: Optional[int] = None,
        overlay_alpha: float = 0.6,
        patch_size: int = 16
):
    """
    Visualize tracking results similar to segmentation_tracking.ipynb.
    
    Shows:
    1. First frame with reference segmentation
    2. Second frame with predicted segmentation
    3. Overlay visualization with alpha blending for non-ignore classes
    4. Individual probability maps for each tree instance
    
    Args:
        ignore_class: Class to ignore in overlay (if None, uses max class index)
        overlay_alpha: Alpha value for overlay blending
    """
    print("=== VISUALIZING TRACKING RESULTS ===")

    # Load original images
    first_image = load_image_for_visualization(first_frame, data_folder)
    second_image = load_image_for_visualization(second_frame, data_folder)

    # Create RGB masks
    reference_rgb = mask_to_rgb(reference_mask, num_classes)
    predicted_rgb = mask_to_rgb(predicted_mask, num_classes)

    # Determine ignore class (background/invalid class)
    if ignore_class is None:
        ignore_class = num_classes - 1  # Use maximum class index as ignore class

    # Create overlay visualizations
    def create_overlay(image, mask, alpha=overlay_alpha, patch_size=16):
        """Create overlay where non-ignore classes are highlighted with alpha blending."""
        image_np = np.array(image)
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)

        # Upsample mask from patch resolution to pixel resolution
        # mask is [H_patches, W_patches], need to make it [H_pixels, W_pixels]
        mask_upsampled = np.repeat(np.repeat(mask, patch_size, axis=0), patch_size, axis=1)

        # Ensure mask matches image dimensions exactly
        img_h, img_w = image_np.shape[:2]
        mask_h, mask_w = mask_upsampled.shape

        if mask_h != img_h or mask_w != img_w:
            # Crop or pad mask to match image size exactly
            min_h, min_w = min(mask_h, img_h), min(mask_w, img_w)
            mask_final = np.full((img_h, img_w), ignore_class, dtype=mask_upsampled.dtype)
            mask_final[:min_h, :min_w] = mask_upsampled[:min_h, :min_w]
        else:
            mask_final = mask_upsampled

        # Create mask for non-ignore areas
        non_ignore_mask = mask_final != ignore_class

        # Create colored overlay from upsampled mask
        overlay = mask_to_rgb(mask_final, num_classes)

        # Apply alpha blending only where mask is not ignore class
        result = image_np.copy().astype(float)
        overlay_float = overlay.astype(float)

        # Blend only non-ignore areas
        for c in range(3):
            result[non_ignore_mask, c] = (
                    (1 - alpha) * image_np[non_ignore_mask, c] +
                    alpha * overlay_float[non_ignore_mask, c]
            )

        return result.astype(np.uint8)

    reference_overlay = create_overlay(first_image, reference_mask, patch_size=patch_size)
    predicted_overlay = create_overlay(second_image, predicted_mask, patch_size=patch_size)

    # Main comparison plot - now 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: Original images + overlays
    axes[0, 0].imshow(first_image)
    axes[0, 0].set_title(f"Reference Frame {first_frame.frame_idx}")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(reference_overlay)
    axes[0, 1].set_title(f"Reference Overlay (α={overlay_alpha})")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(reference_rgb)
    axes[0, 2].set_title("Reference Tree Instances")
    axes[0, 2].axis('off')

    # Bottom row: Target frame + results
    axes[1, 0].imshow(second_image)
    axes[1, 0].set_title(f"Target Frame {second_frame.frame_idx}")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(predicted_overlay)
    axes[1, 1].set_title(f"Predicted Overlay (α={overlay_alpha})")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(predicted_rgb)
    axes[1, 2].set_title("Predicted Tree Instances")
    axes[1, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path / "tracking_comparison.png", dpi=150, bbox_inches='tight')
        print(f"Saved tracking comparison to {save_path / 'tracking_comparison.png'}")

    plt.show()

    # Individual probability maps
    if predicted_probs.shape[0] > 1:  # Only if we have multiple classes
        n_classes = min(predicted_probs.shape[0], 8)  # Limit to 8 for display
        cols = min(4, n_classes)
        rows = (n_classes + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_classes):
            row, col = i // cols, i % cols
            axes[row, col].imshow(predicted_probs[i], cmap='hot', vmin=0, vmax=1)
            axes[row, col].set_title(f"Tree Instance {i}")
            axes[row, col].axis('off')

        # Hide unused subplots
        for i in range(n_classes, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path / "tree_instance_probabilities.png", dpi=150, bbox_inches='tight')
            print(f"Saved probability maps to {save_path / 'tree_instance_probabilities.png'}")

        plt.show()


# ============================================================================
# FRAME EXTRACTION AND PROCESSING FUNCTIONS
# ============================================================================

def extract_frames_from_cache(
        cache_data: Dict[str, Any],
        frame_indices: Optional[Union[List[int], np.ndarray]] = None,
        require_segmentation: bool = True,
        require_detections: bool = False
) -> List[VideoFrameData]:
    """
    Extract frame data from cache similar to run_tracking_processor.py.
    
    Args:
        cache_data: Loaded cache data from tracking_data_processor.py
        frame_indices: Specific frame indices to extract (default: auto-select)
        require_segmentation: Whether frames must have segmentation data
        require_detections: Whether frames must have detection data
        
    Returns:
        List of VideoFrameData objects
    """
    print("=== EXTRACTING FRAMES FROM CACHE ===")

    # Select frames to process
    if frame_indices is None:
        # Auto-select frames with valid data (similar to run_tracking_processor.py)
        valid_frames = []
        for i, frame in enumerate(cache_data['frames']):
            has_detections = len(frame.get('detections', [])) > 0
            has_segmentation = frame.get('segmentation') is not None

            # Apply requirements
            if require_segmentation and not has_segmentation:
                continue
            if require_detections and not has_detections:
                continue

            valid_frames.append(i)

        if len(valid_frames) < 2:
            raise ValueError(f"Need at least 2 frames with valid data, found {len(valid_frames)}")

        # For video tracking, we want a contiguous sequence
        # Take a reasonable subset if too many frames
        if len(valid_frames) > 50:
            # Take first 50 frames for manageable processing
            frame_indices = valid_frames[:50]
        else:
            frame_indices = valid_frames

    print(f"Selected {len(frame_indices)} frames: {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}")

    # Extract frame data
    video_frames = []
    for i, frame_idx in enumerate(frame_indices):
        frame_data = cache_data['frames'][int(frame_idx)]
        video_frame = VideoFrameData(frame_data, frame_idx)
        video_frames.append(video_frame)

        # if i < 5:  # Print details for first few frames
        #     print(f"  Frame {frame_idx}: {video_frame.filename}, "
        #           f"features: {video_frame.features.shape}, "
        #           f"segmentation: {video_frame.has_segmentation}, "
        #           f"detections: {video_frame.has_detections}")

    return video_frames


def create_reference_segmentation(
        first_frame: VideoFrameData,
        tree_class_id: int,
        patch_size: int = 16
) -> Tuple[torch.Tensor, int, Dict, torch.Tensor] | None:
    """
    DEPRECATED: Use TreesTracker.create_initial_instances() instead.
    
    This function uses the old approach with complex ID remapping.
    The new TreesTracker class uses a simpler approach:
    - background = 0
    - tree instances start from 1
    - no complex remapping needed
    
    Args:
        first_frame: First video frame with segmentation data and detections
        tree_class_id: Class ID for trees (should be 2)
        patch_size: Size of patches for coordinate conversion
        
    Returns:
        Tuple of (one_hot_probs, actual_num_classes, class_mapping, instance_mask)
    """
    print("=== CREATING TREE INSTANCE SEGMENTATION ===")

    if not first_frame.has_segmentation:
        raise ValueError("First frame must have segmentation data")

    if not first_frame.has_detections:
        print("Warning: No detections found, returning None")
        return None

    # Get segmentation mask [H_patches, W_patches]
    seg_mask = torch.from_numpy(first_frame.segmentation).long().cuda()
    h_patches, w_patches = seg_mask.shape

    # Get scaled detections (in DINO coordinates)
    detections = first_frame.detections_scaled

    # Debug: print detection structure
    if len(detections) > 0:
        print(f"Detection sample keys: {list(detections[0].keys())}")
        print(f"First detection: {detections[0]}")

    # Filter for tree detections - try multiple possible key names
    tree_detections = []
    for det in detections:
        if det.get('name') == 'trunk':
            tree_detections.append(det)

    print(f"Total detections: {len(detections)}")
    print(f"Found {len(tree_detections)} tree detections")
    print(f"Segmentation mask shape: {seg_mask.shape}")
    print(f"Target image size: {first_frame.target_image_size}")

    # Create new instance mask
    instance_mask = seg_mask.clone()

    # Convert tree patches (class 2) to individual instances based on bounding boxes
    tree_patches = (seg_mask == tree_class_id)
    print(f"Found {tree_patches.sum().item()} tree patches")

    if len(tree_detections) > 0 and tree_patches.sum() > 0:
        # Start instance numbering from a high number to avoid conflicts
        instance_id = 100

        # Convert each detection box to patch coordinates
        for i, detection in enumerate(tree_detections):
            box = detection['box']
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

            # Convert from image coordinates to patch coordinates
            target_w, target_h = first_frame.target_image_size
            patch_x1 = int(x1 / patch_size)
            patch_y1 = int(y1 / patch_size)
            patch_x2 = int(x2 / patch_size)
            patch_y2 = int(y2 / patch_size)

            # Clamp to patch grid bounds
            patch_x1 = max(0, min(patch_x1, w_patches - 1))
            patch_y1 = max(0, min(patch_y1, h_patches - 1))
            patch_x2 = max(0, min(patch_x2, w_patches - 1))
            patch_y2 = max(0, min(patch_y2, h_patches - 1))

            # Find tree patches within this bounding box
            box_tree_patches = tree_patches[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1]

            if box_tree_patches.sum() > 0:
                # Assign unique instance ID to tree patches in this box
                instance_mask[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1][box_tree_patches] = instance_id
                print(f"  Tree instance {instance_id}: bbox=({patch_x1},{patch_y1})->({patch_x2},{patch_y2}), "
                      f"{box_tree_patches.sum().item()} patches out of total {box_tree_patches.nelement()}")
                instance_id += 1

        # Mark remaining tree patches (outside all bounding boxes) as background
        remaining_tree_patches = (instance_mask == tree_class_id)
        if remaining_tree_patches.sum() > 0:
            print(f"Marking {remaining_tree_patches.sum().item()} tree patches outside detections as background")
            instance_mask[remaining_tree_patches] = 255  # Background

    # Get final unique classes
    unique_classes = instance_mask.unique()
    actual_num_classes = len(unique_classes)

    print(f"Final unique classes: {unique_classes.cpu().numpy()}")
    print(f"Total classes: {actual_num_classes}")

    # Create contiguous class mapping (0, 1, 2, ...)
    class_mapping = {}
    for i, class_id in enumerate(unique_classes):
        class_mapping[class_id.item()] = i

    # Remap to contiguous class IDs
    remapped_mask = torch.zeros_like(instance_mask)
    for orig_id, new_id in class_mapping.items():
        remapped_mask[instance_mask == orig_id] = new_id

    # Convert to one-hot probabilities [H, W, M]
    one_hot_probs = F.one_hot(remapped_mask, actual_num_classes).float()

    print(f"Instance segmentation shape: {one_hot_probs.shape}")

    # Print class distribution
    class_counts = [(class_mapping[k.item()], (remapped_mask == class_mapping[k.item()]).sum().item())
                    for k in unique_classes]
    print(f"Class distribution: {class_counts}")

    return one_hot_probs, actual_num_classes, class_mapping, remapped_mask


def load_image_for_visualization(frame: VideoFrameData, data_folder: Path) -> Image.Image:
    """Load the image and resize to DINO target size for visualization purposes."""
    image_path = data_folder / "images" / frame.filename
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    # Resize to DINO target size to match mask resolution
    target_size = frame.target_image_size  # (width, height)
    image_resized = image.resize(target_size, Image.Resampling.BICUBIC)

    return image_resized


def mask_to_rgb(mask: np.ndarray, num_masks: int) -> np.ndarray:
    """
    Convert segmentation mask to RGB visualization.
    Adapted from segmentation_tracking.ipynb
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Exclude background
    background = mask == 0
    mask = mask - 1
    num_masks = num_masks - 1

    # Choose palette
    if num_masks <= 10:
        mask_rgb = plt.get_cmap("tab10")(mask)[..., :3]
    elif num_masks <= 20:
        mask_rgb = plt.get_cmap("tab20")(mask)[..., :3]
    else:
        mask_rgb = plt.get_cmap("gist_rainbow")(mask / (num_masks - 1))[..., :3]

    mask_rgb = (mask_rgb * 255).astype(np.uint8)
    mask_rgb[background, :] = 0
    return mask_rgb


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
    Main video tracking processor.
    
    Args:
        data_folder: Path to dataset folder
        cache_path: Path to cached DINO features and segmentation  
        tree_class_id: Class ID for trees
        debugging_folder: Optional folder for saving debug visualizations
        frame_indices: List of frame indices to process (default: auto-select)
        patch_size: DINO patch size (should match cache)
        image_size: Target image size (should match cache)
    """
    print("=" * 80)
    print("VIDEO SEGMENTATION TRACKING PROCESSOR")
    print("=" * 80)

    # Check cache validity and parameters
    cache_data = load_tracking_cache(cache_path)
    metadata = cache_data.get('metadata')
    if metadata is None:
        warnings.warn("Unable to fetch cache metadata")
        return

    cache_patch_size = metadata.get('patch_size', None)
    cache_image_size = metadata.get('image_size', None)

    if cache_patch_size != patch_size or cache_image_size != image_size:
        print(f"Cache parameters mismatch:")
        print(f"  Cache: patch_size={cache_patch_size}, image_size={cache_image_size}")
        print(f"  Requested: patch_size={patch_size}, image_size={image_size}")
        warnings.warn(f"Please run tracking_data_processor.py with correct parameters first")
        return

    # STAGE 1: Sort and extract frames from cache
    frame_indices = np.array(frame_indices)
    frame_indices.sort()
    print("\n=== STAGE 1: EXTRACTING FRAMES FROM CACHE ===")
    video_frames = extract_frames_from_cache(
        cache_data,
        frame_indices=frame_indices,
        require_segmentation=True,  # Need segmentation for reference
        require_detections=False
    )

    if len(video_frames) == 0:
        print("No valid frames found for processing")
        return

    print(f"Successfully extracted {len(video_frames)} frames")

    # STAGE 2: Create reference segmentation from first frame
    print("\n=== STAGE 2: CREATING REFERENCE SEGMENTATION ===")
    first_frame = video_frames[0]

    try:
        # Use TreesTracker for consistent ID management
        tracker = TreesTracker(tree_class_id=tree_class_id)
        first_seg_mask = torch.from_numpy(first_frame.segmentation).long().cuda()
        reference_instance_mask, class_mapping = tracker.create_initial_instances(
            segmentation_mask=first_seg_mask,
            detections=first_frame.detections_scaled,
            patch_size=patch_size,
            target_image_size=first_frame.target_image_size
        )

        # Convert to one-hot probabilities
        num_classes = len(class_mapping)
        reference_probs = F.one_hot(reference_instance_mask, num_classes).float()

        print("✓ Reference segmentation created successfully")
    except Exception as e:
        print(f"Error creating reference segmentation: {e}")
        return

    # STAGE 3: Prepare for tracking (foundation for next steps)
    print("\n=== STAGE 3: PREPARING TRACKING FOUNDATION ===")

    # Normalize features (similar to segmentation_tracking.ipynb)
    print("Normalizing features...")
    for frame in video_frames:
        # Features are already loaded as tensors, just normalize
        frame.features = F.normalize(frame.features, dim=-1, p=2)

    print(f"✓ Normalized features for {len(video_frames)} frames")

    # STAGE 4: Choose tracking mode
    if len(video_frames) >= 2 and reference_probs is not None:

        # Option A: Simple two-frame demo
        if len(video_frames) == 2:
            print("\n=== STAGE 4: DEMONSTRATING TREE TRACKING (2 FRAMES) ===")

            first_frame = video_frames[0]
            second_frame = video_frames[1]

            # Track from first to second frame
            try:
                current_probs, predicted_mask_patches, predicted_probs = track_first_to_second_frame(
                    first_frame=first_frame,
                    second_frame=second_frame,
                    reference_probs=reference_probs,
                    num_classes=num_classes,
                    neighborhood_size=12.0,
                    neighborhood_shape="circle",
                    topk=5,
                    temperature=0.2
                )

                print("✓ Tree tracking completed successfully")

                # Visualize results
                if debugging_folder:
                    # Use the modified instance mask instead of original segmentation
                    reference_mask = reference_instance_mask.cpu().numpy()
                    visualize_tracking_results(
                        first_frame=first_frame,
                        second_frame=second_frame,
                        reference_mask=reference_mask,
                        predicted_mask=predicted_mask_patches,
                        predicted_probs=predicted_probs,
                        num_classes=num_classes,
                        data_folder=data_folder,
                        save_path=debugging_folder,
                        ignore_class=None,  # Will use max class index
                        overlay_alpha=0.6,
                        patch_size=patch_size
                    )
                    print("✓ Tracking visualization saved")

            except Exception as e:
                print(f"Error in tracking demo: {e}")
                import traceback
                traceback.print_exc()

        # Option B: Multi-frame consecutive processing
        else:
            print("\n=== STAGE 4: CONSECUTIVE FRAMES TREE TRACKING ===")

            try:
                # Process all frames with memory-based tracking
                tracking_results = process_consecutive_frames(
                    video_frames=video_frames,
                    tree_class_id=tree_class_id,
                    patch_size=patch_size,
                    max_context_length=3,
                    neighborhood_size=12.0,
                    neighborhood_shape="circle",
                    topk=5,
                    temperature=0.2
                )

                print("✓ Consecutive frames tracking completed successfully")

                # Save results if debugging folder is provided
                if debugging_folder:
                    import pickle
                    results_path = debugging_folder / "consecutive_tracking_results.pkl"
                    with open(results_path, 'wb') as f:
                        pickle.dump(tracking_results, f)
                    print(f"✓ Tracking results saved to {results_path}")

                    # Create summary visualization for first few frames
                    print("Creating summary visualizations...")
                    for i in range(min(3, len(tracking_results['frames']) - 1)):
                        frame_result = tracking_results['frames'][i + 1]  # Skip reference frame

                        try:
                            reference_mask = tracking_results['frames'][0]['instance_mask']
                            predicted_mask = frame_result['instance_mask']

                            save_path_frame = debugging_folder / f"frame_{i + 1}_tracking.png"

                            # Simple visualization (you can extend this)
                            import matplotlib.pyplot as plt
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

                            ax1.imshow(mask_to_rgb(reference_mask, tracking_results['frames'][0]['num_classes']))
                            ax1.set_title("Reference (Frame 0)")
                            ax1.axis('off')

                            ax2.imshow(mask_to_rgb(predicted_mask, frame_result['num_classes']))
                            ax2.set_title(f"Tracked (Frame {i + 1})")
                            ax2.axis('off')

                            plt.tight_layout()
                            plt.savefig(save_path_frame, dpi=150, bbox_inches='tight')
                            plt.close()

                        except Exception as viz_e:
                            print(f"Warning: Could not create visualization for frame {i + 1}: {viz_e}")

            except Exception as e:
                print(f"Error in consecutive frames tracking: {e}")
                import traceback
                traceback.print_exc()

    # Create debugging folder if specified
    if debugging_folder:
        debugging_folder.mkdir(exist_ok=True, parents=True)
        print(f"✓ Debugging folder ready: {debugging_folder}")

        # Save reference frame visualization
        try:
            ref_image = load_image_for_visualization(first_frame, data_folder)
            # Use the modified instance mask
            ref_mask_rgb = mask_to_rgb(
                reference_instance_mask.cpu().numpy(),
                num_classes
            )

            # Create side-by-side visualization
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            ax1.imshow(ref_image)
            ax1.set_title(f"Reference Frame {first_frame.frame_idx}")
            ax1.axis('off')

            ax2.imshow(ref_mask_rgb)
            ax2.set_title("Reference Segmentation")
            ax2.axis('off')

            plt.tight_layout()
            plt.savefig(debugging_folder / "reference_frame.png", dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✓ Saved reference frame visualization")

        except Exception as e:
            print(f"Warning: Could not save reference frame visualization: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("VIDEO TRACKING COMPLETE")
    print("=" * 80)
    print(f"Total frames loaded: {len(video_frames)}")
    print(f"Reference frame: {first_frame.frame_idx} ({first_frame.filename})")
    print(f"Feature dimensions: {first_frame.features.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {class_mapping}")

    if len(video_frames) >= 2:
        print(f"Tracking demo: Frame {video_frames[0].frame_idx} -> Frame {video_frames[1].frame_idx}")

    if debugging_folder:
        print(f"Debug outputs saved to: {debugging_folder}")

    print("\nTree instance tracking successfully implemented!")
    print("Next: Extend to full video sequence processing")

    return {
        'video_frames': video_frames,
        'reference_probs': reference_probs,
        'reference_instance_mask': reference_instance_mask,
        'num_classes': num_classes,
        'class_mapping': class_mapping,
        'cache_data': cache_data
    }


if __name__ == "__main__":
    # Example usage - update these paths for your setup
    data_folder = Path("/home/nati/source/data/greeting_dev/bags_parsed/turn8_0")
    cache_path = Path("/home/nati/source/data/greeting_dev/bags_parsed/turn8_0/tracking_cache_img_1024_patch_16.pkl")
    debugging_folder = Path("/home/nati/source/data/greeting_dev/bags_parsed/turn8_0/video_tracking_debug")

    datasets_folder = "/home/nati/source/datasets"
    data_accessor = create_dataset_accessor("bw2508", datasets_folder)
    tree_class_id = get_class_idx_from_name(data_accessor, "tree")

    # Run with custom frame indices (first 20 frames for testing)
    frame_indices = np.arange(200, 220)

    result = main(
        data_folder=data_folder,
        cache_path=cache_path,
        tree_class_id=tree_class_id,
        debugging_folder=debugging_folder,
        frame_indices=frame_indices,
        patch_size=16,
        image_size=1024
    )
