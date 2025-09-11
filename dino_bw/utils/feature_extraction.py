#!/usr/bin/env python3
"""
Feature extraction utilities for tree tracking using DINOv3 features.

This module contains functions for:
- Tree object extraction from frames
- Unified PCA computation across multiple frames
- Feature processing and analysis
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from sklearn.decomposition import PCA


@dataclass
class TreeObject:
    """Data structure for a tree object with its features and metadata."""
    object_id: int
    bbox: Dict[str, float]  # x1, y1, x2, y2 in DINO coordinates
    original_bbox: Dict[str, float]  # x1, y1, x2, y2 in original coordinates
    patch_indices: List[Tuple[int, int]]  # (row, col) indices of tree patches
    feature_vectors: torch.Tensor  # Features of all tree patches in this object
    averaged_feature: torch.Tensor  # Mean feature vector for this object
    confidence: float  # Detection confidence
    frame_id: int

    def __post_init__(self):
        """Compute averaged feature after initialization."""
        if len(self.feature_vectors) > 0:
            self.averaged_feature = torch.mean(self.feature_vectors, dim=0)
        else:
            self.averaged_feature = torch.zeros(
                self.feature_vectors.shape[1] if len(self.feature_vectors.shape) > 1 else 384)
