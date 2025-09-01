import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold


def split_to_folds(image_indices: torch.Tensor, n_folds: int = 6, random_state: int = 42) -> List[
    Tuple[np.ndarray, np.ndarray]]:
    """
    Split the dataset into n_folds using image-based splitting to avoid data leakage.

    Args:
        image_indices: Tensor containing image indices for each patch
        n_folds: Number of folds for cross-validation
        random_state: Random seed for reproducibility

    Returns:
        List of (train_mask, val_mask) tuples for each fold
    """
    # Get unique image indices
    unique_images = torch.unique(image_indices).numpy()
    n_images = len(unique_images)

    print(f"Splitting {n_images} images into {n_folds} folds...")

    # Use KFold to split the unique images
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []

    for fold_idx, (train_img_idx, val_img_idx) in enumerate(kfold.split(unique_images)):
        train_images = unique_images[train_img_idx]
        val_images = unique_images[val_img_idx]

        # Create masks for patches based on image membership
        train_mask = np.isin(image_indices.numpy(), train_images)
        val_mask = np.isin(image_indices.numpy(), val_images)

        folds.append((train_mask, val_mask))
        print(f"Fold {fold_idx + 1}: {len(train_images)} train images, {len(val_images)} val images")

    return folds


def calculate_fold_scores(
        features: torch.Tensor,
        labels: torch.Tensor,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        c_values: np.ndarray,
        fold_idx: int
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Calculate scores for a given split across all possible C values.

    Args:
        features: Feature tensor
        labels: Label tensor
        train_mask: Boolean mask for training patches
        val_mask: Boolean mask for validation patches
        c_values: Array of C values to test
        fold_idx: Current fold index for logging

    Returns:
        Tuple of (scores_array, detailed_results_list)
    """
    print(f"Processing fold {fold_idx + 1}...")

    # Prepare data
    train_x = features[train_mask].numpy()
    train_y = (labels[train_mask] > 0).long().numpy()
    val_x = features[val_mask].numpy()
    val_y = (labels[val_mask] > 0).long().numpy()

    scores = np.zeros(len(c_values))
    detailed_results = []

    for j, c in enumerate(c_values):
        print(f"  Training logistic regression with C={c:.2e}")

        # Train classifier
        clf = LogisticRegression(random_state=0, C=c, max_iter=10000).fit(train_x, train_y)

        # Get predictions
        output = clf.predict_proba(val_x)
        precision, recall, thresholds = precision_recall_curve(val_y, output[:, 1])
        ap_score = average_precision_score(val_y, output[:, 1])

        scores[j] = ap_score

        # Store detailed results for plotting
        detailed_results.append({
            'c': c,
            'ap_score': ap_score,
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'fold_idx': fold_idx
        })

        print(f"    C={c:.2e} AP={ap_score:.3f}")

    return scores, detailed_results


def accumulate_cv_statistics(
        all_fold_scores: List[np.ndarray],
        c_values: np.ndarray,
        output_dir: Path
) -> Tuple[float, int]:
    """
    Accumulate statistics across folds and find optimal C.

    Args:
        all_fold_scores: List of score arrays for each fold
        c_values: Array of C values tested
        output_dir: Directory to save plots

    Returns:
        Tuple of (optimal_c, optimal_c_index)
    """
    # Convert to numpy array for easier manipulation
    scores_matrix = np.array(all_fold_scores)  # Shape: (n_folds, n_c_values)

    # Calculate mean and std across folds
    mean_scores = scores_matrix.mean(axis=0)
    std_scores = scores_matrix.std(axis=0)

    # Find optimal C
    optimal_c_idx = np.argmax(mean_scores)
    optimal_c = c_values[optimal_c_idx]

    print(f"Cross-validation results:")
    print(f"Mean scores: {mean_scores}")
    print(f"Std scores: {std_scores}")
    print(f"Optimal C: {optimal_c:.2e} (index {optimal_c_idx}) with mean AP: {mean_scores[optimal_c_idx]:.3f}")

    # Create plots directory
    plots_dir = output_dir / "optimal_clf_debugging"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot cross-validation results
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(c_values)), mean_scores, yerr=std_scores,
                 marker='o', capsize=5, capthick=2)
    plt.xticks(range(len(c_values)), [f"{c:.0e}" for c in c_values], rotation=45)
    plt.xlabel('Regularization Parameter C')
    plt.ylabel('Average Precision Score')
    plt.title('Cross-Validation Results: Mean AP Score vs C')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=float(optimal_c_idx), color='red', linestyle='--',
                label=f'Optimal C = {optimal_c:.1e}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "cv_mean_scores.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot individual fold results
    plt.figure(figsize=(12, 8))
    for fold_idx, fold_scores in enumerate(all_fold_scores):
        plt.plot(range(len(c_values)), fold_scores,
                 marker='o', alpha=0.7, label=f'Fold {fold_idx + 1}')

    plt.plot(range(len(c_values)), mean_scores,
             marker='s', linewidth=3, color='black', label='Mean')
    plt.xticks(range(len(c_values)), [f"{c:.0e}" for c in c_values], rotation=45)
    plt.xlabel('Regularization Parameter C')
    plt.ylabel('Average Precision Score')
    plt.title('Cross-Validation Results: All Folds')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=float(optimal_c_idx), color='red', linestyle='--',
                label=f'Optimal C = {optimal_c:.1e}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "cv_all_folds.png", dpi=300, bbox_inches='tight')
    plt.close()

    return optimal_c, int(optimal_c_idx)


def retrain_optimal_classifier(
        features: torch.Tensor,
        labels: torch.Tensor,
        optimal_c: float,
        class_name: str,
        output_dir: Path
) -> LogisticRegression:
    """
    Retrain optimal classifier on full dataset and save to cache.

    Args:
        features: Full feature tensor
        labels: Full label tensor
        optimal_c: Optimal C value found via cross-validation
        class_name: Name of the class being classified
        output_dir: Directory to save the model

    Returns:
        Trained classifier
    """
    print(f"Retraining classifier with optimal C={optimal_c:.2e} on full dataset...")

    # Prepare full dataset
    x_full = features.numpy()
    y_full = (labels > 0).long().numpy()

    # Train final classifier
    clf = LogisticRegression(
        random_state=0,
        C=optimal_c,
        max_iter=100000,
        verbose=1
    ).fit(x_full, y_full)

    # Save classifier
    classifier_path = output_dir / f"fg_classifier_{class_name}_c_{optimal_c}.pkl"
    print(f"Saving trained classifier to {classifier_path}")
    with open(classifier_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"Final classifier training completed!")
    print(f"Training accuracy: {clf.score(x_full, y_full):.3f}")

    return clf
