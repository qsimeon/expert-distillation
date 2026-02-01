"""Utility functions for expert knowledge distillation.

This module provides helper functions for model analysis, visualization,
and common operations used in the knowledge distillation process.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import logging
from collections import defaultdict
import json
import os

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Total number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate the size of a model in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def compute_sparsity(model: nn.Module, threshold: float = 1e-6) -> Dict[str, Any]:
    """Compute sparsity statistics for model parameters.
    
    Args:
        model: PyTorch model
        threshold: Values below this are considered zero
        
    Returns:
        Dictionary with sparsity metrics per layer and overall
    """
    layer_sparsity = {}
    total_params = 0
    total_zeros = 0
    num_sparse_layers = 0
    total_layers = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_data = param.data.abs()
            num_zeros = (param_data < threshold).sum().item()
            num_params = param.numel()
            
            layer_sp = num_zeros / num_params
            layer_sparsity[name] = layer_sp
            
            total_params += num_params
            total_zeros += num_zeros
            total_layers += 1
            if layer_sp > 0.01:  # More than 1% sparse
                num_sparse_layers += 1
    
    return {
        "overall_sparsity": total_zeros / total_params if total_params > 0 else 0.0,
        "layer_sparsity": layer_sparsity,
        "num_sparse_layers": num_sparse_layers,
        "total_layers": total_layers,
        "total_zeros": total_zeros,
        "total_params": total_params
    }


def compare_models(
    model1: nn.Module,
    model2: nn.Module,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> Dict[str, Any]:
    """Compare two models in terms of size and complexity.
    
    Args:
        model1: First model (typically teacher/larger)
        model2: Second model (typically student/smaller)
        model1_name: Name for first model
        model2_name: Name for second model
        
    Returns:
        Dictionary with comparison metrics
    """
    model1_params = count_parameters(model1)
    model2_params = count_parameters(model2)
    model1_size = get_model_size_mb(model1)
    model2_size = get_model_size_mb(model2)
    
    comparison = {
        model1_name: {
            "parameters": model1_params,
            "trainable_parameters": count_parameters(model1, trainable_only=True),
            "size_mb": model1_size,
            "sparsity": compute_sparsity(model1)["overall_sparsity"]
        },
        model2_name: {
            "parameters": model2_params,
            "trainable_parameters": count_parameters(model2, trainable_only=True),
            "size_mb": model2_size,
            "sparsity": compute_sparsity(model2)["overall_sparsity"]
        },
        # Direct reduction percentages for easy access
        "parameter_reduction": (1 - model2_params / model1_params) * 100 if model1_params > 0 else 0.0,
        "size_reduction": (1 - model2_size / model1_size) * 100 if model1_size > 0 else 0.0,
        "parameter_ratio": model2_params / model1_params if model1_params > 0 else 0.0,
        "size_ratio": model2_size / model1_size if model1_size > 0 else 0.0
    }
    
    return comparison


def analyze_layer_importance(
    importance_scores: Dict[str, torch.Tensor],
    top_k: int = 10
) -> Dict[str, List[Tuple[str, float]]]:
    """Analyze and rank layers by importance.
    
    Args:
        importance_scores: Dictionary mapping layer names to importance tensors
        top_k: Number of top layers to return
        
    Returns:
        Dictionary with top and bottom k layers
    """
    # Compute mean importance per layer
    layer_importance = {}
    for name, scores in importance_scores.items():
        layer_importance[name] = scores.mean().item()
    
    # Sort by importance
    sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "top_layers": sorted_layers[:top_k],
        "bottom_layers": sorted_layers[-top_k:],
        "all_layers": sorted_layers
    }


def create_knowledge_mask(
    importance_scores: Dict[str, torch.Tensor],
    threshold: float
) -> Dict[str, torch.Tensor]:
    """Create binary masks for knowledge retention based on importance.
    
    Args:
        importance_scores: Dictionary mapping layer names to importance tensors
        threshold: Importance threshold for retention
        
    Returns:
        Dictionary mapping layer names to binary masks
    """
    masks = {}
    for name, scores in importance_scores.items():
        masks[name] = (scores > threshold).float()
    return masks


def apply_knowledge_mask(
    model: nn.Module,
    masks: Dict[str, torch.Tensor]
) -> None:
    """Apply knowledge masks to model parameters in-place.
    
    Args:
        model: PyTorch model
        masks: Dictionary mapping parameter names to binary masks
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.data *= masks[name]


def compute_knowledge_retention(
    original_importance: Dict[str, torch.Tensor],
    current_model: nn.Module,
    threshold: float = 1e-6
) -> float:
    """Compute the fraction of important knowledge retained.
    
    Args:
        original_importance: Original importance scores before pruning
        current_model: Current model after pruning
        threshold: Threshold for considering a weight as non-zero
        
    Returns:
        Fraction of important knowledge retained (0 to 1)
    """
    total_important = 0
    retained_important = 0
    
    for name, param in current_model.named_parameters():
        if name in original_importance:
            importance = original_importance[name]
            
            # Important weights are those above median importance
            importance_threshold = importance.median()
            important_mask = importance > importance_threshold
            
            # Check which important weights are still non-zero
            non_zero_mask = param.data.abs() > threshold
            
            total_important += important_mask.sum().item()
            retained_important += (important_mask & non_zero_mask).sum().item()
    
    return retained_important / total_important if total_important > 0 else 0.0


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    metric: str = "accuracy"
) -> float:
    """Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        metric: Metric to compute ('accuracy', 'loss')
        
    Returns:
        Evaluation metric value
    """
    model.eval()
    model = model.to(device)
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            if metric == "accuracy":
                predictions = outputs.argmax(dim=-1)
                total_correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)
            elif metric == "loss":
                loss = nn.functional.cross_entropy(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                total_samples += targets.size(0)
    
    if metric == "accuracy":
        return total_correct / total_samples if total_samples > 0 else 0.0
    elif metric == "loss":
        return total_loss / total_samples if total_samples > 0 else 0.0
    else:
        raise ValueError(f"Unknown metric: {metric}")


def get_layer_statistics(
    model: nn.Module,
    layer_name: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """Get statistical information about model layers.
    
    Args:
        model: PyTorch model
        layer_name: Specific layer to analyze (None for all layers)
        
    Returns:
        Dictionary with statistics for each layer
    """
    stats = {}
    
    for name, param in model.named_parameters():
        if layer_name is None or name == layer_name:
            param_data = param.data.cpu().numpy()
            
            stats[name] = {
                "mean": float(np.mean(param_data)),
                "std": float(np.std(param_data)),
                "min": float(np.min(param_data)),
                "max": float(np.max(param_data)),
                "median": float(np.median(param_data)),
                "num_params": param.numel(),
                "shape": list(param.shape)
            }
    
    return stats


def save_distillation_report(
    report_path: str,
    teacher_model: nn.Module,
    student_model: nn.Module,
    training_history: Dict[str, List[float]],
    importance_scores: Optional[Dict[str, torch.Tensor]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save a comprehensive report of the distillation process.
    
    Args:
        report_path: Path to save the report (JSON format)
        teacher_model: Teacher model
        student_model: Student model
        training_history: Training metrics history
        importance_scores: Optional importance scores
        metadata: Additional metadata to include
    """
    report = {
        "model_comparison": compare_models(teacher_model, student_model, "Teacher", "Student"),
        "training_history": {
            key: [float(v) for v in values]
            for key, values in training_history.items()
        },
        "student_sparsity": compute_sparsity(student_model),
        "metadata": metadata or {}
    }
    
    # Convert any non-serializable values
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    report = make_serializable(report)
    
    if importance_scores:
        analysis = analyze_layer_importance(importance_scores)
        report["layer_importance"] = {
            "top_layers": [(name, float(score)) for name, score in analysis["top_layers"]],
            "bottom_layers": [(name, float(score)) for name, score in analysis["bottom_layers"]]
        }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Distillation report saved to {report_path}")


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to [0, 1] range.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Normalized tensor
    """
    min_val = tensor.min()
    max_val = tensor.max()
    
    if max_val - min_val > 1e-8:
        return (tensor - min_val) / (max_val - min_val)
    return tensor


def compute_task_alignment(
    model: nn.Module,
    task_dataloader: torch.utils.data.DataLoader,
    general_dataloader: torch.utils.data.DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """Compute how well the model is aligned to the specific task vs general knowledge.
    
    Args:
        model: Model to evaluate
        task_dataloader: DataLoader for task-specific data
        general_dataloader: DataLoader for general/diverse data
        device: Device to run on
        
    Returns:
        Dictionary with task and general performance metrics
    """
    task_accuracy = evaluate_model(model, task_dataloader, device, "accuracy")
    general_accuracy = evaluate_model(model, general_dataloader, device, "accuracy")
    
    alignment_score = task_accuracy / (general_accuracy + 1e-8)
    
    return {
        "task_accuracy": task_accuracy,
        "general_accuracy": general_accuracy,
        "alignment_score": alignment_score,
        "specialization_ratio": task_accuracy / (task_accuracy + general_accuracy + 1e-8)
    }


def create_progressive_schedule(
    start_value: float,
    end_value: float,
    num_steps: int,
    schedule_type: str = "linear"
) -> List[float]:
    """Create a progressive schedule for hyperparameters.
    
    Args:
        start_value: Initial value
        end_value: Final value
        num_steps: Number of steps
        schedule_type: Type of schedule ('linear', 'exponential', 'cosine')
        
    Returns:
        List of values for each step
    """
    if num_steps <= 1:
        return [start_value]
        
    if schedule_type == "linear":
        return list(np.linspace(start_value, end_value, num_steps))
    elif schedule_type == "exponential":
        # Handle case where start or end is 0
        if start_value <= 0 or end_value <= 0:
            return list(np.linspace(start_value, end_value, num_steps))
        return list(np.geomspace(start_value, end_value, num_steps))
    elif schedule_type == "cosine":
        steps = np.arange(num_steps)
        cosine_vals = 0.5 * (1 + np.cos(np.pi * steps / (num_steps - 1)))
        return list(start_value + (end_value - start_value) * (1 - cosine_vals))
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> None:
    """Plot comprehensive training history visualization.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save the figure (displays if None)
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not installed. Skipping visualization.")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Expert Knowledge Distillation - Training Progress', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    epochs = range(1, len(history.get("total_loss", [])) + 1)
    
    # Color palette
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'tertiary': '#F18F01',
        'quaternary': '#C73E1D',
        'success': '#3A7D44',
        'light': '#E8E8E8'
    }
    
    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    if "total_loss" in history:
        ax1.plot(epochs, history["total_loss"], 'o-', color=colors['primary'], 
                 linewidth=2, markersize=4, label='Total Loss')
    if "task_loss" in history:
        ax1.plot(epochs, history["task_loss"], 's--', color=colors['secondary'], 
                 linewidth=2, markersize=4, label='Task Loss')
    if "distill_loss" in history:
        ax1.plot(epochs, history["distill_loss"], '^:', color=colors['tertiary'], 
                 linewidth=2, markersize=4, label='Distillation Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss Curves During Distillation', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    ax2 = axes[0, 1]
    if "train_accuracy" in history:
        train_acc = [a * 100 for a in history["train_accuracy"]]
        ax2.plot(epochs, train_acc, 'o-', color=colors['primary'], 
                 linewidth=2, markersize=4, label='Training')
    if "val_accuracy" in history and any(history["val_accuracy"]):
        val_acc = [a * 100 for a in history["val_accuracy"]]
        ax2.plot(epochs, val_acc, 's-', color=colors['success'], 
                 linewidth=2, markersize=4, label='Validation')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Model Accuracy Over Training', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Alpha schedule
    ax3 = axes[0, 2]
    if "alpha" in history:
        ax3.fill_between(epochs, history["alpha"], alpha=0.3, color=colors['tertiary'])
        ax3.plot(epochs, history["alpha"], 'o-', color=colors['tertiary'], 
                 linewidth=2, markersize=4)
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Balanced (α=0.5)')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Alpha (α)', fontsize=11)
    ax3.set_title('Distillation Weight Schedule', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Add annotation for alpha meaning
    ax3.annotate('← More distillation', xy=(0.05, 0.85), xycoords='axes fraction',
                 fontsize=9, color='gray')
    ax3.annotate('More task focus →', xy=(0.05, 0.1), xycoords='axes fraction',
                 fontsize=9, color='gray')
    
    # Plot 4: Sparsity over time
    ax4 = axes[1, 0]
    if "sparsity" in history:
        sparsity_pct = [s * 100 for s in history["sparsity"]]
        ax4.fill_between(epochs, sparsity_pct, alpha=0.3, color=colors['quaternary'])
        ax4.plot(epochs, sparsity_pct, 'o-', color=colors['quaternary'], 
                 linewidth=2, markersize=4)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Sparsity (%)', fontsize=11)
    ax4.set_title('Model Sparsity (Knowledge Removal)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Pruned parameters per epoch
    ax5 = axes[1, 1]
    if "pruned_params" in history:
        ax5.bar(epochs, history["pruned_params"], color=colors['secondary'], 
                alpha=0.7, edgecolor='black', linewidth=0.5)
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('Parameters Pruned', fontsize=11)
    ax5.set_title('Knowledge Pruning Per Epoch', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Combined efficiency view
    ax6 = axes[1, 2]
    if "train_accuracy" in history and "sparsity" in history:
        train_acc = [a * 100 for a in history["train_accuracy"]]
        sparsity_pct = [s * 100 for s in history["sparsity"]]
        
        ax6_twin = ax6.twinx()
        
        line1, = ax6.plot(epochs, train_acc, 'o-', color=colors['success'], 
                          linewidth=2, markersize=4, label='Accuracy')
        line2, = ax6_twin.plot(epochs, sparsity_pct, 's-', color=colors['quaternary'], 
                                linewidth=2, markersize=4, label='Sparsity')
        
        ax6.set_xlabel('Epoch', fontsize=11)
        ax6.set_ylabel('Accuracy (%)', fontsize=11, color=colors['success'])
        ax6_twin.set_ylabel('Sparsity (%)', fontsize=11, color=colors['quaternary'])
        ax6.set_title('Accuracy vs Sparsity Trade-off', fontsize=12, fontweight='bold')
        
        ax6.tick_params(axis='y', labelcolor=colors['success'])
        ax6_twin.tick_params(axis='y', labelcolor=colors['quaternary'])
        
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='center right', fontsize=9)
    
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_model_comparison(
    comparison: Dict[str, Any],
    teacher_name: str = "Teacher",
    student_name: str = "Student", 
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """Plot model comparison visualization.
    
    Args:
        comparison: Model comparison dictionary from compare_models()
        teacher_name: Name of teacher model
        student_name: Name of student model
        save_path: Path to save the figure
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not installed. Skipping visualization.")
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Teacher vs Student Model Comparison', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    colors = ['#2E86AB', '#F18F01']
    
    # Get data
    teacher_data = comparison.get(teacher_name, {})
    student_data = comparison.get(student_name, {})
    
    # Plot 1: Parameters
    ax1 = axes[0]
    params = [teacher_data.get('parameters', 0), student_data.get('parameters', 0)]
    bars1 = ax1.bar([teacher_name, student_name], params, color=colors, 
                    edgecolor='black', linewidth=1)
    ax1.set_ylabel('Parameters', fontsize=11)
    ax1.set_title('Model Parameters', fontsize=12, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars1, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.02,
                 f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add reduction annotation
    reduction = comparison.get('parameter_reduction', 0)
    ax1.annotate(f'{reduction:.1f}% reduction', 
                 xy=(1, params[1]), xytext=(1.3, params[0]*0.5),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=10, color='gray')
    
    # Plot 2: Model Size
    ax2 = axes[1]
    sizes = [teacher_data.get('size_mb', 0), student_data.get('size_mb', 0)]
    bars2 = ax2.bar([teacher_name, student_name], sizes, color=colors,
                    edgecolor='black', linewidth=1)
    ax2.set_ylabel('Size (MB)', fontsize=11)
    ax2.set_title('Model Size', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars2, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.02,
                 f'{val:.2f} MB', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    size_reduction = comparison.get('size_reduction', 0)
    ax2.annotate(f'{size_reduction:.1f}% reduction',
                 xy=(1, sizes[1]), xytext=(1.3, sizes[0]*0.5),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=10, color='gray')
    
    # Plot 3: Compression summary pie chart
    ax3 = axes[2]
    compression = [comparison.get('parameter_ratio', 1) * 100, 
                   (1 - comparison.get('parameter_ratio', 1)) * 100]
    
    wedges, texts, autotexts = ax3.pie(
        compression, 
        labels=['Student', 'Removed'],
        colors=['#F18F01', '#E8E8E8'],
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0),
        wedgeprops=dict(edgecolor='black', linewidth=1)
    )
    ax3.set_title('Knowledge Compression', fontsize=12, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"Model comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_layer_sparsity(
    model: nn.Module,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """Plot per-layer sparsity visualization.
    
    Args:
        model: PyTorch model
        save_path: Path to save the figure
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Skipping visualization.")
        return
    
    sparsity_stats = compute_sparsity(model)
    layer_sparsity = sparsity_stats.get("layer_sparsity", {})
    
    if not layer_sparsity:
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Simplify layer names
    layers = list(layer_sparsity.keys())
    sparsities = [layer_sparsity[l] * 100 for l in layers]
    
    # Create short names
    short_names = []
    for name in layers:
        parts = name.split('.')
        if len(parts) > 2:
            short_names.append('.'.join(parts[-2:]))
        else:
            short_names.append(name)
    
    # Color by sparsity level
    colors = ['#3A7D44' if s < 20 else '#F18F01' if s < 50 else '#C73E1D' 
              for s in sparsities]
    
    bars = ax.barh(range(len(layers)), sparsities, color=colors, 
                   edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel('Sparsity (%)', fontsize=11)
    ax.set_title('Per-Layer Sparsity Distribution', fontsize=12, fontweight='bold')
    
    # Add overall sparsity line
    overall = sparsity_stats["overall_sparsity"] * 100
    ax.axvline(x=overall, color='#2E86AB', linestyle='--', linewidth=2, 
               label=f'Overall: {overall:.1f}%')
    ax.legend(loc='lower right', fontsize=10)
    
    ax.set_xlim([0, 100])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"Layer sparsity plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_distillation_summary(
    teacher_acc: float,
    student_acc: float,
    comparison: Dict[str, Any],
    sparsity: float,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """Create a summary dashboard of distillation results.
    
    Args:
        teacher_acc: Teacher model accuracy (0-100)
        student_acc: Student model accuracy (0-100)
        comparison: Model comparison dictionary
        sparsity: Final sparsity (0-1)
        save_path: Path to save the figure
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, Circle
    except ImportError:
        logger.warning("matplotlib not installed. Skipping visualization.")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'Knowledge Distillation Results', fontsize=18, 
            fontweight='bold', ha='center', va='center')
    
    # Metric boxes
    def draw_metric_box(x, y, width, height, title, value, unit='', color='#2E86AB'):
        rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                               boxstyle="round,pad=0.05", facecolor='white',
                               edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.3, title, fontsize=10, ha='center', va='center', 
                color='gray')
        ax.text(x, y - 0.2, f'{value:.1f}{unit}', fontsize=20, ha='center', 
                va='center', fontweight='bold', color=color)
    
    # Row 1: Accuracy comparison
    draw_metric_box(2.5, 5.5, 2, 1.5, 'Teacher Accuracy', teacher_acc, '%', '#2E86AB')
    draw_metric_box(5, 5.5, 2, 1.5, 'Student Accuracy', student_acc, '%', '#F18F01')
    
    acc_diff = student_acc - teacher_acc
    color = '#3A7D44' if acc_diff >= 0 else '#C73E1D'
    sign = '+' if acc_diff >= 0 else ''
    draw_metric_box(7.5, 5.5, 2, 1.5, 'Accuracy Δ', acc_diff, '%', color)
    
    # Row 2: Compression metrics
    param_reduction = comparison.get('parameter_reduction', 0)
    size_reduction = comparison.get('size_reduction', 0)
    
    draw_metric_box(2.5, 3.5, 2, 1.5, 'Param Reduction', param_reduction, '%', '#A23B72')
    draw_metric_box(5, 3.5, 2, 1.5, 'Size Reduction', size_reduction, '%', '#A23B72')
    draw_metric_box(7.5, 3.5, 2, 1.5, 'Sparsity', sparsity * 100, '%', '#C73E1D')
    
    # Bottom: Key insight
    if acc_diff >= -2:
        insight = "[OK] Successful distillation: Student maintains accuracy with significant compression"
        insight_color = '#3A7D44'
    else:
        insight = "[!] Trade-off: Accuracy reduced for compression (consider adjusting parameters)"
        insight_color = '#F18F01'
    
    ax.text(5, 1.5, insight, fontsize=11, ha='center', va='center', 
            color=insight_color, style='italic')
    
    # Speedup estimate (rough: inversely proportional to size)
    speedup = 1 / (comparison.get('size_ratio', 1) + 0.01)
    ax.text(5, 0.8, f'Estimated Inference Speedup: ~{speedup:.1f}x', 
            fontsize=10, ha='center', va='center', color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"Summary dashboard saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
