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


def compute_sparsity(model: nn.Module, threshold: float = 1e-6) -> Dict[str, float]:
    """Compute sparsity statistics for model parameters.
    
    Args:
        model: PyTorch model
        threshold: Values below this are considered zero
        
    Returns:
        Dictionary with sparsity metrics per layer and overall
    """
    sparsity_stats = {}
    total_params = 0
    total_zeros = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_data = param.data.abs()
            num_zeros = (param_data < threshold).sum().item()
            num_params = param.numel()
            
            layer_sparsity = num_zeros / num_params
            sparsity_stats[name] = layer_sparsity
            
            total_params += num_params
            total_zeros += num_zeros
    
    sparsity_stats["overall"] = total_zeros / total_params if total_params > 0 else 0.0
    return sparsity_stats


def compare_models(
    model1: nn.Module,
    model2: nn.Module,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> Dict[str, Any]:
    """Compare two models in terms of size and complexity.
    
    Args:
        model1: First model
        model2: Second model
        model1_name: Name for first model
        model2_name: Name for second model
        
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        model1_name: {
            "parameters": count_parameters(model1),
            "trainable_parameters": count_parameters(model1, trainable_only=True),
            "size_mb": get_model_size_mb(model1),
            "sparsity": compute_sparsity(model1)["overall"]
        },
        model2_name: {
            "parameters": count_parameters(model2),
            "trainable_parameters": count_parameters(model2, trainable_only=True),
            "size_mb": get_model_size_mb(model2),
            "sparsity": compute_sparsity(model2)["overall"]
        }
    }
    
    # Compute reduction ratios
    comparison["reduction"] = {
        "parameter_ratio": comparison[model2_name]["parameters"] / comparison[model1_name]["parameters"],
        "size_ratio": comparison[model2_name]["size_mb"] / comparison[model1_name]["size_mb"]
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
    if schedule_type == "linear":
        return list(np.linspace(start_value, end_value, num_steps))
    elif schedule_type == "exponential":
        return list(np.geomspace(start_value, end_value, num_steps))
    elif schedule_type == "cosine":
        steps = np.arange(num_steps)
        cosine_vals = 0.5 * (1 + np.cos(np.pi * steps / (num_steps - 1)))
        return list(start_value + (end_value - start_value) * (1 - cosine_vals))
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
