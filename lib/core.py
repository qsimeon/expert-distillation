"""Core module for expert knowledge distillation.

This module provides the main functionality for distilling knowledge from a large
teacher model into a smaller, specialized student model. It implements progressive
knowledge removal and task-specific optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation process.
    
    Attributes:
        temperature: Temperature for softening probability distributions
        alpha: Weight for distillation loss (1-alpha for task loss)
        pruning_rate: Rate at which to prune neurons/weights per iteration
        specialization_threshold: Minimum importance score to retain knowledge
        max_iterations: Maximum number of distillation iterations
        task_weight_schedule: How to adjust task vs distillation weight over time
    """
    temperature: float = 3.0
    alpha: float = 0.7
    pruning_rate: float = 0.1
    specialization_threshold: float = 0.3
    max_iterations: int = 100
    task_weight_schedule: str = "linear"  # 'linear', 'exponential', 'constant'


class KnowledgeDistiller:
    """Main class for performing expert knowledge distillation.
    
    This class handles the distillation process from a large teacher model to a
    smaller student model, with progressive knowledge removal to create a
    specialized expert model.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: DistillationConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the knowledge distiller.
        
        Args:
            teacher_model: Pre-trained large model with broad knowledge
            student_model: Smaller model to be specialized
            config: Configuration for distillation process
            device: Device to run models on
        """
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.config = config
        self.device = device
        self.teacher.eval()  # Teacher is always in eval mode
        
        # Track importance scores for knowledge components
        self.knowledge_importance: Dict[str, torch.Tensor] = {}
        self.iteration = 0
        
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """Compute knowledge distillation loss using KL divergence.
        
        Args:
            student_logits: Raw outputs from student model
            teacher_logits: Raw outputs from teacher model
            temperature: Temperature for softening distributions
            
        Returns:
            Distillation loss value
        """
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        
        distill_loss = F.kl_div(
            soft_student,
            soft_targets,
            reduction="batchmean"
        ) * (temperature ** 2)
        
        return distill_loss
    
    def compute_knowledge_importance(
        self,
        task_dataloader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute importance scores for different knowledge components.
        
        Uses gradient-based importance estimation to determine which parts of
        the model are most relevant for the target task.
        
        Args:
            task_dataloader: DataLoader with task-specific data
            num_batches: Number of batches to use (None for all)
            
        Returns:
            Dictionary mapping layer names to importance scores
        """
        self.student.eval()
        importance_scores = {}
        
        # Initialize importance accumulators
        for name, param in self.student.named_parameters():
            if param.requires_grad:
                importance_scores[name] = torch.zeros_like(param.data)
        
        batches_processed = 0
        for batch_idx, (inputs, targets) in enumerate(task_dataloader):
            if num_batches and batch_idx >= num_batches:
                break
                
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.student.zero_grad()
            outputs = self.student(inputs)
            
            # Compute loss and gradients
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            # Accumulate squared gradients as importance measure
            for name, param in self.student.named_parameters():
                if param.grad is not None:
                    importance_scores[name] += param.grad.data.abs()
            
            batches_processed += 1
        
        # Normalize importance scores
        for name in importance_scores:
            importance_scores[name] /= batches_processed
            
        self.knowledge_importance = importance_scores
        return importance_scores
    
    def prune_irrelevant_knowledge(
        self,
        pruning_rate: Optional[float] = None
    ) -> int:
        """Prune weights with low importance scores.
        
        Args:
            pruning_rate: Fraction of weights to prune (uses config if None)
            
        Returns:
            Number of parameters pruned
        """
        if pruning_rate is None:
            pruning_rate = self.config.pruning_rate
            
        if not self.knowledge_importance:
            logger.warning("No importance scores computed. Skipping pruning.")
            return 0
        
        total_pruned = 0
        
        for name, param in self.student.named_parameters():
            if name not in self.knowledge_importance or not param.requires_grad:
                continue
                
            importance = self.knowledge_importance[name]
            
            # Determine threshold for pruning
            threshold = torch.quantile(importance.flatten(), pruning_rate)
            
            # Create mask for weights to keep
            mask = importance > threshold
            
            # Apply pruning
            with torch.no_grad():
                pruned_count = (~mask).sum().item()
                param.data *= mask.float()
                total_pruned += pruned_count
        
        logger.info(f"Pruned {total_pruned} parameters")
        return total_pruned
    
    def get_alpha_schedule(self, iteration: int) -> float:
        """Get the alpha value based on iteration and schedule.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Alpha value for balancing distillation and task loss
        """
        if self.config.task_weight_schedule == "constant":
            return self.config.alpha
        elif self.config.task_weight_schedule == "linear":
            # Linearly decrease alpha (increase task focus)
            progress = iteration / self.config.max_iterations
            return self.config.alpha * (1 - progress)
        elif self.config.task_weight_schedule == "exponential":
            # Exponentially decrease alpha
            progress = iteration / self.config.max_iterations
            return self.config.alpha * np.exp(-3 * progress)
        else:
            return self.config.alpha
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        task_loss_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """Perform a single training step with distillation.
        
        Args:
            inputs: Input batch
            targets: Target labels
            optimizer: Optimizer for student model
            task_loss_fn: Custom task loss function (uses CrossEntropy if None)
            
        Returns:
            Dictionary with loss components
        """
        self.student.train()
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Get predictions from both models
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        
        student_logits = self.student(inputs)
        
        # Compute losses
        if task_loss_fn is None:
            task_loss = F.cross_entropy(student_logits, targets)
        else:
            task_loss = task_loss_fn(student_logits, targets)
        
        distill_loss = self.distillation_loss(
            student_logits,
            teacher_logits,
            self.config.temperature
        )
        
        # Combine losses with schedule
        alpha = self.get_alpha_schedule(self.iteration)
        total_loss = alpha * distill_loss + (1 - alpha) * task_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "distill_loss": distill_loss.item(),
            "task_loss": task_loss.item(),
            "alpha": alpha
        }
    
    def distill(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        prune_every: int = 5,
        eval_fn: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """Run the complete distillation process.
        
        Args:
            train_dataloader: DataLoader for task-specific training data
            optimizer: Optimizer for student model
            num_epochs: Number of training epochs
            prune_every: Prune every N epochs
            eval_fn: Optional evaluation function
            
        Returns:
            Training history with metrics
        """
        history = {
            "total_loss": [],
            "distill_loss": [],
            "task_loss": [],
            "pruned_params": []
        }
        
        for epoch in range(num_epochs):
            epoch_losses = {"total_loss": 0, "distill_loss": 0, "task_loss": 0}
            num_batches = 0
            
            for inputs, targets in train_dataloader:
                losses = self.train_step(inputs, targets, optimizer)
                
                for key in epoch_losses:
                    epoch_losses[key] += losses[key]
                num_batches += 1
                self.iteration += 1
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
                history[key].append(epoch_losses[key])
            
            # Periodic pruning
            if (epoch + 1) % prune_every == 0:
                self.compute_knowledge_importance(train_dataloader, num_batches=10)
                pruned = self.prune_irrelevant_knowledge()
                history["pruned_params"].append(pruned)
            
            # Evaluation
            if eval_fn is not None:
                eval_metric = eval_fn(self.student)
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                          f"Loss: {epoch_losses['total_loss']:.4f} - "
                          f"Eval: {eval_metric:.4f}")
            else:
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                          f"Loss: {epoch_losses['total_loss']:.4f}")
        
        return history


class SpecializedExpert:
    """Wrapper for a specialized expert model after distillation."""
    
    def __init__(self, model: nn.Module, task_name: str, metadata: Optional[Dict] = None):
        """Initialize specialized expert.
        
        Args:
            model: The distilled specialized model
            task_name: Name of the specialized task
            metadata: Additional metadata about the expert
        """
        self.model = model
        self.task_name = task_name
        self.metadata = metadata or {}
        
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Make predictions using the specialized model.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Model predictions
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs
    
    def save(self, path: str) -> None:
        """Save the specialized expert model.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "task_name": self.task_name,
            "metadata": self.metadata
        }, path)
        logger.info(f"Saved specialized expert to {path}")
    
    @classmethod
    def load(cls, path: str, model_class: nn.Module) -> "SpecializedExpert":
        """Load a specialized expert model.
        
        Args:
            path: Path to load the model from
            model_class: The model class to instantiate
            
        Returns:
            Loaded SpecializedExpert instance
        """
        checkpoint = torch.load(path)
        model = model_class
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return cls(
            model=model,
            task_name=checkpoint["task_name"],
            metadata=checkpoint.get("metadata", {})
        )
