"""Core module for expert knowledge distillation.

This module provides the main functionality for distilling knowledge from a large
teacher model into a smaller, specialized student model. It implements progressive
knowledge removal and task-specific optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation process.
    
    Attributes:
        temperature: Temperature for softening probability distributions
        alpha_start: Initial weight for distillation loss (vs task loss)
        alpha_end: Final weight for distillation loss
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        pruning_start_epoch: Epoch to start pruning
        pruning_end_epoch: Epoch to end pruning
        target_sparsity: Target sparsity level (fraction of weights to prune)
        importance_threshold: Threshold for importance-based pruning
        prune_every: Prune every N epochs
    """
    temperature: float = 4.0
    alpha_start: float = 0.7
    alpha_end: float = 0.3
    learning_rate: float = 0.001
    num_epochs: int = 20
    pruning_start_epoch: int = 5
    pruning_end_epoch: int = 15
    target_sparsity: float = 0.5
    importance_threshold: float = 0.1
    prune_every: int = 3


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
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.config = config
        self.device = device
        self.teacher_model.eval()  # Teacher is always in eval mode
        
        # Track importance scores for knowledge components
        self.knowledge_importance: Dict[str, torch.Tensor] = {}
        self.iteration = 0
        self.current_epoch = 0
        
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
    
    def get_alpha(self, epoch: int) -> float:
        """Get the alpha value (distillation weight) for current epoch.
        
        Linearly interpolates from alpha_start to alpha_end over training.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Alpha value for balancing distillation and task loss
        """
        progress = min(1.0, epoch / max(1, self.config.num_epochs - 1))
        alpha = self.config.alpha_start + (self.config.alpha_end - self.config.alpha_start) * progress
        return alpha
    
    def get_pruning_rate(self, epoch: int) -> float:
        """Get the pruning rate for current epoch.
        
        Gradually increases pruning during the pruning phase.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Pruning rate (fraction of weights to prune)
        """
        if epoch < self.config.pruning_start_epoch:
            return 0.0
        if epoch > self.config.pruning_end_epoch:
            return self.config.target_sparsity
            
        pruning_progress = (epoch - self.config.pruning_start_epoch) / (
            self.config.pruning_end_epoch - self.config.pruning_start_epoch
        )
        return self.config.target_sparsity * pruning_progress
    
    def compute_knowledge_importance(
        self,
        dataloader: DataLoader,
        num_batches: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute importance scores for different knowledge components.
        
        Uses gradient-based importance estimation to determine which parts of
        the model are most relevant for the target task.
        
        Args:
            dataloader: DataLoader with task-specific data
            num_batches: Number of batches to use (None for all)
            
        Returns:
            Dictionary mapping layer names to importance scores
        """
        self.student_model.eval()
        importance_scores = {}
        
        # Initialize importance accumulators
        for name, param in self.student_model.named_parameters():
            if param.requires_grad:
                importance_scores[name] = torch.zeros_like(param.data)
        
        batches_processed = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if num_batches and batch_idx >= num_batches:
                break
                
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.student_model.zero_grad()
            outputs = self.student_model(inputs)
            
            # Compute loss and gradients
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            # Accumulate squared gradients as importance measure
            for name, param in self.student_model.named_parameters():
                if param.grad is not None:
                    importance_scores[name] += param.grad.data.abs()
            
            batches_processed += 1
        
        # Normalize importance scores
        if batches_processed > 0:
            for name in importance_scores:
                importance_scores[name] /= batches_processed
            
        self.knowledge_importance = importance_scores
        return importance_scores
    
    def prune_irrelevant_knowledge(
        self,
        pruning_rate: float
    ) -> int:
        """Prune weights with low importance scores.
        
        Args:
            pruning_rate: Fraction of weights to prune
            
        Returns:
            Number of parameters pruned
        """
        if pruning_rate <= 0:
            return 0
            
        if not self.knowledge_importance:
            logger.warning("No importance scores computed. Skipping pruning.")
            return 0
        
        total_pruned = 0
        
        for name, param in self.student_model.named_parameters():
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
        
        return total_pruned
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        alpha: float
    ) -> Dict[str, float]:
        """Perform a single training step with distillation.
        
        Args:
            inputs: Input batch
            targets: Target labels
            optimizer: Optimizer for student model
            alpha: Weight for distillation loss
            
        Returns:
            Dictionary with loss components
        """
        self.student_model.train()
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Get predictions from both models
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
        
        student_logits = self.student_model(inputs)
        
        # Compute losses
        task_loss = F.cross_entropy(student_logits, targets)
        distill_loss = self.distillation_loss(
            student_logits,
            teacher_logits,
            self.config.temperature
        )
        
        # Combine losses
        total_loss = alpha * distill_loss + (1 - alpha) * task_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = student_logits.max(1)
        correct = predicted.eq(targets).sum().item()
        accuracy = correct / targets.size(0)
        
        return {
            "total_loss": total_loss.item(),
            "distill_loss": distill_loss.item(),
            "task_loss": task_loss.item(),
            "alpha": alpha,
            "accuracy": accuracy
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the student model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.student_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.student_model(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                total_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        
        return {
            "loss": total_loss / total if total > 0 else 0.0,
            "accuracy": correct / total if total > 0 else 0.0
        }
    
    def distill(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Run the complete distillation process.
        
        Args:
            train_loader: DataLoader for task-specific training data
            val_loader: Optional validation data loader
            verbose: Whether to print progress
            
        Returns:
            Training history with metrics
        """
        optimizer = torch.optim.Adam(
            self.student_model.parameters(), 
            lr=self.config.learning_rate
        )
        
        history = {
            "total_loss": [],
            "distill_loss": [],
            "task_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "val_loss": [],
            "alpha": [],
            "pruned_params": [],
            "sparsity": []
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print("KNOWLEDGE DISTILLATION TRAINING")
            print(f"{'='*60}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Get current alpha
            alpha = self.get_alpha(epoch)
            
            # Training
            epoch_losses = {"total_loss": 0, "distill_loss": 0, "task_loss": 0, "accuracy": 0}
            num_batches = 0
            
            self.student_model.train()
            for inputs, targets in train_loader:
                losses = self.train_step(inputs, targets, optimizer, alpha)
                
                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key] += losses[key]
                num_batches += 1
                self.iteration += 1
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= max(1, num_batches)
            
            history["total_loss"].append(epoch_losses["total_loss"])
            history["distill_loss"].append(epoch_losses["distill_loss"])
            history["task_loss"].append(epoch_losses["task_loss"])
            history["train_accuracy"].append(epoch_losses["accuracy"])
            history["alpha"].append(alpha)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_accuracy"].append(val_metrics["accuracy"])
                history["val_loss"].append(val_metrics["loss"])
            
            # Periodic pruning
            pruned = 0
            if epoch >= self.config.pruning_start_epoch and epoch <= self.config.pruning_end_epoch:
                if (epoch - self.config.pruning_start_epoch) % self.config.prune_every == 0:
                    self.compute_knowledge_importance(train_loader, num_batches=10)
                    pruning_rate = self.get_pruning_rate(epoch)
                    pruned = self.prune_irrelevant_knowledge(pruning_rate)
            
            history["pruned_params"].append(pruned)
            
            # Compute current sparsity
            from utils import compute_sparsity
            sparsity_stats = compute_sparsity(self.student_model)
            history["sparsity"].append(sparsity_stats["overall_sparsity"])
            
            if verbose:
                val_acc_str = f", Val Acc: {history['val_accuracy'][-1]*100:.2f}%" if val_loader else ""
                prune_str = f", Pruned: {pruned:,}" if pruned > 0 else ""
                print(f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                      f"Loss: {epoch_losses['total_loss']:.4f}, "
                      f"Train Acc: {epoch_losses['accuracy']*100:.2f}%"
                      f"{val_acc_str}{prune_str}, α: {alpha:.3f}")
        
        if verbose:
            print(f"{'='*60}")
            print("DISTILLATION COMPLETE!")
            print(f"{'='*60}")
        
        return history


class SpecializedExpert:
    """Wrapper for a specialized expert model after distillation."""
    
    def __init__(
        self, 
        model: nn.Module, 
        task_name: str, 
        config: Optional[DistillationConfig] = None,
        metadata: Optional[Dict] = None
    ):
        """Initialize specialized expert.
        
        Args:
            model: The distilled specialized model
            task_name: Name of the specialized task
            config: Distillation config used (optional)
            metadata: Additional metadata about the expert
        """
        self.model = model
        self.task_name = task_name
        self.config = config
        self.metadata = metadata or {}
        
    def predict(self, inputs: torch.Tensor) -> List[Dict[str, Any]]:
        """Make predictions using the specialized model.
        
        Args:
            inputs: Input tensor
            
        Returns:
            List of prediction dictionaries with class and confidence
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            probabilities = F.softmax(outputs, dim=-1)
            confidences, predictions = probabilities.max(dim=-1)
        
        results = []
        for pred, conf in zip(predictions, confidences):
            results.append({
                "predicted_class": pred.item(),
                "confidence": conf.item() * 100,
                "probabilities": probabilities[len(results)].cpu().tolist()
            })
        
        return results
    
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
    def load(cls, path: str, model: nn.Module) -> "SpecializedExpert":
        """Load a specialized expert model.
        
        Args:
            path: Path to load the model from
            model: Model instance to load weights into
            
        Returns:
            Loaded SpecializedExpert instance
        """
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return cls(
            model=model,
            task_name=checkpoint["task_name"],
            metadata=checkpoint.get("metadata", {})
        )
