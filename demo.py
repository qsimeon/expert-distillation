"""
Expert Knowledge Distillation Demo Script

This script demonstrates how to take a large, general-purpose model and distill it
into a lean, specialized expert model for a specific task. The process gradually
removes irrelevant knowledge while retaining task-specific expertise.

The demo uses a sentiment analysis task as an example, showing how to:
1. Create a large "teacher" model with general knowledge
2. Progressively distill it into a specialized "student" model
3. Analyze the knowledge retention and model compression
4. Evaluate the specialized model's performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
import sys
import os

# Add lib directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# Import from our custom modules
from core import DistillationConfig, KnowledgeDistiller, SpecializedExpert
from utils import (
    count_parameters, 
    get_model_size_mb, 
    compute_sparsity,
    compare_models,
    analyze_layer_importance,
    create_knowledge_mask,
    apply_knowledge_mask,
    compute_knowledge_retention,
    evaluate_model,
    get_layer_statistics,
    save_distillation_report,
    compute_task_alignment,
    create_progressive_schedule
)


# ============================================================================
# SYNTHETIC DATASET FOR DEMONSTRATION
# ============================================================================

class SentimentDataset(Dataset):
    """
    Synthetic sentiment analysis dataset.
    Simulates text embeddings and sentiment labels (positive/negative).
    """
    def __init__(self, num_samples: int = 1000, embedding_dim: int = 128, 
                 task_specific: bool = True, seed: int = 42):
        """
        Args:
            num_samples: Number of samples to generate
            embedding_dim: Dimension of text embeddings
            task_specific: If True, generates task-specific patterns
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim
        
        # Generate synthetic embeddings
        if task_specific:
            # Task-specific data has clear patterns
            # Positive sentiment: higher values in first half of embedding
            # Negative sentiment: higher values in second half
            positive_samples = num_samples // 2
            negative_samples = num_samples - positive_samples
            
            # Positive embeddings
            pos_embeddings = np.random.randn(positive_samples, embedding_dim).astype(np.float32)
            pos_embeddings[:, :embedding_dim//2] += 1.5  # Boost first half
            
            # Negative embeddings
            neg_embeddings = np.random.randn(negative_samples, embedding_dim).astype(np.float32)
            neg_embeddings[:, embedding_dim//2:] += 1.5  # Boost second half
            
            self.embeddings = torch.from_numpy(
                np.vstack([pos_embeddings, neg_embeddings])
            )
            self.labels = torch.cat([
                torch.ones(positive_samples, dtype=torch.long),
                torch.zeros(negative_samples, dtype=torch.long)
            ])
        else:
            # General data has random patterns
            self.embeddings = torch.randn(num_samples, embedding_dim)
            self.labels = torch.randint(0, 2, (num_samples,), dtype=torch.long)
        
        # Shuffle
        indices = torch.randperm(num_samples)
        self.embeddings = self.embeddings[indices]
        self.labels = self.labels[indices]
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class LargeGeneralModel(nn.Module):
    """
    Large general-purpose model (teacher) with extensive capacity.
    This model is trained on multiple tasks and has broad knowledge.
    """
    def __init__(self, input_dim: int = 128, hidden_dims: List[int] = None, 
                 num_classes: int = 2, dropout: float = 0.3):
        super(LargeGeneralModel, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SpecializedStudentModel(nn.Module):
    """
    Smaller specialized model (student) that will learn task-specific knowledge.
    This model has reduced capacity but will be highly efficient for the target task.
    """
    def __init__(self, input_dim: int = 128, hidden_dims: List[int] = None, 
                 num_classes: int = 2, dropout: float = 0.2):
        super(SpecializedStudentModel, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]  # Much smaller than teacher
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_teacher_model(model: nn.Module, train_loader: DataLoader, 
                        num_epochs: int = 10, device: str = 'cpu') -> Dict[str, List[float]]:
    """
    Train the large teacher model on general data.
    
    Args:
        model: Teacher model to train
        train_loader: Training data loader
        num_epochs: Number of training epochs
        device: Device to train on
    
    Returns:
        Dictionary containing training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'loss': [], 'accuracy': []}
    
    print(f"\n{'='*60}")
    print("TRAINING LARGE GENERAL MODEL (TEACHER)")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return history


def evaluate_on_task(model: nn.Module, test_loader: DataLoader, 
                     device: str = 'cpu', task_name: str = "Task") -> Dict[str, float]:
    """
    Evaluate model on a specific task.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
        task_name: Name of the task for display
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    print(f"\n{task_name} Evaluation:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    return {'loss': avg_loss, 'accuracy': accuracy}


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """
    Main demonstration of expert knowledge distillation.
    Shows the complete pipeline from training a large model to creating
    a specialized expert.
    """
    print("\n" + "="*80)
    print(" EXPERT KNOWLEDGE DISTILLATION DEMONSTRATION")
    print("="*80)
    print("\nThis demo shows how to distill a large general-purpose model into")
    print("a lean, specialized expert for sentiment analysis.")
    print("="*80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # ========================================================================
    # STEP 1: CREATE DATASETS
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 1: Creating Datasets")
    print("-"*80)
    
    # Task-specific dataset (sentiment analysis)
    task_train_dataset = SentimentDataset(num_samples=1000, task_specific=True, seed=42)
    task_test_dataset = SentimentDataset(num_samples=200, task_specific=True, seed=123)
    
    # General dataset (for teacher pre-training)
    general_train_dataset = SentimentDataset(num_samples=1000, task_specific=False, seed=456)
    
    # Create data loaders
    task_train_loader = DataLoader(task_train_dataset, batch_size=32, shuffle=True)
    task_test_loader = DataLoader(task_test_dataset, batch_size=32, shuffle=False)
    general_train_loader = DataLoader(general_train_dataset, batch_size=32, shuffle=True)
    
    print(f"✓ Task-specific training samples: {len(task_train_dataset)}")
    print(f"✓ Task-specific test samples: {len(task_test_dataset)}")
    print(f"✓ General training samples: {len(general_train_dataset)}")
    
    # ========================================================================
    # STEP 2: CREATE AND TRAIN TEACHER MODEL
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 2: Creating and Training Teacher Model")
    print("-"*80)
    
    teacher_model = LargeGeneralModel(
        input_dim=128,
        hidden_dims=[512, 256, 128, 64],
        num_classes=2,
        dropout=0.3
    )
    
    print(f"\nTeacher Model Architecture:")
    print(f"  Total parameters: {count_parameters(teacher_model):,}")
    print(f"  Trainable parameters: {count_parameters(teacher_model, trainable_only=True):,}")
    print(f"  Model size: {get_model_size_mb(teacher_model):.2f} MB")
    
    # Train teacher on general data
    teacher_history = train_teacher_model(
        teacher_model, 
        general_train_loader, 
        num_epochs=15,
        device=device
    )
    
    # Evaluate teacher on task-specific data
    print("\n" + "-"*80)
    print("Teacher Model Performance on Sentiment Analysis Task:")
    print("-"*80)
    teacher_task_metrics = evaluate_on_task(
        teacher_model, 
        task_test_loader, 
        device=device,
        task_name="Teacher on Task-Specific Data"
    )
    
    # ========================================================================
    # STEP 3: CREATE STUDENT MODEL
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 3: Creating Student Model")
    print("-"*80)
    
    student_model = SpecializedStudentModel(
        input_dim=128,
        hidden_dims=[256, 128],
        num_classes=2,
        dropout=0.2
    )
    
    print(f"\nStudent Model Architecture:")
    print(f"  Total parameters: {count_parameters(student_model):,}")
    print(f"  Trainable parameters: {count_parameters(student_model, trainable_only=True):,}")
    print(f"  Model size: {get_model_size_mb(student_model):.2f} MB")
    
    # Compare models
    print("\n" + "-"*80)
    print("Model Comparison:")
    print("-"*80)
    comparison = compare_models(teacher_model, student_model, "Teacher", "Student")
    print(f"  Parameter reduction: {comparison['parameter_reduction']:.2f}%")
    print(f"  Size reduction: {comparison['size_reduction']:.2f}%")
    
    # ========================================================================
    # STEP 4: CONFIGURE DISTILLATION
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 4: Configuring Knowledge Distillation")
    print("-"*80)
    
    config = DistillationConfig(
        temperature=4.0,
        alpha_start=0.7,
        alpha_end=0.3,
        learning_rate=0.001,
        num_epochs=20,
        pruning_start_epoch=5,
        pruning_end_epoch=15,
        target_sparsity=0.5,
        importance_threshold=0.1
    )
    
    print(f"\nDistillation Configuration:")
    print(f"  Temperature: {config.temperature}")
    print(f"  Alpha (KD weight) schedule: {config.alpha_start} → {config.alpha_end}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Total epochs: {config.num_epochs}")
    print(f"  Pruning epochs: {config.pruning_start_epoch} → {config.pruning_end_epoch}")
    print(f"  Target sparsity: {config.target_sparsity * 100}%")
    
    # ========================================================================
    # STEP 5: PERFORM KNOWLEDGE DISTILLATION
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 5: Performing Knowledge Distillation")
    print("-"*80)
    print("\nThis process will:")
    print("  1. Transfer knowledge from teacher to student")
    print("  2. Gradually remove irrelevant knowledge")
    print("  3. Specialize the student for sentiment analysis")
    print("-"*80)
    
    distiller = KnowledgeDistiller(
        teacher_model=teacher_model,
        student_model=student_model,
        config=config,
        device=device
    )
    
    # Perform distillation
    distillation_history = distiller.distill(
        train_loader=task_train_loader,
        val_loader=task_test_loader
    )
    
    # ========================================================================
    # STEP 6: ANALYZE RESULTS
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 6: Analyzing Distillation Results")
    print("-"*80)
    
    # Get final student model
    specialized_student = distiller.student_model
    
    # Compute sparsity
    sparsity_stats = compute_sparsity(specialized_student)
    print(f"\nStudent Model Sparsity:")
    print(f"  Overall sparsity: {sparsity_stats['overall_sparsity']*100:.2f}%")
    print(f"  Sparse layers: {sparsity_stats['num_sparse_layers']}/{sparsity_stats['total_layers']}")
    
    # Evaluate specialized student
    print("\n" + "-"*80)
    print("Student Model Performance on Sentiment Analysis Task:")
    print("-"*80)
    student_task_metrics = evaluate_on_task(
        specialized_student,
        task_test_loader,
        device=device,
        task_name="Specialized Student on Task-Specific Data"
    )
    
    # Get layer statistics
    print("\n" + "-"*80)
    print("Layer Statistics (Student Model):")
    print("-"*80)
    layer_stats = get_layer_statistics(specialized_student)
    for layer_name, stats in list(layer_stats.items())[:3]:  # Show first 3 layers
        print(f"\n  {layer_name}:")
        print(f"    Mean: {stats['mean']:.6f}")
        print(f"    Std: {stats['std']:.6f}")
        print(f"    Min: {stats['min']:.6f}")
        print(f"    Max: {stats['max']:.6f}")
    
    # ========================================================================
    # STEP 7: CREATE SPECIALIZED EXPERT
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 7: Creating Specialized Expert")
    print("-"*80)
    
    expert = SpecializedExpert(
        model=specialized_student,
        task_name="Sentiment Analysis",
        config=config
    )
    
    # Test expert predictions
    print("\nTesting Expert Predictions:")
    test_samples, test_labels = next(iter(task_test_loader))
    test_samples = test_samples[:5].to(device)  # Take 5 samples
    test_labels = test_labels[:5]
    
    predictions = expert.predict(test_samples)
    
    print("\nSample Predictions:")
    for i, (pred, true_label) in enumerate(zip(predictions, test_labels)):
        pred_class = pred['predicted_class']
        confidence = pred['confidence']
        correct = "✓" if pred_class == true_label.item() else "✗"
        print(f"  Sample {i+1}: Predicted={pred_class}, True={true_label.item()}, "
              f"Confidence={confidence:.2f}% {correct}")
    
    # ========================================================================
    # STEP 8: SAVE RESULTS
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 8: Saving Results")
    print("-"*80)
    
    # Save expert model
    expert_path = "specialized_sentiment_expert.pt"
    expert.save(expert_path)
    print(f"✓ Saved specialized expert to: {expert_path}")
    
    # Save distillation report
    report_path = "distillation_report.json"
    save_distillation_report(
        report_path=report_path,
        teacher_model=teacher_model,
        student_model=specialized_student,
        training_history=distillation_history,
        metadata={
            'task': 'Sentiment Analysis',
            'teacher_accuracy': teacher_task_metrics['accuracy'],
            'student_accuracy': student_task_metrics['accuracy'],
            'compression_ratio': comparison['parameter_reduction'],
            'final_sparsity': sparsity_stats['overall_sparsity']
        }
    )
    print(f"✓ Saved distillation report to: {report_path}")
    
    # ========================================================================
    # STEP 9: FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("DISTILLATION SUMMARY")
    print("="*80)
    
    print(f"\n📊 Model Compression:")
    print(f"  Teacher parameters: {count_parameters(teacher_model):,}")
    print(f"  Student parameters: {count_parameters(specialized_student):,}")
    print(f"  Reduction: {comparison['parameter_reduction']:.2f}%")
    print(f"  Sparsity: {sparsity_stats['overall_sparsity']*100:.2f}%")
    
    print(f"\n🎯 Performance:")
    print(f"  Teacher accuracy: {teacher_task_metrics['accuracy']:.2f}%")
    print(f"  Student accuracy: {student_task_metrics['accuracy']:.2f}%")
    
    accuracy_diff = student_task_metrics['accuracy'] - teacher_task_metrics['accuracy']
    if accuracy_diff >= 0:
        print(f"  Performance gain: +{accuracy_diff:.2f}%")
    else:
        print(f"  Performance loss: {accuracy_diff:.2f}%")
    
    print(f"\n💾 Model Size:")
    print(f"  Teacher: {get_model_size_mb(teacher_model):.2f} MB")
    print(f"  Student: {get_model_size_mb(specialized_student):.2f} MB")
    print(f"  Reduction: {comparison['size_reduction']:.2f}%")
    
    print(f"\n✨ Specialization Benefits:")
    print(f"  ✓ Smaller model size (easier deployment)")
    print(f"  ✓ Faster inference (fewer parameters)")
    print(f"  ✓ Task-specific expertise (focused knowledge)")
    print(f"  ✓ Reduced memory footprint")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nThe specialized expert model is now ready for deployment.")
    print("It has been optimized specifically for sentiment analysis,")
    print("with irrelevant knowledge removed and task-specific knowledge retained.")
    print("="*80 + "\n")
    
    return {
        'teacher_model': teacher_model,
        'student_model': specialized_student,
        'expert': expert,
        'teacher_metrics': teacher_task_metrics,
        'student_metrics': student_task_metrics,
        'comparison': comparison,
        'sparsity': sparsity_stats,
        'history': distillation_history
    }


if __name__ == "__main__":
    try:
        results = main()
        print("✓ Demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
