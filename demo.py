"""
Expert Knowledge Distillation Demo Script

This script demonstrates how to take a large, general-purpose model and distill it
into a lean, specialized expert model for a specific task. The process gradually
removes irrelevant knowledge while retaining task-specific expertise.

The demo uses the Wisconsin Breast Cancer dataset as a real-world example, showing:
1. Create a large "teacher" model with general knowledge
2. Progressively distill it into a specialized "student" model
3. Visualize the knowledge distillation process
4. Analyze the knowledge retention and model compression
5. Evaluate the specialized model's performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple, Dict, Optional
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
    create_progressive_schedule,
    plot_training_history,
    plot_model_comparison,
    plot_layer_sparsity,
    plot_distillation_summary
)


# ============================================================================
# REAL-WORLD DATASET: Wisconsin Breast Cancer
# ============================================================================

def load_breast_cancer_dataset(
    test_size: float = 0.2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Load the Wisconsin Breast Cancer dataset.
    
    This is a real-world medical dataset with 569 samples and 30 features,
    used for binary classification (malignant vs benign).
    
    Args:
        test_size: Fraction of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, test_loader, dataset_info
    """
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("sklearn not found. Installing...")
        os.system('pip install scikit-learn')
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    
    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    dataset_info = {
        "name": "Wisconsin Breast Cancer",
        "n_samples": len(data.data),
        "n_features": X.shape[1],
        "n_classes": len(data.target_names),
        "class_names": list(data.target_names),
        "feature_names": list(data.feature_names),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "class_distribution": {
            name: int((y == i).sum()) 
            for i, name in enumerate(data.target_names)
        }
    }
    
    return train_loader, test_loader, dataset_info


def create_synthetic_general_dataset(
    n_samples: int = 1000,
    n_features: int = 30,
    seed: int = 456
) -> DataLoader:
    """
    Create a synthetic general dataset for teacher pre-training.
    
    This simulates a broader dataset that the teacher might have been
    trained on, with more diverse patterns.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        seed: Random seed
        
    Returns:
        DataLoader with synthetic data
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate random data with some structure
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Create labels based on complex patterns (simulating multi-task knowledge)
    # Use multiple features to determine class
    score = np.zeros(n_samples)
    for i in range(0, n_features, 3):
        score += X[:, i] * (i % 5 - 2)  # Variable importance per feature group
    
    y = (score > np.median(score)).astype(np.int64)
    
    # Add noise to make it harder
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=32, shuffle=True)


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class LargeGeneralModel(nn.Module):
    """
    Large general-purpose model (teacher) with extensive capacity.
    This model represents a pre-trained model with broad knowledge.
    """
    def __init__(self, input_dim: int = 30, hidden_dims: List[int] = None, 
                 num_classes: int = 2, dropout: float = 0.3):
        super(LargeGeneralModel, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        
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
    def __init__(self, input_dim: int = 30, hidden_dims: List[int] = None, 
                 num_classes: int = 2, dropout: float = 0.2):
        super(SpecializedStudentModel, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]  # Much smaller than teacher
        
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

def train_teacher_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    num_epochs: int = 20, 
    device: str = 'cpu'
) -> Dict[str, List[float]]:
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
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
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return history


def evaluate_on_task(
    model: nn.Module, 
    test_loader: DataLoader, 
    device: str = 'cpu', 
    task_name: str = "Task"
) -> Dict[str, float]:
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
    
    # For additional metrics
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
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
    a specialized expert using a real-world dataset.
    """
    print("\n" + "="*80)
    print(" EXPERT KNOWLEDGE DISTILLATION DEMONSTRATION")
    print(" Using Wisconsin Breast Cancer Dataset")
    print("="*80)
    print("\nThis demo shows how to distill a large general-purpose model into")
    print("a lean, specialized expert for breast cancer classification.")
    print("="*80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create output directory for plots
    output_dir = "distillation_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}/")
    
    # ========================================================================
    # STEP 1: LOAD REAL-WORLD DATASET
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 1: Loading Wisconsin Breast Cancer Dataset")
    print("-"*80)
    
    task_train_loader, task_test_loader, dataset_info = load_breast_cancer_dataset(
        test_size=0.2, seed=42
    )
    
    # General dataset for teacher pre-training
    general_train_loader = create_synthetic_general_dataset(
        n_samples=1000, n_features=30, seed=456
    )
    
    print(f"\n📊 Dataset: {dataset_info['name']}")
    print(f"  Total samples: {dataset_info['n_samples']}")
    print(f"  Features: {dataset_info['n_features']}")
    print(f"  Classes: {dataset_info['class_names']}")
    print(f"  Training samples: {dataset_info['train_samples']}")
    print(f"  Test samples: {dataset_info['test_samples']}")
    print(f"  Class distribution: {dataset_info['class_distribution']}")
    
    # ========================================================================
    # STEP 2: CREATE AND TRAIN TEACHER MODEL
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 2: Creating and Training Teacher Model")
    print("-"*80)
    
    teacher_model = LargeGeneralModel(
        input_dim=30,
        hidden_dims=[256, 128, 64, 32],
        num_classes=2,
        dropout=0.3
    )
    
    print(f"\nTeacher Model Architecture:")
    print(f"  Total parameters: {count_parameters(teacher_model):,}")
    print(f"  Trainable parameters: {count_parameters(teacher_model, trainable_only=True):,}")
    print(f"  Model size: {get_model_size_mb(teacher_model):.4f} MB")
    
    # Train teacher on general data (simulating pre-training)
    teacher_history = train_teacher_model(
        teacher_model, 
        general_train_loader, 
        num_epochs=20,
        device=device
    )
    
    # Fine-tune teacher on task-specific data
    print("\n" + "-"*60)
    print("Fine-tuning teacher on task-specific data...")
    print("-"*60)
    teacher_history_task = train_teacher_model(
        teacher_model,
        task_train_loader,
        num_epochs=15,
        device=device
    )
    
    # Evaluate teacher on task-specific data
    print("\n" + "-"*80)
    print("Teacher Model Performance on Breast Cancer Classification:")
    print("-"*80)
    teacher_task_metrics = evaluate_on_task(
        teacher_model, 
        task_test_loader, 
        device=device,
        task_name="Teacher on Test Data"
    )
    
    # ========================================================================
    # STEP 3: CREATE STUDENT MODEL
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 3: Creating Student Model")
    print("-"*80)
    
    student_model = SpecializedStudentModel(
        input_dim=30,
        hidden_dims=[64, 32],
        num_classes=2,
        dropout=0.2
    )
    
    print(f"\nStudent Model Architecture:")
    print(f"  Total parameters: {count_parameters(student_model):,}")
    print(f"  Trainable parameters: {count_parameters(student_model, trainable_only=True):,}")
    print(f"  Model size: {get_model_size_mb(student_model):.4f} MB")
    
    # Compare models
    print("\n" + "-"*80)
    print("Model Comparison:")
    print("-"*80)
    comparison = compare_models(teacher_model, student_model, "Teacher", "Student")
    print(f"  Teacher parameters: {comparison['Teacher']['parameters']:,}")
    print(f"  Student parameters: {comparison['Student']['parameters']:,}")
    print(f"  Parameter reduction: {comparison['parameter_reduction']:.2f}%")
    print(f"  Size reduction: {comparison['size_reduction']:.2f}%")
    
    # Plot model comparison
    plot_model_comparison(
        comparison, 
        save_path=os.path.join(output_dir, "model_comparison.png")
    )
    print(f"  ✓ Saved model comparison plot")
    
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
        num_epochs=25,
        pruning_start_epoch=8,
        pruning_end_epoch=20,
        target_sparsity=0.4,
        importance_threshold=0.1,
        prune_every=3
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
    print("  2. Gradually remove irrelevant knowledge (pruning)")
    print("  3. Specialize the student for breast cancer classification")
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
        val_loader=task_test_loader,
        verbose=True
    )
    
    # Plot training history
    plot_training_history(
        distillation_history,
        save_path=os.path.join(output_dir, "training_history.png")
    )
    print(f"\n✓ Saved training history plot")
    
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
    
    # Plot layer sparsity
    plot_layer_sparsity(
        specialized_student,
        save_path=os.path.join(output_dir, "layer_sparsity.png")
    )
    print(f"  ✓ Saved layer sparsity plot")
    
    # Evaluate specialized student
    print("\n" + "-"*80)
    print("Student Model Performance on Breast Cancer Classification:")
    print("-"*80)
    student_task_metrics = evaluate_on_task(
        specialized_student,
        task_test_loader,
        device=device,
        task_name="Specialized Student on Test Data"
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
        print(f"    Sparsity: {sparsity_stats['layer_sparsity'].get(layer_name, 0)*100:.1f}%")
    
    # ========================================================================
    # STEP 7: CREATE SPECIALIZED EXPERT
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 7: Creating Specialized Expert")
    print("-"*80)
    
    expert = SpecializedExpert(
        model=specialized_student,
        task_name="Breast Cancer Classification",
        config=config,
        metadata={
            "dataset": dataset_info['name'],
            "accuracy": student_task_metrics['accuracy'],
            "sparsity": sparsity_stats['overall_sparsity']
        }
    )
    
    # Test expert predictions
    print("\nTesting Expert Predictions:")
    test_samples, test_labels = next(iter(task_test_loader))
    test_samples = test_samples[:5].to(device)  # Take 5 samples
    test_labels = test_labels[:5]
    
    predictions = expert.predict(test_samples)
    
    class_names = dataset_info['class_names']
    print("\nSample Predictions:")
    for i, (pred, true_label) in enumerate(zip(predictions, test_labels)):
        pred_class = pred['predicted_class']
        confidence = pred['confidence']
        pred_name = class_names[pred_class]
        true_name = class_names[true_label.item()]
        correct = "✓" if pred_class == true_label.item() else "✗"
        print(f"  Sample {i+1}: Predicted={pred_name}, True={true_name}, "
              f"Confidence={confidence:.1f}% {correct}")
    
    # ========================================================================
    # STEP 8: SAVE RESULTS
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 8: Saving Results")
    print("-"*80)
    
    # Save expert model
    expert_path = os.path.join(output_dir, "specialized_cancer_expert.pt")
    expert.save(expert_path)
    print(f"✓ Saved specialized expert to: {expert_path}")
    
    # Save distillation report
    report_path = os.path.join(output_dir, "distillation_report.json")
    save_distillation_report(
        report_path=report_path,
        teacher_model=teacher_model,
        student_model=specialized_student,
        training_history=distillation_history,
        metadata={
            'task': 'Breast Cancer Classification',
            'dataset': dataset_info['name'],
            'teacher_accuracy': teacher_task_metrics['accuracy'],
            'student_accuracy': student_task_metrics['accuracy'],
            'compression_ratio': comparison['parameter_reduction'],
            'final_sparsity': sparsity_stats['overall_sparsity']
        }
    )
    print(f"✓ Saved distillation report to: {report_path}")
    
    # Plot summary dashboard
    plot_distillation_summary(
        teacher_acc=teacher_task_metrics['accuracy'],
        student_acc=student_task_metrics['accuracy'],
        comparison=comparison,
        sparsity=sparsity_stats['overall_sparsity'],
        save_path=os.path.join(output_dir, "distillation_summary.png")
    )
    print(f"✓ Saved summary dashboard")
    
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
    print(f"  Teacher: {get_model_size_mb(teacher_model):.4f} MB")
    print(f"  Student: {get_model_size_mb(specialized_student):.4f} MB")
    print(f"  Reduction: {comparison['size_reduction']:.2f}%")
    
    print(f"\n📁 Output Files:")
    print(f"  • {output_dir}/model_comparison.png")
    print(f"  • {output_dir}/training_history.png")
    print(f"  • {output_dir}/layer_sparsity.png")
    print(f"  • {output_dir}/distillation_summary.png")
    print(f"  • {output_dir}/specialized_cancer_expert.pt")
    print(f"  • {output_dir}/distillation_report.json")
    
    print(f"\n✨ Specialization Benefits:")
    print(f"  ✓ {comparison['parameter_reduction']:.0f}% smaller model size")
    print(f"  ✓ Faster inference (fewer parameters)")
    print(f"  ✓ Task-specific expertise (focused knowledge)")
    print(f"  ✓ Easier deployment on resource-constrained devices")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nThe specialized expert model is now ready for deployment.")
    print("It has been optimized specifically for breast cancer classification,")
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
        'history': distillation_history,
        'dataset_info': dataset_info
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
