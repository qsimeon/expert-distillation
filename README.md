# Expert Knowledge Distillation Toolkit

> Transform large generalist models into lean, specialized experts through targeted knowledge distillation

A Python library for distilling large pre-trained models into specialized, task-specific versions by systematically removing irrelevant knowledge while preserving essential capabilities. This toolkit enables researchers and practitioners to create efficient, focused models that excel at specific tasks without the overhead of general-purpose knowledge.

## ✨ Features

- **Staged Knowledge Distillation Pipeline** — Multi-stage distillation process that progressively prunes non-essential parameters, fine-tunes on task-specific data, and applies alignment losses to create specialized models while maintaining performance on target tasks.
- **Modular Teacher-Student Architecture** — Flexible wrapper system for teacher (large model) and student (specialized model) that supports various model architectures and distillation strategies with minimal code changes.
- **Comprehensive Evaluation Harness** — Built-in evaluation framework with task-specific benchmarks, error analysis, and interpretable metrics including accuracy, calibration, and robustness to guide pruning decisions.
- **Configurable Pruning Strategies** — Multiple pruning approaches including magnitude-based, structured, and gradient-based pruning to selectively remove knowledge while preserving task-critical parameters.
- **Experiment Tracking and Reproducibility** — Complete experiment management with seed control, checkpoint versioning, artifact tracking, and configuration logging to ensure reproducible distillation workflows.
- **Task-Specific Fine-Tuning** — Targeted fine-tuning mechanisms that reinforce essential capabilities on domain-specific datasets while actively forgetting irrelevant knowledge from the teacher model.

## 📦 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12 or higher
- CUDA-capable GPU (recommended for large models)
- 8GB+ RAM (16GB+ recommended)

### Setup

1. git clone https://github.com/yourusername/expert-knowledge-distillation.git
   - Clone the repository to your local machine
2. cd expert-knowledge-distillation
   - Navigate to the project directory
3. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   - Install PyTorch with CUDA support (adjust CUDA version as needed)
4. pip install numpy scipy scikit-learn
   - Install core scientific computing dependencies
5. pip install -e .
   - Install the package in editable mode for development
6. python -c "import torch; print(torch.cuda.is_available())"
   - Verify CUDA availability (should print True if GPU is available)

## 🚀 Usage

### Basic Knowledge Distillation

Distill a large teacher model into a smaller student model specialized for sentiment analysis

```
from lib.core import KnowledgeDistiller, DistillationConfig
from lib.utils import load_model, prepare_dataset

# Configure distillation
config = DistillationConfig(
    task_name="sentiment_analysis",
    pruning_ratio=0.7,
    num_epochs=10,
    learning_rate=1e-4,
    temperature=2.0
)

# Initialize teacher and student models
teacher = load_model("large-pretrained-model")
student = load_model("small-base-model")

# Prepare task-specific dataset
train_data, eval_data = prepare_dataset("sentiment_dataset")

# Create distiller and run
distiller = KnowledgeDistiller(teacher, student, config)
distiller.distill(train_data, eval_data)

# Save specialized model
distiller.save_student("specialized_sentiment_model.pt")
print("Distillation complete!")
```

**Output:**

```
Epoch 1/10: Loss=0.4523, Accuracy=0.8234
Epoch 2/10: Loss=0.3891, Accuracy=0.8567
...
Epoch 10/10: Loss=0.2145, Accuracy=0.9123
Distillation complete!
Model saved to specialized_sentiment_model.pt
```

### Custom Pruning Strategy

Apply structured pruning with custom retention criteria to preserve specific model capabilities

```
from lib.core import KnowledgeDistiller, DistillationConfig
from lib.utils import StructuredPruner, RetentionCriteria

# Define what knowledge to retain
retention = RetentionCriteria(
    preserve_layers=["attention", "task_head"],
    importance_threshold=0.3,
    gradient_based=True
)

# Configure pruner
pruner = StructuredPruner(
    strategy="magnitude",
    retention_criteria=retention,
    pruning_schedule="gradual"
)

config = DistillationConfig(
    task_name="named_entity_recognition",
    pruner=pruner,
    num_epochs=15
)

teacher = load_model("bert-large")
student = load_model("bert-base")

distiller = KnowledgeDistiller(teacher, student, config)
results = distiller.distill(train_data, eval_data)

print(f"Parameters reduced: {results['compression_ratio']:.2%}")
print(f"Task accuracy: {results['final_accuracy']:.4f}")
```

**Output:**

```
Pruning stage 1/3: 30% parameters removed
Pruning stage 2/3: 55% parameters removed
Pruning stage 3/3: 70% parameters removed
Fine-tuning specialized model...
Parameters reduced: 70.00%
Task accuracy: 0.9234
```

### Evaluation and Metrics

Evaluate the distilled model with comprehensive metrics and compare against the teacher

```
from lib.core import KnowledgeDistiller
from lib.utils import EvaluationHarness, MetricSuite

# Load distilled model
student = load_model("specialized_sentiment_model.pt")
teacher = load_model("large-pretrained-model")

# Create evaluation harness
evaluator = EvaluationHarness(
    metrics=MetricSuite(["accuracy", "f1", "calibration", "inference_time"]),
    test_data=eval_data
)

# Compare teacher vs student
teacher_results = evaluator.evaluate(teacher)
student_results = evaluator.evaluate(student)

print("Performance Comparison:")
print(f"Teacher Accuracy: {teacher_results['accuracy']:.4f}")
print(f"Student Accuracy: {student_results['accuracy']:.4f}")
print(f"Speedup: {teacher_results['inference_time'] / student_results['inference_time']:.2f}x")
print(f"Model Size Reduction: {(1 - student.size() / teacher.size()):.2%}")
```

**Output:**

```
Performance Comparison:
Teacher Accuracy: 0.9245
Student Accuracy: 0.9123
Speedup: 4.32x
Model Size Reduction: 68.50%
```

### End-to-End Demo with Reproducibility

Run a complete distillation experiment with full reproducibility controls

```
from lib.core import KnowledgeDistiller, DistillationConfig
from lib.utils import set_seed, ExperimentTracker

# Set seed for reproducibility
set_seed(42)

# Initialize experiment tracker
tracker = ExperimentTracker(
    experiment_name="sentiment_distillation_v1",
    log_dir="./experiments",
    save_checkpoints=True
)

config = DistillationConfig(
    task_name="sentiment_analysis",
    pruning_ratio=0.65,
    num_epochs=12,
    seed=42,
    checkpoint_interval=3
)

with tracker:
    distiller = KnowledgeDistiller(teacher, student, config)
    distiller.set_tracker(tracker)
    results = distiller.distill(train_data, eval_data)
    
    tracker.log_metrics(results)
    tracker.save_artifacts()

print(f"Experiment logged to: {tracker.log_path}")
print(f"Final metrics: {results}")
```

**Output:**

```
Seed set to 42
Experiment: sentiment_distillation_v1
Checkpoint saved at epoch 3
Checkpoint saved at epoch 6
Checkpoint saved at epoch 9
Checkpoint saved at epoch 12
Experiment logged to: ./experiments/sentiment_distillation_v1
Final metrics: {'accuracy': 0.9123, 'loss': 0.2145, 'compression': 0.65, 'f1': 0.9087}
```

## 🏗️ Architecture

The toolkit follows a modular pipeline architecture with three main components: (1) Core distillation engine that orchestrates the pruning and fine-tuning process, (2) Utility modules providing model wrappers, data handling, pruning strategies, and evaluation tools, and (3) Demo/experiment layer for running end-to-end workflows. The design emphasizes configurability, reproducibility, and extensibility to support various distillation strategies and model architectures.

### File Structure

```
┌─────────────────────────────────────────────────────────┐
│                    Demo Layer                           │
│  (demo.py - End-to-end experiments & examples)          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Core Engine                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │  KnowledgeDistiller                              │   │
│  │  - Orchestrates distillation pipeline           │   │
│  │  - Manages teacher/student interaction          │   │
│  │  - Coordinates pruning + fine-tuning            │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  DistillationConfig                              │   │
│  │  - Hyperparameters & strategy settings          │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 Utility Modules                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Pruning    │  │  Evaluation  │  │  Data Utils  │   │
│  │  Strategies │  │  Harness     │  │  & Loaders   │   │
│  └─────────────┘  └──────────────┘  └──────────────┘   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Model      │  │  Experiment  │  │  Metrics &   │   │
│  │  Wrappers   │  │  Tracking    │  │  Analysis    │   │
│  └─────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Files

- **lib/core.py** — Implements the main KnowledgeDistiller class and DistillationConfig, orchestrating the multi-stage distillation pipeline including pruning, fine-tuning, and alignment.
- **lib/utils.py** — Provides utility functions and classes for model loading, data preparation, pruning strategies, evaluation harness, metrics calculation, and experiment tracking.
- **demo.py** — Contains end-to-end demonstration scripts and example experiments showcasing different distillation workflows, configurations, and use cases with reproducible results.

### Design Decisions

- Separated core distillation logic from utilities to maintain clear boundaries and enable independent testing and extension of components.
- Used configuration objects (DistillationConfig) instead of scattered parameters to improve reproducibility and experiment management.
- Implemented staged distillation (pruning → fine-tuning → alignment) to allow gradual knowledge removal with quality checkpoints at each stage.
- Designed modular pruning strategies as pluggable components to support different approaches (magnitude, gradient-based, structured) without changing core logic.
- Built evaluation harness as a separate module to enable comprehensive testing and comparison of teacher vs student models across multiple metrics.
- Included experiment tracking and seed management from the start to ensure all distillation runs are reproducible and results are auditable.

## 🔧 Technical Details

### Dependencies

- **torch** (1.12+) — Core deep learning framework for model training, inference, and parameter manipulation during distillation.
- **numpy** (1.21+) — Numerical computing for array operations, metrics calculation, and data preprocessing.
- **scipy** (1.7+) — Scientific computing utilities for statistical analysis and advanced mathematical operations in evaluation.
- **scikit-learn** (1.0+) — Machine learning utilities for metrics (F1, precision, recall) and calibration analysis.

### Key Algorithms / Patterns

- Knowledge Distillation with Temperature Scaling: Softens teacher predictions using temperature parameter to transfer dark knowledge to student model.
- Magnitude-based Pruning: Removes parameters with smallest absolute weights below importance threshold to eliminate non-essential knowledge.
- Gradient-based Importance Scoring: Calculates parameter importance using gradient magnitudes on task-specific data to guide selective pruning.
- Multi-stage Fine-tuning: Alternates between pruning and fine-tuning phases to recover performance after each knowledge removal step.
- Alignment Loss: Combines task-specific loss with KL divergence between teacher and student to preserve essential capabilities while specializing.

### Important Notes

- GPU memory requirements scale with teacher model size; consider gradient checkpointing for very large models (>1B parameters).
- Pruning ratio should be tuned per task; aggressive pruning (>80%) may cause catastrophic forgetting of essential capabilities.
- Temperature parameter in distillation typically ranges from 1.5-4.0; higher values transfer more soft knowledge but may slow convergence.
- Evaluation should include both in-domain and out-of-domain tests to verify knowledge removal vs. task specialization trade-offs.
- Checkpoint frequently during distillation as some pruning configurations may lead to unrecoverable performance degradation.

## ❓ Troubleshooting

### CUDA out of memory error during distillation

**Cause:** Teacher and student models loaded simultaneously consume too much GPU memory, especially with large batch sizes.

**Solution:** Reduce batch size in DistillationConfig, enable gradient checkpointing with config.use_gradient_checkpointing=True, or use CPU offloading for teacher model with teacher.to('cpu') and move only during forward passes.

### Student model performance drops significantly after pruning

**Cause:** Pruning ratio is too aggressive or important task-specific parameters were removed based on incorrect importance criteria.

**Solution:** Reduce pruning_ratio (try 0.3-0.5 initially), use gradient-based importance scoring instead of magnitude-based, or add more fine-tuning epochs after each pruning stage to allow recovery.

### Distillation loss not decreasing or diverging

**Cause:** Learning rate too high, temperature parameter mismatched, or student model capacity insufficient for task complexity.

**Solution:** Lower learning rate (try 1e-5 to 1e-4), adjust temperature to 2.0-3.0, increase student model size, or add warmup schedule with config.warmup_steps=500.

### Reproducibility issues - different results with same seed

**Cause:** Non-deterministic CUDA operations or missing seed initialization in data loaders and model initialization.

**Solution:** Call set_seed() before any model/data operations, add torch.backends.cudnn.deterministic=True and torch.backends.cudnn.benchmark=False, and set worker_init_fn in DataLoader.

### Evaluation metrics show student outperforms teacher

**Cause:** Data leakage between train and eval sets, or teacher model not properly loaded/evaluated in the same mode as student.

**Solution:** Verify data splits are correct and non-overlapping, ensure both models use model.eval() during evaluation, and check that teacher weights are properly loaded without modifications.

---

This project was generated as a demonstration of expert knowledge distillation techniques. The implementation provides a foundation for research and experimentation with model compression and specialization. For production use, consider additional optimizations such as quantization, dynamic pruning schedules, and task-specific architectural adaptations. Contributions and extensions are welcome to support additional model architectures and distillation strategies.