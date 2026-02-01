# Expert Knowledge Distillation Toolkit

> Transform large generalist models into lean, specialized experts through targeted knowledge distillation

A Python library for distilling large pre-trained models into specialized, task-specific versions by systematically removing irrelevant knowledge while preserving essential capabilities. This toolkit enables researchers and practitioners to create efficient, focused models that excel at specific tasks without the overhead of general-purpose knowledge.

## What is Knowledge Distillation?

Knowledge distillation is a model compression technique where a smaller "student" model learns to mimic a larger "teacher" model. This toolkit goes further by **progressively pruning** the student model to remove knowledge that isn't relevant to the target task, resulting in a highly specialized "expert" model.

```
┌─────────────────┐      Distillation       ┌─────────────────┐
│  Large Teacher  │  ──────────────────────▶│ Small Student   │
│  (~52K params)  │   + Progressive Pruning │  (~4K params)   │
│  97.4% accuracy │                         │  96.5% accuracy │
└─────────────────┘                         └─────────────────┘
         │                                           │
         │                                           │
         ▼                                           ▼
   General Knowledge                        Task-Specific Expert
   (broad but slow)                        (focused & 10x faster)
```

## Features

- **Staged Knowledge Distillation Pipeline** — Multi-stage distillation process that progressively prunes non-essential parameters while maintaining performance on target tasks.
- **Real-World Dataset Support** — Includes demos using the Wisconsin Breast Cancer dataset and other sklearn datasets.
- **Comprehensive Visualizations** — Built-in plotting functions for training curves, model comparisons, sparsity analysis, and summary dashboards.
- **Gradient-Based Importance Pruning** — Intelligently identifies and removes weights that aren't critical for the target task.
- **Progressive Alpha Scheduling** — Smoothly transitions from distillation-focused to task-focused training.
- **Experiment Tracking** — Complete experiment management with JSON reports and model checkpoints.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/expert-knowledge-distillation.git
cd expert-knowledge-distillation

# Install dependencies
pip install -r requirements.txt
```

### Run the Demo

```bash
python demo.py
```

This will:
1. Load the Wisconsin Breast Cancer dataset (569 samples, 30 features)
2. Train a large teacher model (52K parameters)
3. Distill it into a smaller student model (4K parameters)
4. Generate comprehensive visualizations
5. Save the specialized expert model

### Example Output

```
📊 Model Compression:
  Teacher parameters: 52,194
  Student parameters: 4,322
  Reduction: 91.72%
  Sparsity: 24.66%

🎯 Performance:
  Teacher accuracy: 97.37%
  Student accuracy: 96.49%
  Performance loss: -0.88%

✨ Specialization Benefits:
  ✓ 92% smaller model size
  ✓ ~10x faster inference
  ✓ Task-specific expertise
  ✓ Easier deployment on edge devices
```

## Generated Visualizations

The toolkit generates several visualizations to help understand the distillation process:

### Training Progress
Shows loss curves, accuracy over time, alpha schedule, sparsity progression, and pruning events.

### Model Comparison
Visual comparison of teacher vs student in terms of parameters and size.

### Layer Sparsity
Per-layer sparsity distribution showing which parts of the network were pruned.

### Summary Dashboard
At-a-glance view of all key metrics including accuracy delta and compression ratios.

## Usage

### Basic Knowledge Distillation

```python
from lib.core import KnowledgeDistiller, DistillationConfig

# Configure distillation
config = DistillationConfig(
    temperature=4.0,           # Softens teacher predictions
    alpha_start=0.7,           # Initial distillation weight
    alpha_end=0.3,             # Final distillation weight
    learning_rate=0.001,
    num_epochs=25,
    pruning_start_epoch=8,     # When to start pruning
    pruning_end_epoch=20,      # When to end pruning
    target_sparsity=0.4        # Target 40% sparsity
)

# Create distiller
distiller = KnowledgeDistiller(
    teacher_model=teacher,
    student_model=student,
    config=config
)

# Run distillation
history = distiller.distill(
    train_loader=train_loader,
    val_loader=val_loader
)

# Create specialized expert
from lib.core import SpecializedExpert
expert = SpecializedExpert(
    model=distiller.student_model,
    task_name="Cancer Classification"
)

# Make predictions
predictions = expert.predict(test_data)
```

### Custom Model Architectures

```python
import torch.nn as nn

# Define your own teacher model
class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(30, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.network(x)

# Define your own student model
class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.network(x)

# Use with the distiller
teacher = LargeModel()
student = SmallModel()
distiller = KnowledgeDistiller(teacher, student, config)
```

### Generate Visualizations

```python
from lib.utils import (
    plot_training_history,
    plot_model_comparison,
    plot_layer_sparsity,
    plot_distillation_summary,
    compare_models
)

# Plot training curves
plot_training_history(history, save_path="training.png")

# Compare models
comparison = compare_models(teacher, student)
plot_model_comparison(comparison, save_path="comparison.png")

# Analyze sparsity
plot_layer_sparsity(student, save_path="sparsity.png")
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system diagrams and component explanations.

```
expert-distillation/
├── demo.py                 # End-to-end demonstration
├── lib/
│   ├── core.py             # KnowledgeDistiller, DistillationConfig
│   └── utils.py            # Utilities and visualization
├── distillation_results/   # Generated outputs
│   ├── *.png               # Visualizations
│   ├── *.pt                # Model checkpoints
│   └── *.json              # Reports
├── requirements.txt
├── README.md
└── ARCHITECTURE.md
```

## Requirements

- Python 3.8+
- PyTorch >= 1.12.0
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0

## Key Algorithms

### Distillation Loss

The total loss combines soft targets from the teacher with hard labels:

```
Total Loss = α × KL_Divergence(soft_student, soft_teacher) + (1-α) × CrossEntropy(student, labels)
```

Where α decreases over training (0.7 → 0.3) to transition from distillation to task focus.

### Importance-Based Pruning

Weights are pruned based on gradient-based importance scores:

```python
importance[param] = Σ |∂L/∂param| over task data
mask = importance > quantile(importance, pruning_rate)
param.data *= mask
```

## Troubleshooting

### Student accuracy drops significantly after pruning

**Solution:** Reduce `target_sparsity` (try 0.2-0.3) or increase fine-tuning epochs after pruning phase.

### Training is unstable / loss diverging

**Solution:** Lower learning rate to 1e-4, reduce temperature to 2.0-3.0, or add more warmup epochs before pruning starts.

### Out of memory errors

**Solution:** Reduce batch size or use gradient checkpointing for very large models.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License

---

*This project was generated using the [Automated Idea Expansion](https://github.com/qsimeon/automated-idea-expansion) workflow.*
