# Expert Knowledge Distillation - System Architecture

This document provides a comprehensive overview of the Expert Knowledge Distillation Toolkit's architecture, including system diagrams and component explanations.

## Overview

The toolkit transforms large, general-purpose neural networks ("teachers") into smaller, specialized "expert" models ("students") through a process called **knowledge distillation**. The key innovation is **progressive knowledge removal** - systematically pruning irrelevant knowledge while preserving task-specific capabilities.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXPERT KNOWLEDGE DISTILLATION                         │
│                           System Architecture                                │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   Input Data    │
                              │  (Real-World    │
                              │    Dataset)     │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
         ┌──────────────────┐                   ┌──────────────────┐
         │  General Data    │                   │  Task-Specific   │
         │  (Pre-training)  │                   │    Data          │
         └────────┬─────────┘                   └────────┬─────────┘
                  │                                      │
                  ▼                                      │
┌─────────────────────────────────────┐                  │
│         TEACHER MODEL               │                  │
│  ┌───────────────────────────────┐  │                  │
│  │    Large General-Purpose      │  │                  │
│  │    Neural Network             │  │                  │
│  │                               │  │                  │
│  │  • Many parameters            │  │                  │
│  │  • Broad knowledge            │  │                  │
│  │  • High capacity              │  │                  │
│  └───────────────────────────────┘  │                  │
│                                     │                  │
│  Parameters: ~240K                  │                  │
│  Size: ~0.93 MB                     │                  │
└──────────────────┬──────────────────┘                  │
                   │                                     │
                   │  Soft Targets                       │
                   │  (Temperature-scaled                │
                   │   predictions)                      │
                   │                                     │
                   ▼                                     ▼
         ┌─────────────────────────────────────────────────────────┐
         │              KNOWLEDGE DISTILLATION ENGINE              │
         │  ┌─────────────────────────────────────────────────┐    │
         │  │                                                 │    │
         │  │  ┌─────────────┐    ┌─────────────────────┐    │    │
         │  │  │ Distillation│    │  Progressive       │    │    │
         │  │  │    Loss     │───▶│   Pruning          │    │    │
         │  │  │  (KL Div)   │    │  (Importance-based)│    │    │
         │  │  └─────────────┘    └─────────────────────┘    │    │
         │  │        │                      │                │    │
         │  │        ▼                      ▼                │    │
         │  │  ┌─────────────┐    ┌─────────────────────┐    │    │
         │  │  │   Task      │    │  Alpha Schedule     │    │    │
         │  │  │   Loss      │◀───│  (KD → Task focus)  │    │    │
         │  │  │ (CrossEnt)  │    │                     │    │    │
         │  │  └─────────────┘    └─────────────────────┘    │    │
         │  │                                                 │    │
         │  └─────────────────────────────────────────────────┘    │
         │                                                         │
         │  Configuration:                                         │
         │  • Temperature: 4.0                                     │
         │  • Alpha: 0.7 → 0.3 (progressive)                       │
         │  • Target Sparsity: 40%                                 │
         │  • Epochs: 25                                           │
         └────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
         ┌─────────────────────────────────────────────────────────┐
         │              STUDENT MODEL (Specialized)                │
         │  ┌───────────────────────────────────────────────────┐  │
         │  │         Compact Specialized Expert                │  │
         │  │                                                   │  │
         │  │    • Fewer parameters (~67K)                      │  │
         │  │    • Task-specific knowledge                      │  │
         │  │    • Sparse weights (pruned)                      │  │
         │  │    • Fast inference                               │  │
         │  └───────────────────────────────────────────────────┘  │
         │                                                         │
         │  Parameters: ~67K (72% reduction)                       │
         │  Size: ~0.26 MB (72% smaller)                           │
         │  Sparsity: ~40% of remaining weights                    │
         └────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   SPECIALIZED EXPERT    │
                    │                         │
                    │  • Optimized for task   │
                    │  • Deployable           │
                    │  • Interpretable        │
                    └─────────────────────────┘
```

## Data Flow Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           DISTILLATION PIPELINE                              │
└──────────────────────────────────────────────────────────────────────────────┘

  STAGE 1: Teacher Training        STAGE 2: Knowledge Transfer       STAGE 3: Specialization
  ═══════════════════════          ═════════════════════════         ════════════════════════
  
  ┌─────────────────┐              ┌──────────────────────┐          ┌─────────────────────┐
  │ General Data    │              │ Task-Specific Data   │          │ Pruning + Fine-tune │
  └────────┬────────┘              └──────────┬───────────┘          └──────────┬──────────┘
           │                                  │                                 │
           ▼                                  ▼                                 ▼
  ┌─────────────────┐              ┌──────────────────────┐          ┌─────────────────────┐
  │ Pre-train       │              │  Compute             │          │  Calculate          │
  │ Teacher Model   │─────────────▶│  Distillation Loss   │─────────▶│  Importance Scores  │
  └─────────────────┘              │  + Task Loss         │          │  (Gradient-based)   │
                                   └──────────────────────┘          └──────────┬──────────┘
                                              │                                 │
                                              │                                 ▼
                                              │                      ┌─────────────────────┐
                                              │                      │  Prune Low-         │
                                              │                      │  Importance Weights │
                                              │                      └──────────┬──────────┘
                                              │                                 │
                                              ▼                                 ▼
                                   ┌──────────────────────┐          ┌─────────────────────┐
                                   │  Update Student      │◀─────────│  Repeat Until       │
                                   │  Weights             │          │  Target Sparsity    │
                                   └──────────────────────┘          └─────────────────────┘
                                              │
                                              ▼
                                   ┌──────────────────────┐
                                   │  Specialized Expert  │
                                   │  Ready for Deploy    │
                                   └──────────────────────┘
```

## Component Architecture

### Core Components (`lib/core.py`)

```
┌────────────────────────────────────────────────────────────────────┐
│                          lib/core.py                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  DistillationConfig                                         │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  • temperature: float        (softmax temperature)          │   │
│  │  • alpha_start/end: float    (KD loss weight schedule)      │   │
│  │  • learning_rate: float      (optimizer LR)                 │   │
│  │  • num_epochs: int           (training iterations)          │   │
│  │  • pruning_start/end: int    (pruning schedule)             │   │
│  │  • target_sparsity: float    (final pruning ratio)          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  KnowledgeDistiller                                         │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  Methods:                                                   │   │
│  │  • distill(train_loader, val_loader)                        │   │
│  │  • distillation_loss(student_logits, teacher_logits, T)     │   │
│  │  • compute_knowledge_importance(dataloader)                 │   │
│  │  • prune_irrelevant_knowledge(pruning_rate)                 │   │
│  │  • train_step(inputs, targets, optimizer, alpha)            │   │
│  │  • evaluate(dataloader)                                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  SpecializedExpert                                          │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  Methods:                                                   │   │
│  │  • predict(inputs) → predictions + confidence               │   │
│  │  • save(path) → serialize to disk                           │   │
│  │  • load(path, model) → deserialize from disk                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Utility Components (`lib/utils.py`)

```
┌────────────────────────────────────────────────────────────────────┐
│                          lib/utils.py                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Model Analysis Functions                                  │    │
│  │  ──────────────────────────────────────────────────────    │    │
│  │  • count_parameters(model)                                 │    │
│  │  • get_model_size_mb(model)                                │    │
│  │  • compute_sparsity(model)                                 │    │
│  │  • compare_models(model1, model2)                          │    │
│  │  • get_layer_statistics(model)                             │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Pruning & Knowledge Functions                             │    │
│  │  ──────────────────────────────────────────────────────    │    │
│  │  • analyze_layer_importance(scores)                        │    │
│  │  • create_knowledge_mask(scores, threshold)                │    │
│  │  • apply_knowledge_mask(model, masks)                      │    │
│  │  • compute_knowledge_retention(importance, model)          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Visualization Functions                                   │    │
│  │  ──────────────────────────────────────────────────────    │    │
│  │  • plot_training_history(history)                          │    │
│  │  • plot_model_comparison(comparison)                       │    │
│  │  • plot_layer_sparsity(model)                              │    │
│  │  • plot_distillation_summary(metrics)                      │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Evaluation & Reporting                                    │    │
│  │  ──────────────────────────────────────────────────────    │    │
│  │  • evaluate_model(model, dataloader)                       │    │
│  │  • compute_task_alignment(model, task_data, general_data)  │    │
│  │  • save_distillation_report(path, teacher, student, ...)   │    │
│  │  • create_progressive_schedule(start, end, steps)          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Knowledge Distillation Algorithm

### Loss Function

The total loss combines distillation loss (soft targets from teacher) and task loss (hard labels):

```
Total Loss = α × Distillation_Loss + (1 - α) × Task_Loss

Where:
  Distillation_Loss = T² × KL_Divergence(
      softmax(student_logits / T),
      softmax(teacher_logits / T)
  )
  
  Task_Loss = CrossEntropy(student_logits, true_labels)
  
  α = alpha_start → alpha_end (decreases over training)
  T = temperature (typically 2-4)
```

### Progressive Pruning

```
For each pruning epoch:
    1. Compute gradient-based importance scores:
       importance[param] = Σ |∂L/∂param| over task data
    
    2. Determine pruning threshold:
       threshold = quantile(importance, current_pruning_rate)
    
    3. Create and apply mask:
       mask = importance > threshold
       param.data *= mask
    
    4. Current_pruning_rate increases from 0 → target_sparsity
```

## File Structure

```
expert-distillation/
├── demo.py                 # Main demonstration script
│                           # - Loads Wisconsin Breast Cancer dataset
│                           # - Trains teacher & distills student
│                           # - Generates visualizations
│
├── lib/
│   ├── core.py             # Core distillation engine
│   │                       # - DistillationConfig
│   │                       # - KnowledgeDistiller
│   │                       # - SpecializedExpert
│   │
│   └── utils.py            # Utility functions
│                           # - Model analysis
│                           # - Visualization
│                           # - Reporting
│
├── distillation_results/   # Output directory
│   ├── model_comparison.png
│   ├── training_history.png
│   ├── layer_sparsity.png
│   ├── distillation_summary.png
│   ├── specialized_cancer_expert.pt
│   └── distillation_report.json
│
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── ARCHITECTURE.md         # This file
```

## Key Design Decisions

1. **Staged Distillation**: Separates teacher training from knowledge transfer, allowing for flexible workflows.

2. **Progressive Alpha Schedule**: Starts with high distillation weight (α=0.7) and transitions to task-focused training (α=0.3), enabling smooth knowledge transfer.

3. **Gradient-Based Importance**: Uses gradient magnitudes on task data to identify which weights are critical for the target task.

4. **Modular Architecture**: Clean separation between core engine, utilities, and demo allows for easy extension and testing.

5. **Comprehensive Visualization**: Built-in plotting functions provide interpretable insights into the distillation process.

## Usage Example

```python
from lib.core import DistillationConfig, KnowledgeDistiller

# Configure distillation
config = DistillationConfig(
    temperature=4.0,
    alpha_start=0.7,
    alpha_end=0.3,
    num_epochs=25,
    target_sparsity=0.4
)

# Create distiller
distiller = KnowledgeDistiller(
    teacher_model=teacher,
    student_model=student,
    config=config
)

# Run distillation
history = distiller.distill(train_loader, val_loader)
```
