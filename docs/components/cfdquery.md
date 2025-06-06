# CFDQuery: Conceptual Understanding

CFDQuery evaluates LLMs' conceptual understanding of computational fluid dynamics through carefully curated multiple-choice questions.

## Overview

- **90 graduate-level questions** from CFD lecture materials
- **Four-choice format** with single correct answer
- **Comprehensive coverage** of key CFD concepts

## Topics Covered

### Numerical Methods
- Finite difference methods
- Finite volume methods  
- Finite element methods
- Discretization schemes
- Stability and convergence

### Turbulence Modeling
- Reynolds-averaged equations
- Large eddy simulation
- Direct numerical simulation
- Turbulence models (k-ε, k-ω, RSM)

### Solver Theory
- Pressure-velocity coupling
- Solution algorithms
- Boundary conditions
- Grid generation

## Example Question

**What is the primary advantage of the finite volume method over finite difference methods?**

A) Higher-order accuracy
B) Conservation properties
C) Computational efficiency  
D) Easier implementation

*Correct Answer: B - Conservation properties*

## Evaluation

Models are scored based on **accuracy**: the percentage of questions answered correctly.

```python
def evaluate_cfdquery(responses, ground_truth):
    correct = sum(1 for r, gt in zip(responses, ground_truth) if r == gt)
    return correct / len(ground_truth)
```

## Running CFDQuery

```bash
python run_cfdquery.py --model your_model --output results/cfdquery/
```
