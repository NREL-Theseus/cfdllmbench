# CFDCodeBench: Code Generation

CFDCodeBench assesses LLMs' ability to generate executable CFD code from natural language descriptions.

## Overview

- **24 diverse CFD problems** with varying complexity
- **Multiple implementation approaches** (FDM, FVM, analytical)
- **Execution-based evaluation** with numerical verification

## Problem Categories

### Basic Problems
- 1D heat conduction
- Simple flow calculations
- Analytical solutions

### Intermediate Problems  
- 2D heat transfer
- Navier-Stokes equations
- Boundary layer analysis

### Advanced Problems
- Turbulent flows
- Multi-physics coupling
- Complex geometries

## Example Problem

**Problem**: "Write Python code to solve 1D steady-state heat conduction with constant thermal conductivity. Use finite difference method with Dirichlet boundary conditions T(0)=100°C and T(L)=0°C."

**Expected Output**: Working Python code that:
- Implements finite difference discretization
- Applies boundary conditions correctly
- Solves the linear system
- Returns temperature distribution

## Evaluation Metrics

### 1. Execution Success
```python
def check_execution(code):
    try:
        exec(code)
        return True
    except:
        return False
```

### 2. Numerical Accuracy
```python
def compute_similarity(generated_output, reference_output):
    return 1 - np.mean(np.abs(generated_output - reference_output))
```

### 3. Code Quality
- Proper error handling
- Appropriate algorithms
- Clear structure

## Running CFDCodeBench

```bash
python run_cfdcodebench.py --model your_model --problems all --output results/
```
