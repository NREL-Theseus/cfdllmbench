# FoamBench: OpenFOAM Automation

FoamBench evaluates end-to-end automation capabilities for OpenFOAM simulations from natural language instructions.

## Overview

- **126 total tasks**: 110 Basic + 16 Advanced
- **Complete workflow**: Case setup → Execution → Post-processing
- **Real OpenFOAM integration** with actual simulation runs

## Task Categories

### Basic Tasks (110)
- Standard solvers (simpleFoam, pimpleFoam)
- Common geometries (channel, cylinder, cavity)
- Standard boundary conditions
- Basic post-processing

### Advanced Tasks (16)
- Custom boundary conditions
- Complex geometries
- Multi-phase flows
- Advanced post-processing
- Parameter studies

## Workflow Steps

### 1. Case Creation
```bash
# Example: Create channel flow case
blockMesh
# Generate computational mesh
```

### 2. Boundary Conditions
```cpp
// Example boundary condition setup
inlet
{
    type            fixedValue;
    value           uniform (1 0 0);
}
```

### 3. Solver Execution
```bash
# Run appropriate solver
simpleFoam
```

### 4. Post-processing
```bash
# Extract results
postProcess -func wallShearStress
```

## Example Task

**Instruction**: "Set up a 2D channel flow simulation with Re=1000. Use a 100x20 mesh, inlet velocity 1 m/s, and extract pressure drop across the channel."

**Expected Actions**:
1. Create blockMeshDict for 100x20 mesh
2. Set inlet velocity boundary condition
3. Configure simpleFoam solver
4. Run simulation
5. Calculate pressure drop

## Evaluation

### Success Criteria
- Case setup completes without errors
- Simulation converges successfully  
- Required outputs are generated
- Results are physically reasonable

### Scoring
```python
def evaluate_foambench_task(task_result):
    score = 0
    if task_result.setup_success:
        score += 0.3
    if task_result.execution_success:
        score += 0.4  
    if task_result.postprocess_success:
        score += 0.3
    return score
```

## Running FoamBench

```bash
# Basic tasks
python run_foambench.py --level basic --model your_model

# Advanced tasks  
python run_foambench.py --level advanced --model your_model

# Specific task
python run_foambench.py --task channel_flow --model your_model
```

## Prerequisites

- OpenFOAM 8+ installed and configured
- Sufficient computational resources
- Write permissions for case directories
