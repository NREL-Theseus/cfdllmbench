# OpenFOAM Benchmark Framework

A framework for running and evaluating different algorithms on OpenFOAM benchmark cases.

## Project Structure

```
openfoam_benchmark/
├── Dataset/                  # Contains all benchmark datasets
│   └── [dataset_name]/      # Each dataset has multiple cases
│       └── [case_number]/   # Case-specific files and requirements
├── algorithm/               # Contains different algorithm implementations
│   └── [algorithm_name]/   # Each algorithm has its own directory
│       └── foambench_main.py  # Main script for the algorithm
├── logs/                    # Log files for benchmark runs
├── runs/                    # Output files from benchmark runs
├── run_benchmarks.py        # Local benchmark runner
└── run_perlmutter.sh        # Perlmutter cluster benchmark runner
```

## Prerequisites

- Python 3.x
- OpenFOAM installation
- WM_PROJECT_DIR environment variable set to OpenFOAM installation path
- Access to Perlmutter cluster (for distributed runs)

## Setup

1. Clone this repository:
```bash
git clone [repository_url]
cd openfoam_benchmark
```

2. Download the benchmark datasets:
```bash
# Download datasets from HuggingFace to Dataset/ directory
```

3. Add your algorithm:
```bash
# Clone or copy your algorithm implementation to algorithm/[algorithm_name]/
# Ensure it contains a foambench_main.py script
```

4. Set OpenFOAM environment variable:
```bash
export WM_PROJECT_DIR="/path/to/your/OpenFOAM-10"
```

## Available Datasets

The framework includes the following benchmark datasets:
- BernardCells
- Cavity
- counterFlowFlame2D
- Cylinder
- forwardStep
- damBreakWithObstacle
- obliqueShock
- pitzDaily
- shallowWaterWithSquareBump
- squareBend
- wedge

Each dataset has 10 cases (numbered 1-10) for comprehensive testing.

## Usage

### Local Execution

To run benchmarks locally:
```bash
python run_benchmarks.py <algorithm_name>
```

Example:
```bash
python run_benchmarks.py example
```

### Perlmutter Cluster Execution

To run benchmarks on Perlmutter:
```bash
./run_perlmutter.sh <algorithm_name>
```

Example:
```bash
./run_perlmutter.sh example
```

## Algorithm Implementation

To add a new algorithm:
1. Create a directory in `algorithm/` with your algorithm name
2. Implement `foambench_main.py` with the following interface:
   ```python
   # Required command line arguments:
   --openfoam_path    # Path to OpenFOAM installation
   --output          # Directory to store results
   --prompt_path     # Path to user requirements file
   ```

## Output

- Log files are stored in `logs/` directory
- Results are stored in `results/[dataset]/[case]/` directories
- Each run creates separate output and error logs

## Error Handling

The framework includes checks for:
- Algorithm directory existence
- Main script existence
- OpenFOAM environment setup
- Dataset and case availability

## Contributing

1. Fork the repository
2. Create your algorithm implementation
3. Test with the benchmark framework
4. Submit a pull request

## License

[Add your license information here]



