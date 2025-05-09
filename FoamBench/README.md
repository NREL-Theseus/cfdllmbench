
# OpenFOAM Benchmark Framework

A framework for running and evaluating different algorithms on OpenFOAM benchmark cases.

## Project Structure

```
FoamBench/
├── Dataset/                  # Contains the benchmark datasets
│   ├── FoamBench_basic.json # To be downloaded from data repository
│   ├── FoamBench_advanced.json # To be downloaded from data repository
│   ├── Basic/ # json will be unpacked to this folder
│   │   └── [dataset_name]/
│   │       └── [1-10]/
│   │           ├── GT_Files/ # Reference OpenFOAM files unpacked from json
│   │           └── algorithm_files/ # OpenFOAM files produced by the framework
│   └── Advanced/
│       └── [dataset_name]/
│           ├── GT_Files/
│           └── algorithm_files/
├── algorithm/
│   └── framework_name/         # Folder top place the framework you want to run the benchmark with example, MetaOpenFOAM, Foam-Agent etc
├── logs/                    # Log files for benchmark runs
├── results/                 # Output and analysis results
├── runs/                    # Runtime files
├── execution_report.py
├── nmse_report.py
├── similarity_report.py
├── score_calculation.py
├── read_json_basic.py
├── read_json_advanced.py
└── run_benchmarks.py # Example file to run the benchmark with MetaOpenFOAM
```

## Prerequisites

- Python 3.x
- OpenFOAM version 10
- Set up OpenFOAM environment:

```bash
source /path/to/OpenFOAM-10/etc/bashrc
export WM_PROJECT_DIR="/path/to/OpenFOAM-10"
```

## Setup

1. Clone this repository:
```bash
git clone <repo_url>
cd FoamBench
```

2. Download benchmark dataset from `<repo_url>` and place:
   - `FoamBench_basic.json` into `Dataset/`
   - `FoamBench_advanced.json` into `Dataset/`

3. Add your algorithm:
```bash
# For example, to add MetaOpenFOAM:
cd algorithm
git clone git@github.com:Terry-cyx/MetaOpenFOAM.git
```

## Running Benchmarks

1. Unpack JSON benchmarks:
```bash
python read_json_basic.py      # Unpacks Basic dataset
python read_json_advanced.py   # Unpacks Advanced dataset
```

2. Run benchmark:
```bash
python run_benchmarks.py       # Runs benchmarks on both Basic and Advanced. Note that this script is custom built to run only with MetaOpenFOAM. Following the same logic you can connect the benchmark to any other framework as well.
```

After running, folder structure becomes:
```
Dataset/
├── Basic/
│   └── [dataset_name]/
│       └── [1-10]/
│           ├── GT_Files/
│           └── framework_generated_files/
└── Advanced/
    └── [dataset_name]/
        ├── GT_Files/
        └── framework_generated_files/
```

## Evaluation & Scoring

Run the following scripts in order:

1. **Execution Score** 
```bash
python execution_report.py
```
This generates:
- `execution_status_basic.csv`
- `execution_status_advanced.csv`

2. **Structural Similarity (Tree/ROUGE SCORE)** 
```bash
python similarity_report.py
```

3. **Numerical Accuracy (NMSE)**  
```bash
python nmse_report.py
```

4. **Final Metrics Calculation**
```bash
python score_calculation.py
```
This script outputs:
- `scores_basic.csv`
- `scores_advanced.csv`

Metrics computed:
- `M_exec`   — Execution score
- `M_NMSE`   — NMSE quality score
- `M_struct`   — Tree similarity score
- `M_file`   — Code-level ROUGE score
- `Success Ratio` — Overall success score
