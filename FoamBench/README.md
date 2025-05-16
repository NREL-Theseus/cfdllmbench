
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
- packages - pandas, pyvista and rogue_scores (pip install pyvista, pip install pandas, pip install rouge-score)
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

2. Download benchmark dataset from `https://kaggle.com/datasets/f7e918624a9d2e5321ea2ec1e4ef818919c89f120cb065ba04c9596b80f1297c` and place:
   - From FoamBench folder in the dataset, copy files `FoamBench_basic.json` and `FoamBench_advanced.json` into `Dataset/` folder.

3. Add your algorithm:
```bash
# For example, to add MetaOpenFOAM:
cd algorithm
git clone git@github.com:Terry-cyx/MetaOpenFOAM.git
```

## Running Benchmarks

1. Unpack JSON benchmarks:
```bash
python read_json_basic.py      # Unpacks Basic dataset. You will have to add certain paths and keys in this script
python read_json_advanced.py   # Unpacks Advanced dataset. You will have to add certain paths and keys in this script
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


## Docker
We provide a Dockerfile to simplify setting up the OpenFOAM + benchmarking environment.

🔧 Build the Docker Image
First, make sure Docker is installed and running on your system.

Then, from the root of this repository, build the Docker image:
```bash
docker build -t foam-bench .
```
This installs OpenFOAM 10 inside the container and copies all the benchmark files to the container. Then run the command
```bash
docker run -it --rm foam-bench
```
This will take you to the folder in container where the benchmark files are present. 
To know the location of OpenFOAM within container, perform
```bash
echo $WM_PROJECT_DIR
```
Further run the scripts as mentioned above, in sequence within the container. 
