import os
import yaml
import subprocess
import openai
import argparse


DATASETS = [
    "BernardCells", "Cavity", "counterFlowFlame2D", "Cylinder",
    "forwardStep", "obliqueShock", "pitzDaily", "squareBend",
    "wedge", "shallowWaterWithSquareBump", "damBreakWithObstacle"
]
CASES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Get OpenFOAM path from environment variable
WM_PROJECT_DIR = os.environ.get('WM_PROJECT_DIR')
if not WM_PROJECT_DIR:
    print("Error: WM_PROJECT_DIR is not set in the environment.")
    exit(1)

BASE_DIR = "Dataset"  # Base directory where all datasets are stored


def read_user_requirement(file_path):
    """Reads and returns the content of user_requirement.txt."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
            return content
    else:
        print(f"File not found: {file_path}")
        return ""


def run_benchmark(dataset, case, algorithm_name):
    """Creates user_requirement.txt and runs foambench_main.py."""
    folder_path = os.path.abspath(os.path.join(BASE_DIR, dataset, str(case)))
    requirement_txt_path = os.path.abspath(os.path.join(folder_path, "user_requirement.txt"))
    output_folder = os.path.abspath(os.path.join("runs", dataset, str(case)))
    os.makedirs(output_folder, exist_ok=True)

    usr_requirement = read_user_requirement(requirement_txt_path)

    command = f"python ./algorithm/{algorithm_name}/foambench_main.py --openfoam_path {WM_PROJECT_DIR} --output {output_folder} --prompt_path {requirement_txt_path}"
    print(f"Running: {command}")

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark for {dataset}/{case}: {e}")

if __name__ == "__main__":
    """Loops through all datasets and runs benchmarks for cases"""

    parser = argparse.ArgumentParser(description='Run OpenFOAM benchmarks')
    parser.add_argument('--algorithm_name', type=str, default="OpenFOAM-Agent", help='Name of the algorithm (same as the directory name in algorithm folder)')
    args = parser.parse_args()

    print(f"Running benchmarks for {args.algorithm_name}")

    for dataset in DATASETS:
        for case in CASES:
            run_benchmark(dataset, case, args.algorithm_name)
