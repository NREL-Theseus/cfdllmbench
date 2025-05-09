import os
import subprocess
import argparse
import shutil

# === Paths & Constants ===
BASIC_ROOT = "./Dataset/Basic"
ADVANCED_ROOT = "./Dataset/Advanced"
SRC_DIR = "./algorithm/MetaOpenFOAM/src"
PYTHON = "python"

# === Workflow Step Scripts ===
SCRIPTS = [
    ("config_path", "config_path.py"),
    ("postprocess", "Tutorial_postprocess.py"),
    ("db_add_summary", "Langchain_database_add_tutorial_summary.py"),
    ("db_add_tutorial", "Langchain_database_add_tutorial.py"),
    ("db_add_command", "Langchain_database_add_command.py"),
    ("db_add_allrun", "Langchain_database_add_allrun.py"),
    ("main", "metaOpenfoam_v2.py"),
]

def run_workflow(input_yaml):
    """Set CONFIG_FILE_PATH and run full workflow on one YAML case."""
    input_dir = os.path.dirname(input_yaml)
    case_name = os.path.basename(input_dir)
    os.environ["CONFIG_FILE_PATH"] = input_yaml
    os.environ["INPUT_DIR"] = os.path.dirname(input_yaml)
    os.environ["INPUT_DIR_METAFOAM"] = os.path.dirname(input_yaml)
    os.environ["WM_PROJECT_DIR"] = "xxxxx" # provide your openfoam path here

    meta_root = "./algorithm/MetaOpenFOAM"
    run_source = os.path.join(meta_root, "run")

    # === Step 1: Create a fresh run/ folder before the workflow ===
    if os.path.exists(run_source):
        shutil.rmtree(run_source)
    os.makedirs(run_source, exist_ok=True)

    for label, script in SCRIPTS:
        script_path = os.path.join(SRC_DIR, script)
        print(f"Running {label}: {script_path}")

        try:
            subprocess.run([PYTHON, "-u", script_path, input_yaml] if label == "config_path" else [PYTHON, "-u", script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed on {label} for {input_yaml}: {e}")
            return

            # === Step 3: Move contents of run/ into case directory ===
    if os.path.exists(run_source):
        print(f"Moving contents of 'run/' → {input_dir}")
        for item in os.listdir(run_source):
            src_item = os.path.join(run_source, item)
            dst_item = os.path.join(input_dir, item)

            # Clean existing files if needed
            if os.path.exists(dst_item):
                if os.path.isdir(dst_item):
                    shutil.rmtree(dst_item)
                else:
                    os.remove(dst_item)

            shutil.move(src_item, dst_item)

        shutil.rmtree(run_source)
    else:
        print(f"Source run folder not found: {run_source}")

def run_basic_cases():
    for dataset in os.listdir(BASIC_ROOT):
        for case_id in range(1, 11):
            case_yaml = os.path.join(BASIC_ROOT, dataset, str(case_id), f"{dataset}.yaml")
            if os.path.isfile(case_yaml):
                print(f"\nRunning BASIC case: {dataset}/{case_id}")
                run_workflow(case_yaml)

def run_advanced_cases():
    for dataset in os.listdir(ADVANCED_ROOT):
        case_yaml = os.path.join(ADVANCED_ROOT, dataset, f"{dataset}.yaml")
        if os.path.isfile(case_yaml):
            print(f"Running ADVANCED case: {dataset}")
            run_workflow(case_yaml)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Benchmark Runner for CFD Workflow")
    parser.add_argument('--mode', type=str, choices=['basic', 'advanced', 'all'], default='all',
                        help='Which group of cases to run')
    args = parser.parse_args()

    print(f"Running CFD workflow for: {args.mode.upper()}")
    if args.mode in ('basic', 'all'):
        run_basic_cases()
    if args.mode in ('advanced', 'all'):
        run_advanced_cases()

