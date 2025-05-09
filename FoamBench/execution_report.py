import os
import csv
import pandas as pd

def analyze_basic_structure(base_dir):
    dataset_path = os.path.join(base_dir, "Basic")
    datasets = os.listdir(dataset_path)
    results = []

    for dataset in datasets:
        dataset_dir = os.path.join(dataset_path, dataset)
        if not os.path.isdir(dataset_dir):
            continue

        for i in range(1, 11):
            case_dir = os.path.join(dataset_dir, str(i))
            if not os.path.exists(case_dir):
                continue

            run_folders = [f for f in os.listdir(case_dir)
                           if os.path.isdir(os.path.join(case_dir, f)) and f != "GT_Files"]
            if not run_folders:
                results.append([dataset, i, 0])
                continue

            run_path = os.path.join(case_dir, run_folders[0])
            success = 0

            for root, dirs, files in os.walk(run_path):
                for dir_name in dirs:
                    folder_path = os.path.join(root, dir_name)
                    log_files = [f for f in os.listdir(folder_path) if f.startswith("log.") and f.endswith("Foam")]
                    for log_file in log_files:
                        log_path = os.path.join(folder_path, log_file)
                        try:
                            with open(log_path, "r") as file:
                                lines = file.readlines()
                                last_line = lines[-2].strip() if len(lines) > 1 else ""
                                if last_line == "End":
                                    success = 1
                                    break
                        except:
                            continue
                    if success:
                        break
                if success:
                    break

            results.append([dataset, i, success])
    return results


def analyze_advanced_structure(base_dir):
    dataset_path = os.path.join(base_dir, "Advanced")
    datasets = os.listdir(dataset_path)
    results = []

    for dataset in datasets:
        case_dir = os.path.join(dataset_path, dataset)
        if not os.path.isdir(case_dir):
            continue

        run_folders = [f for f in os.listdir(case_dir)
                       if os.path.isdir(os.path.join(case_dir, f)) and f != "GT_Files"]
        if not run_folders:
            results.append([dataset, 1, 0])
            continue

        run_path = os.path.join(case_dir, run_folders[0])
        success = 0

        for root, dirs, files in os.walk(run_path):
            for dir_name in dirs:
                folder_path = os.path.join(root, dir_name)
                log_files = [f for f in os.listdir(folder_path) if f.startswith("log.") and f.endswith("Foam")]
                for log_file in log_files:
                    log_path = os.path.join(folder_path, log_file)
                    try:
                        with open(log_path, "r") as file:
                            lines = file.readlines()
                            last_line = lines[-2].strip() if len(lines) > 1 else ""
                            if last_line == "End":
                                success = 1
                                break
                    except:
                        continue
                if success:
                    break
            if success:
                break

        results.append([dataset, 1, success])

    return results

base_dir = "Dataset"
basic_results = analyze_basic_structure(base_dir)
advanced_results = analyze_advanced_structure(base_dir)

pd.DataFrame(basic_results, columns=["Dataset", "Directory", "Success"]).to_csv("basic_success_report.csv", index=False)
pd.DataFrame(advanced_results, columns=["Dataset", "Directory", "Success"]).to_csv("advanced_success_report.csv", index=False)

print("✅ Reports saved: 'basic_success_report.csv' and 'advanced_success_report.csv'")
