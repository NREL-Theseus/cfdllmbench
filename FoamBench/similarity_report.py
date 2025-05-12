import os
import csv
from rouge_score import rouge_scorer

BASE_DIR = "Dataset"
SUBDIRS = ["0", "constant", "system"]

def read_cleaned_code(path):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except:
        return ""

    filtered_lines = [line for line in lines if not line.strip().startswith(("//", "/*", "*"))]
    content_started = False
    cleaned = []
    for line in filtered_lines:
        if "FoamFile" in line or "{" in line:
            content_started = True
        if content_started:
            cleaned.append(line)
    return "".join(cleaned)

def get_all_files(base_path):
    file_list = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if not f.startswith("log"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, base_path)
                if "polyMesh" not in rel_path.replace("\\", "/"):
                    file_list.append((rel_path, full_path))
    return file_list

def compute_rouge_score(gt_file, llm_file):
    gt_code = read_cleaned_code(gt_file)
    llm_code = read_cleaned_code(llm_file)
    if not gt_code or not llm_code:
        return 0.0
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        score = scorer.score(gt_code, llm_code)
        return score['rougeL'].fmeasure
    except:
        return 0.0

def compare_dir_pair(gt_dir, llm_dir):
    gt_files = get_all_files(gt_dir)
    llm_files_dict = dict(get_all_files(llm_dir)) if os.path.exists(llm_dir) else {}

    sim_scores = []
    tree_matches = 0

    for rel_path, gt_path in gt_files:
        llm_path = llm_files_dict.get(rel_path)
        if llm_path and os.path.exists(llm_path):
            sim = compute_rouge_score(gt_path, llm_path)
            sim_scores.append(sim)
            tree_matches += 1
        else:
            sim_scores.append(0.0)

    total_gt_files = len(gt_files)
    codebleu_score = sum(sim_scores) / total_gt_files if total_gt_files > 0 else 0.0
    tree_score = tree_matches / total_gt_files if total_gt_files > 0 else 0.0
    return codebleu_score, tree_score

def process_basic():
    output_file = "similarity_report_basic.csv"
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset", "Directory", "CodeBLEU", "TreeScore"])

        basic_path = os.path.join(BASE_DIR, "Basic")
        for dataset in os.listdir(basic_path):
            for i in range(1, 11):
                case_dir = os.path.join(basic_path, dataset, str(i))
                gt_dir = os.path.join(case_dir, "GT_Files")
                if not os.path.exists(gt_dir):
                    continue

                run_folders = [f for f in os.listdir(case_dir)
                               if os.path.isdir(os.path.join(case_dir, f)) and f != "GT_Files"]
                if not run_folders:
                    continue
                run_path = os.path.join(case_dir, run_folders[0])

                total_rogue = 0
                total_tree = 0
                count = 0

                for sub in SUBDIRS:
                    gt_sub = os.path.join(gt_dir, sub)
                    llm_sub = os.path.join(run_path, sub)
                    if os.path.exists(gt_sub) and os.path.exists(llm_sub):
                        rogue, tree = compare_dir_pair(gt_sub, llm_sub)
                        total_rogue += rogue
                        total_tree += tree
                        count += 1

                if count > 0:
                    writer.writerow([dataset, i,
                                     round(total_rogue / count, 4),
                                     round(total_tree / count, 4)])

def process_advanced():
    output_file = "similarity_report_advanced.csv"
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset", "Directory", "CodeBLEU", "TreeScore"])

        adv_path = os.path.join(BASE_DIR, "Advanced")
        for dataset in os.listdir(adv_path):
            case_dir = os.path.join(adv_path, dataset)
            gt_dir = os.path.join(case_dir, "GT_Files")
            if not os.path.exists(gt_dir):
                continue

            # LLM run folder (anything except GT_Files)
            run_folders = [f for f in os.listdir(case_dir)
                           if os.path.isdir(os.path.join(case_dir, f)) and f != "GT_Files"]
            if not run_folders:
                continue
            run_path = os.path.join(case_dir, run_folders[0])

            total_rogue = 0
            total_tree = 0
            count = 0

            for sub in SUBDIRS:
                gt_sub = os.path.join(gt_dir, sub)
                llm_sub = os.path.join(run_path, sub)
                if os.path.exists(gt_sub) and os.path.exists(llm_sub):
                    rogue, tree = compare_dir_pair(gt_sub, llm_sub)
                    total_rogue += rogue
                    total_tree += tree
                    count += 1

            if count > 0:
                writer.writerow([dataset, 1,
                                 round(total_rogue / count, 4),
                                 round(total_tree / count, 4)])

if __name__ == "__main__":
    process_basic()
    process_advanced()
    print("Finished. Reports: 'similarity_report_basic.csv' and 'similarity_report_advanced.csv'")
