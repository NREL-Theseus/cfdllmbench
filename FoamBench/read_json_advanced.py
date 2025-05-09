import os
import json
import yaml

input_json = "./Dataset/FoamBench_advanced.json"
output_root = "./Dataset/Advanced"

base_config = {
    "MetaGPT_PATH": "xxxx",  # Add your MetaGPT path
    "OPENAI_API_KEY": "xxxx",  # Add your OpenAI API key
    "OPENAI_BASE_URL": "xxxx",  # Add your OpenAI base URL
    "OPENAI_PROXY": "",
    "batchsize": 10,
    "max_loop": 10,
    "model": "gpt-4o",
    "run_times": 1,
    "searchdocs": 2,
    "temperature": 0.0
}

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

for dataset_name, files in data.items():
    base_path = os.path.join(output_root, dataset_name)
    gt_files_path = os.path.join(base_path, "GT_Files")
    os.makedirs(gt_files_path, exist_ok=True)

    for rel_path, content in files.items():
        if rel_path == "usr_requirement":
            # Save usr_requirement.txt at base path
            usr_path = os.path.join(base_path, "usr_requirement.txt")
            with open(usr_path, "w", encoding="utf-8") as f:
                f.write(content)
            # Write YAML next to usr_requirement
            config = base_config.copy()
            config["usr_requirment"] = content
            yaml_path = os.path.join(base_path, f"{dataset_name}.yaml")
            with open(yaml_path, "w", encoding="utf-8") as yfile:
                yaml.dump(config, yfile, default_flow_style=False, sort_keys=False)
        elif rel_path == "Allrun":
            allrun_path = os.path.join(gt_files_path, "Allrun")
            with open(allrun_path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            # Everything else goes inside GT_Files
            full_path = os.path.join(gt_files_path, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

print("✅ Recovery completed. Files are saved under:", output_root)
