import os
import json
import yaml

json_path = "./Dataset/FoamBench_basic.json"
output_root = "./Dataset/Basic"

# Base YAML config
base_config = {
    "MetaGPT_PATH": "/home/somasn/Documents/LLM/MetaOpenFOAM_path/MetaGPT", # Add your metagpt path
    "OPENAI_API_KEY": "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", # Add openai key
    "OPENAI_BASE_URL": "https://api.openai.com/v1", # add openai url
    "OPENAI_PROXY": "",
    "batchsize": 10,
    "max_loop": 1,
    "model": "gpt-4o",
    "run_times": 1,
    "searchdocs": 2,
    "temperature": 0.0
}

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for case_key, files in data.items():
    case_root = os.path.join(output_root, case_key)
    case_path = os.path.join(output_root, case_key, "GT_Files")
    os.makedirs(case_path, exist_ok=True)

    for file_path, content in files.items():
        if file_path == "usr_requirement":
            # Save this at the parent of GT_Files
            usr_req_path = os.path.join(output_root, case_key, "usr_requirement.txt")
            with open(usr_req_path, "w", encoding="utf-8") as f:
                f.write(content)
            # Write corresponding YAML
            dataset_name = case_key.split("/")[0]
            yaml_config = base_config.copy()
            yaml_config["usr_requirment"] = content
            yaml_path = os.path.join(case_root, f"{dataset_name}.yaml")
            with open(yaml_path, "w", encoding="utf-8") as yf:
                yaml.dump(yaml_config, yf, default_flow_style=False, sort_keys=False)
            continue

        # GT_Files content (e.g., Allrun, 0/U, constant/transportProperties)
        full_path = os.path.join(case_path, file_path)
        folder = os.path.dirname(full_path)
        os.makedirs(folder, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

print(f"✅ Reconstruction complete. Output saved in: {output_root}")
