import os
import numpy as np
import pandas as pd
import pyvista as pv

def touch_foam_file(directory, file_name):
    foam_path = os.path.join(directory, file_name)
    with open(foam_path, 'w') as f:
        pass
    return foam_path

def calculate_nmse(array1, array2):
    return np.sum((array1 - array2) ** 2) / np.sum(array1 ** 2)

def align_and_scale_mesh(gt_mesh, meta_mesh):
    gt_bounds = np.array(gt_mesh[0].bounds).reshape(3, 2)
    meta_bounds = np.array(meta_mesh[0].bounds).reshape(3, 2)

    if np.allclose(gt_bounds, meta_bounds, atol=1e-6):
        return meta_mesh

    shift = gt_bounds[:, 0] - meta_bounds[:, 0]
    gt_size = gt_bounds[:, 1] - gt_bounds[:, 0]
    meta_size = meta_bounds[:, 1] - meta_bounds[:, 0]
    scaling = np.divide(gt_size, meta_size, out=np.ones_like(gt_size), where=meta_size != 0)

    aligned_meta_mesh = meta_mesh[0].copy()
    aligned_meta_mesh.points = (aligned_meta_mesh.points - meta_bounds[:, 0]) * scaling + gt_bounds[:, 0]
    return [aligned_meta_mesh]

def get_inner_run_folder(path):
    for f in os.listdir(path):
        full = os.path.join(path, f)
        if os.path.isdir(full) and f != "GT_Files":
            return full
    return None

def evaluate_nmse(gt_dir, llm_dir):
    fields = ["U", "p", "rho", "T"]
    gt_foam = touch_foam_file(gt_dir, os.path.basename(gt_dir) + ".foam")
    llm_foam = touch_foam_file(llm_dir, os.path.basename(llm_dir) + ".foam")

    try:
        gt_reader = pv.OpenFOAMReader(gt_foam)
        last_time = gt_reader.time_values[-1]
        gt_reader.set_active_time_value(last_time)
        meta_reader = pv.OpenFOAMReader(llm_foam)
        closest_time = min(meta_reader.time_values, key=lambda x: abs(x - last_time))
        meta_reader.set_active_time_value(closest_time)
    except:
        return 9999

    gt_mesh = gt_reader.read()
    meta_mesh = meta_reader.read()
    meta_mesh = align_and_scale_mesh(gt_mesh, meta_mesh)

    scores = []
    for field in fields:
        if field in gt_mesh[0].cell_data and field in meta_mesh[0].cell_data:
            try:
                if gt_mesh[0].n_cells != meta_mesh[0].n_cells:
                    try:
                        OG_mesh = gt_mesh.copy()
                        if field in OG_mesh[0].point_data:
                            del OG_mesh[0].point_data[field]
                        if field in OG_mesh[0].cell_data:
                            del OG_mesh[0].cell_data[field]
                        projected_mesh = [OG_mesh[0].interpolate(meta_mesh[0], sharpness=0.1, n_points=50)]
                        projected_mesh[0] = projected_mesh[0].point_data_to_cell_data()
                        meta_data = projected_mesh[0][field]
                    except:
                        scores.append(9999)
                        continue
                else:
                    meta_data = meta_mesh[0][field]

                gt_data = gt_mesh[0][field]
                score = calculate_nmse(gt_data, meta_data)
                scores.append(score)
            except:
                scores.append(9999)
    if scores:
        return np.mean(scores)
    else:
        return 9999

def process_all(base_dir, is_advanced=False):
    records = []
    base_path = os.path.join(base_dir, "Advanced" if is_advanced else "Basic")
    datasets = os.listdir(base_path)

    for dataset in datasets:
        dataset_dir = os.path.join(base_path, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        dirs = [1] if is_advanced else range(1, 11)
        for i in dirs:
            case_dir = dataset_dir if is_advanced else os.path.join(dataset_dir, str(i))
            gt_dir = os.path.join(case_dir, "GT_Files")
            llm_dir = get_inner_run_folder(case_dir)
            if not gt_dir or not llm_dir or not os.path.exists(gt_dir) or not os.path.exists(llm_dir):
                score = 9999
            else:
                score = evaluate_nmse(gt_dir, llm_dir)
            records.append([dataset, i, score])
    return pd.DataFrame(records, columns=["Dataset", "Directory", "NMSE"])

# Run for both Basic and Advanced
basic_df = process_all("Dataset", is_advanced=False)
advanced_df = process_all("Dataset", is_advanced=True)

basic_df.to_csv("basic_nmse_report.csv", index=False)
advanced_df.to_csv("advanced_nmse_report.csv", index=False)
