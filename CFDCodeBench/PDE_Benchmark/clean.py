import os
import shutil
import glob


def make_clean(root_dir):
    """
    Delete all generated folders and `.npy` files to return to a clean state.
    Similar to `make clean` in C/C++ projects.
    """
    # Folders to remove
    generated_dirs = [
        "solver",
        "results/prediction",
        "report",
        "compare",
        "compare_images",
        "table",
        "image",
        "convergent"
    ]

    print("🧹 Cleaning generated folders...\n")

    for subdir in generated_dirs:
        full_path = os.path.join(root_dir, subdir)
        if os.path.exists(full_path):
            try:
                shutil.rmtree(full_path)
                print(f"✅ Removed directory: {full_path}")
            except Exception as e:
                print(f"❌ Failed to remove {full_path}: {e}")
        else:
            print(f"ℹ️ Skipped (not found): {full_path}")

    # Delete all .npy files in root_dir (non-recursive)
    npy_files = glob.glob(os.path.join(root_dir, "*.npy"))
    for file_path in npy_files:
        try:
            os.remove(file_path)
            print(f"🗑️ Deleted file: {file_path}")
        except Exception as e:
            print(f"❌ Failed to delete {file_path}: {e}")

    # Delete prompt/prompt.json
    prompt_json_path = os.path.join(root_dir, "prompt", "prompts.json")
    if os.path.exists(prompt_json_path):
        try:
            os.remove(prompt_json_path)
            print(f"🗑️ Deleted file: {prompt_json_path}")
        except Exception as e:
            print(f"❌ Failed to delete {prompt_json_path}: {e}")
    else:
        print(f"ℹ️ Skipped (not found): {prompt_json_path}")

    print("\n✨ Done. Project workspace is clean.")


make_clean("/opt/CFD-Benchmark/PDE_Benchmark")
