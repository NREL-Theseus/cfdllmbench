import os


def should_skip(filename):
    skip_extensions = [".log", ".csv"]
    return any(filename.endswith(ext) for ext in skip_extensions)


def generate_tree_limited_depth(root_dir, max_depth=2, current_depth=0, prefix=""):
    tree_str = ""
    if current_depth >= max_depth:
        return tree_str

    try:
        entries = sorted(os.listdir(root_dir))
    except Exception:
        return ""

    # except unnecessary files
    entries = [e for e in entries if not should_skip(e)]
    pointers = ["├── "] * (len(entries) - 1) + ["└── "] if entries else []

    for pointer, name in zip(pointers, entries):
        path = os.path.join(root_dir, name)
        tree_str += f"{prefix}{pointer}{name}\n"

        if os.path.isdir(path):
            extension = "│   " if pointer == "├── " else "    "
            tree_str += generate_tree_limited_depth(path, max_depth, current_depth + 1, prefix + extension)

    return tree_str


if __name__ == "__main__":
    root = "."  # current folder
    tree_output = generate_tree_limited_depth(root)

    # print or save as Markdown
    print("```\n" + tree_output + "```")

    with open("folder_structure.md", "w") as f:
        f.write("```\n")
        f.write(tree_output)
        f.write("```\n")
