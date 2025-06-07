import os

def gather_py_files(root_dirs):
    """Collect all .py files from the specified root directories."""
    code_blocks = []
    for root_dir in root_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.endswith('.py'):
                    path = os.path.join(dirpath, fname)
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            code_blocks.append(f.read())
                    except Exception as e:
                        print(f"Could not read {path}: {e}")
    return code_blocks

if __name__ == "__main__":
    # Adjust these paths if your local copy is different
    roots = ['../build123d/examples', '../build123d/docs']
    code_blocks = gather_py_files(roots)

    with open("build123d_all_examples.txt", "w", encoding="utf-8") as f:
        for block in code_blocks:
            f.write(block)
            f.write("\n" + "="*60 + "\n")

    print(f"Saved {len(code_blocks)} Python files from {roots}.")
