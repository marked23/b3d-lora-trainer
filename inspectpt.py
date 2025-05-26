import torch
import sys

def inspect_pt_file(filename):
    data = torch.load(filename, map_location='cpu')
    print(f"Top-level type: {type(data)}")
    if isinstance(data, dict):
        print("Top-level keys:", list(data.keys()))
        for k, v in data.items():
            print(f"Key: {k} | Type: {type(v)}")
            # For tensors, print shape and dtype
            if isinstance(v, torch.Tensor):
                print(f"  Tensor shape: {v.shape}, dtype: {v.dtype}")
            # For nested dicts, print their keys
            elif isinstance(v, dict):
                print(f"  Nested dict keys: {list(v.keys())}")
            # For lists, print length
            elif isinstance(v, list):
                print(f"  List length: {len(v)}")
    else:
        print("Data is not a dict. Type:", type(data))
        print(data)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_pt_file.py <file.pt>")
    else:
        inspect_pt_file(sys.argv[1])
