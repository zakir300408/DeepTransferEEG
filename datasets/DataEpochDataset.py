import os
from scipy.io import loadmat

def inspect_mat_file(path):
    mat = loadmat(path)
    shapes = {k: v.shape for k, v in mat.items() if not k.startswith('__')}
    print(f"\nFile: {os.path.basename(path)}")
    for name, shp in shapes.items():
        print(f"  {name}: shape={shp}")
    return set(shapes.keys())

def inspect_dataset(root_folder):
    all_keys = set()
    print(f"Scanning folder: {root_folder}")
    for fn in sorted(os.listdir(root_folder)):
        if fn.lower().endswith('.mat'):
            full = os.path.join(root_folder, fn)
            keys = inspect_mat_file(full)
            all_keys |= keys
    print("\nSummary of all keys in dataset:", all_keys)
    return all_keys

def print_assumptions():
    print("\nAssumptions before adopting MOABB‐style API:")
    print("1) Every .mat contains 'MyEpoch' (n_trials, n_samples, n_channels).")
    print("2) Every .mat contains 'MyLabel' as shape (1, n_trials) or (n_trials,).")
    print("3) Sampling rate is constant across files/trials.")
    print("4) Time window and channel order match downstream preprocessing.")
    print("5) Labels are integers starting at 0 or 1, with no gaps.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Inspect and print assumptions for your .mat EEG epoch files"
    )
    parser.add_argument(
        '--data_folder',
        default=r"E:\Exoskeleton_DL\XK_work\Data_Epoch",
        help="folder containing .mat epoch files"
    )
    args = parser.parse_args()

    keys = inspect_dataset(args.data_folder)
    print_assumptions()
