#!/usr/bin/env python3
"""
Reorganize .mat files from nested subdirectories under Data_Epoch
into their own folders in the root Data_Epoch directory.
"""

import os
import shutil
from pathlib import Path

def main():
    # Define the root Data_Epoch directory
    root_dir = Path(r"E:\Exoskeleton_DL\XK_work\Data_Epoch")

    # Counters for summary
    processed_count = 0
    errors = []

    # Gather all .mat files under root_dir (before moving)
    mat_files = list(root_dir.rglob("*.mat"))

    for source_path in mat_files:
        try:
            # Ensure it's a file
            if not source_path.is_file():
                continue

            # Compute the new folder name (filename without extension)
            folder_name = source_path.stem
            dest_folder = root_dir / folder_name

            # Create destination folder if it doesn't exist
            dest_folder.mkdir(parents=True, exist_ok=True)

            # Define destination file path
            dest_file = dest_folder / source_path.name

            # If a file already exists at destination, overwrite it
            if dest_file.exists():
                dest_file.unlink()

            # Move the .mat file into its new folder
            shutil.move(str(source_path), str(dest_file))

            processed_count += 1

        except Exception as e:
            # Record any errors for the summary
            errors.append((str(source_path), str(e)))

    # Print summary
    print(f"Total .mat files processed: {processed_count}")
    if errors:
        print("Errors encountered:")
        for file_path, error_msg in errors:
            print(f" - {file_path}: {error_msg}")

if __name__ == "__main__":
    main()
