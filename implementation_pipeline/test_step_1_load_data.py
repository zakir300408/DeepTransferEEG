import os
import unittest
import numpy as np
import pandas as pd

# Adjust the import path if necessary, assuming test_step_1_load_data.py is in the same directory
# as step_1_load_data.py or that the package structure allows this import.
from step_1_load_data import load_custom_epoch_data # Renamed import

# Define the root directory for datasets, similar to the main script
# This ensures tests can locate the data consistently.
# Assuming 'data' directory is at E:\Exoskeleton_DL\DeepTransferEEG\data
# and this test script is in E:\Exoskeleton_DL\DeepTransferEEG\implementation_pipeline
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT_DIR = os.path.join(BASE_PROJECT_DIR, "data")


class TestLoadData(unittest.TestCase):
    def setUp(self):
        self.dataset_name = "CustomEpoch"
        # The load_custom_epoch_data function will check for file existence.
        # We assume that if the files are missing, download_data.py needs to be run for CustomEpoch.
        # For tests to pass, the data must exist at the expected location.
        # Expected location: E:\Exoskeleton_DL\DeepTransferEEG\data\CustomEpoch
        
        # You might want to ensure download_data.py has run and created the files.
        # For simplicity, this test assumes the files are present.
        # If not, load_custom_epoch_data will raise FileNotFoundError, failing the test.
        pass

    def test_load_and_verify_valid_custom_epoch_data(self):
        print(f"\n[Test Case] Running test_load_and_verify_valid_custom_epoch_data for {self.dataset_name}...")
        try:
            # Ensure DATA_ROOT_DIR is used by the function being tested,
            # which it does by its internal definition.
            X, labels, meta = load_custom_epoch_data(self.dataset_name) # Use renamed function
            
            self.assertIsNotNone(X)
            self.assertIsNotNone(labels)
            self.assertIsNotNone(meta)
            
            # Verification assertions (previously in _verify_loaded_data and also here)
            self.assertEqual(X.ndim, 3, f"X should be 3D (trials, channels, samples), but got {X.ndim}D")
            self.assertEqual(labels.ndim, 1, f"labels should be 1D, but got {labels.ndim}D")
            self.assertEqual(X.shape[0], len(labels),
                f"Number of trials in X ({X.shape[0]}) must match number of labels ({len(labels)})")
            
            self.assertTrue('n_trials' in meta.columns, 
                            "'n_trials' column not found in meta.csv. This column is expected for CustomEpoch.")
            if 'n_trials' in meta.columns: # Guarding the sum operation
                self.assertEqual(meta['n_trials'].sum(), X.shape[0],
                    f"Sum of 'n_trials' in meta ({meta['n_trials'].sum()}) must match total trials in X ({X.shape[0]})")

            print("[Test Case] test_load_and_verify_valid_custom_epoch_data PASSED.")
        except FileNotFoundError:
            self.fail(
                f"FileNotFoundError during test: Data for '{self.dataset_name}' not found. "
                f"Ensure download_data.py has been run to generate it in {os.path.join(DATA_ROOT_DIR, self.dataset_name)}."
            )
        except AssertionError as e: # Catch assertion errors from the test's own checks
            self.fail(f"AssertionError during test (data verification failed): {e}")


    def test_load_non_existent_dataset(self):
        print("\n[Test Case] Running test_load_non_existent_dataset...")
        non_existent_dataset = "NonExistentDataset123"
        with self.assertRaises(FileNotFoundError):
            load_custom_epoch_data(non_existent_dataset) # Use renamed function
        print("[Test Case] test_load_non_existent_dataset PASSED (FileNotFoundError correctly raised).")

if __name__ == '__main__':
    print("Running unittests for step_1_load_data.py...")
    unittest.main()
