import os
from pathlib import Path
import numpy as np
import pandas as pd
import logging

# Allow override via environment variable; otherwise assume project root is one level up
PROJECT_DIR = Path(os.getenv("PROJECT_ROOT", Path(__file__).parent.parent)).resolve()
DATA_ROOT = PROJECT_DIR / "data"

X_FILENAME = "X.npy"
LABELS_FILENAME = "labels.npy"
META_FILENAME = "meta.csv"

logger = logging.getLogger(__name__)


def _load_data_from_files(data_dir: Path, dataset_name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load X, labels, and metadata from a dataset directory.
    Raises FileNotFoundError if any required file is missing.
    """
    paths = {
        "X": data_dir / X_FILENAME,
        "labels": data_dir / LABELS_FILENAME,
        "meta": data_dir / META_FILENAME,
    }

    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        missing_str = ", ".join(f"'{n}'" for n in missing)
        raise FileNotFoundError(
            f"Missing {missing_str} in {data_dir!s}. "
            f"Place '{dataset_name}' data under {DATA_ROOT!s}."
        )

    X = np.load(paths["X"])
    labels = np.load(paths["labels"])
    meta = pd.read_csv(paths["meta"])
    return X, labels, meta


def load_custom_epoch_data(dataset_name: str = "CustomEpoch") -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load data for the given dataset folder under DATA_ROOT.
    Returns (X, labels, meta).
    """
    data_dir = DATA_ROOT / dataset_name
    return _load_data_from_files(data_dir, dataset_name)


def load_single_trial(
    dataset_name: str = "CustomEpoch",
    trial_index: int = 0
) -> tuple[np.ndarray, np.generic, pd.Series]:
    """
    Load a single trial, its label, and its session metadata.

    Args:
        dataset_name: Name of the dataset folder under DATA_ROOT.
        trial_index: Global trial index across all sessions.

    Returns:
        trial_data: np.ndarray of shape (channels, samples)
        trial_label: scalar label for that trial
        session_meta_row: pd.Series for the session containing this trial

    Raises:
        FileNotFoundError: if data files are missing.
        IndexError: if trial_index is out of bounds.
        KeyError: if 'n_trials' column is missing.
    """
    X, labels, meta = load_custom_epoch_data(dataset_name)

    if not (0 <= trial_index < X.shape[0]):
        raise IndexError(
            f"Trial index {trial_index} is out of bounds for '{dataset_name}' ({X.shape[0]} trials)."
        )

    if "n_trials" not in meta.columns:
        raise KeyError("meta.csv must include an 'n_trials' column.")

    cumulative = meta["n_trials"].cumsum().to_numpy()
    session_idx = int(np.searchsorted(cumulative, trial_index, side="right"))
    session_meta_row = meta.iloc[session_idx]

    trial_data = X[trial_index]
    trial_label = labels[trial_index]
    return trial_data, trial_label, session_meta_row


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Starting data loader")

    dataset_name = "CustomEpoch"
    logger.info(f"Loading '{dataset_name}'")
    try:
        X, labels, meta = load_custom_epoch_data(dataset_name)
        logger.info(f"Loaded X.shape={X.shape}, labels.shape={labels.shape}")
        logger.info(f"Unique labels: {np.unique(labels)}")
        logger.info("Meta (first 5 rows):\n%s", meta.head().to_string())

        if X.shape[0] > 0:
            # Example 1: Load an early trial (e.g., index 320)
            trial_idx_1 = 320
            if trial_idx_1 < X.shape[0]:
                logger.info(f"\n--- Loading single trial at index {trial_idx_1} ---")
                data, label, session_meta = load_single_trial(dataset_name, trial_idx_1)
                logger.info(f"Trial data shape: {data.shape}, Label: {label}")
                logger.info(f"Session/file: {session_meta.get('file', 'N/A')}")
                logger.info("Session metadata:\n%s", session_meta.to_string())
            else:
                logger.info(f"Skipping example for trial index {trial_idx_1} (out of bounds).")

            # Example 2: Load the first trial of the second session, if it exists
            if len(meta) > 1: # Check if there is more than one session
                first_session_n_trials = int(meta["n_trials"].iloc[0])
                trial_idx_2 = first_session_n_trials # Index of the first trial of the second session
                
                if trial_idx_2 < X.shape[0]: # Check if this trial index is valid
                    logger.info(f"\n--- Loading single trial at index {trial_idx_2} (start of second session) ---")
                    data, label, session_meta = load_single_trial(dataset_name, trial_idx_2)
                    logger.info(f"Trial data shape: {data.shape}, Label: {label}")
                    logger.info(f"Session/file: {session_meta.get('file', 'N/A')}")
                    logger.info("Session metadata:\n%s", session_meta.to_string())
                else:
                    logger.info(f"Skipping example for trial index {trial_idx_2} (out of bounds or not enough trials for a second session example).")
        else:
            logger.info(f"Dataset '{dataset_name}' is empty; skipping single trial examples.")

    except FileNotFoundError as e:
        logger.error("Data loading failed: %s", e)
    except IndexError as e:
        logger.error("Single trial loading failed: %s", e)
    except KeyError as e:
        logger.error("Metadata error: %s", e)
    except Exception:
        logger.exception("Unexpected error during data loading")

    logger.info("To run tests: cd to project directory and run `python -m unittest`")
