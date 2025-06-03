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


def load_custom_epoch_data(dataset_name: str = "CustomEpoch") -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load X, labels, and metadata from a dataset directory under DATA_ROOT.
    Raises FileNotFoundError if any required file is missing.
    """
    data_dir = DATA_ROOT / dataset_name
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


def log_single_trial(
    dataset_name: str = "CustomEpoch",
    trial_index: int = 0
) -> None:
    """
    Load and log a single trial; if out of bounds, log a skip message.
    Logs errors for missing files (`FileNotFoundError`) or missing metadata columns (`KeyError`).

    Args:
        dataset_name: Name of the dataset folder under DATA_ROOT.
        trial_index: Global trial index across all sessions.
    """
    try:
        X, labels, meta = load_custom_epoch_data(dataset_name)
        if "n_trials" not in meta.columns:
            raise KeyError("meta.csv must include an 'n_trials' column.")

        if not (0 <= trial_index < X.shape[0]):
            raise IndexError()

        # Find session metadata
        cumulative = meta["n_trials"].cumsum().to_numpy()
        session_idx = int(np.searchsorted(cumulative, trial_index, side="right"))
        session_meta = meta.iloc[session_idx]

        data = X[trial_index]
        label = labels[trial_index]

        logger.info(f"--- Trial index {trial_index} ---")
        logger.info(f"Data shape: {data.shape}, Label: {label}")
        logger.info(f"Session/file: {session_meta.get('file', 'N/A')}")
        logger.info("Session metadata:\n%s", session_meta.to_string())

    except IndexError:
        logger.info(f"Skipping trial index {trial_index} (out of bounds).")
    except (FileNotFoundError, KeyError) as e:
        logger.error("Cannot load trial %d: %s", trial_index, e)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info("Starting data loader")

    dataset_name = "CustomEpoch"
    try:
        X, labels, meta = load_custom_epoch_data(dataset_name)

        logger.info(f"Loaded X.shape={X.shape}, labels.shape={labels.shape}")
        logger.info(f"Unique labels: {np.unique(labels)}")
        logger.info("Meta preview:\n%s", meta.head().to_string())

        # Demonstrate logging of a couple of trials
        total_trials = X.shape[0]
        FIRST_EXAMPLE_INDEX = 2  # Example index chosen arbitrarily for demonstration
        example_indices = [FIRST_EXAMPLE_INDEX]
        if len(meta) > 1:
            # index at the end of the first session
            example_indices.append(int(meta["n_trials"].iloc[0]))

        for idx in example_indices:
            # log_single_trial handles out-of-bounds internally
            log_single_trial(dataset_name, trial_index=idx)

    except (FileNotFoundError, KeyError) as e:
        logger.error("Data loading failed: %s", e)
    except Exception:
        logger.exception("Unexpected error during data loading")

    logger.info("Done.")
