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

    except (FileNotFoundError, KeyError) as e:
        logger.error("Data loading failed: %s", e)
    except Exception:
        logger.exception("Unexpected error during data loading")

    logger.info("Done.")
