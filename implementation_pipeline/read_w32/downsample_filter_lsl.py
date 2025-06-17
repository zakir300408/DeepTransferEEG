import time
import logging
from read_lsl import EEGStreamFilter, STREAM_NAME

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define sampling‐rate and tolerance constants
ORIGINAL_RATE        = 500.0
DEFAULT_TARGET_RATE  = 200.0
DEFAULT_MAX_DOWNSAMP = 100
ORIG_TOLERANCE       = 0.04   # ±4%
DS_TOLERANCE         = 0.05   # ±5%
EXPECTED_CHANNELS    = 31     # Expected number of channels


def downsample_lsl(stream_name, target_rate, max_downsamples):
    """
    Reads samples from an EEGStreamFilter at 500 Hz, downsamples to `target_rate`
    by selecting every Nth sample (computed), and performs sanity checks on
    timestamps. Returns the downsampled data as a dictionary.
    
    Args:
        stream_name (str): Name of the LSL stream to resolve.
        target_rate (float): Desired downsample rate in Hz.
        max_downsamples (int): Number of downsampled points to collect.
    

    Example:
        Input: stream_name="EEG_Stream", target_rate=200.0, max_downsamples=5
        Output: {
            'timestamps': [1234567890.123, 1234567890.128, 1234567890.133, ...],
            'values': [[0.001, -0.002, 0.003, ...], [0.004, -0.001, 0.002, ...], ...],
            'channels': ['Fp1', 'Fp2', 'F3', 'F4', ..., 'O2'],
            'metadata': {
                'original_rate': 500.2,
                'downsampled_rate': 199.8,
                'target_rate': 200.0,
                'decimation_factor': 3,
                'num_samples': 5,
                'num_channels': 31
            }
        }
    
    Returns:
        dict: Dictionary containing 'timestamps', 'values', 'channels', and 'metadata'
    """
    # 1) Instantiate the filter (resolves stream, selects channels, etc.)
    eeg_filter = EEGStreamFilter(stream_name)
    filtered_channel_names = eeg_filter.filtered_channel_names  # list of channels

    # Sanity check: verify channel count
    _sanity_check_channels(filtered_channel_names)

    # compute decimation factor for the known original rate
    orig_rate_expected = ORIGINAL_RATE
    decimation_factor   = int(round(orig_rate_expected / target_rate))

    # 2) Prepare containers for original and downsampled timestamps/values
    original_timestamps = []
    down_timestamps = []
    down_values = []

    # 3) Loop until we've collected enough downsampled frames
    sample_counter = 0
    while len(down_timestamps) < max_downsamples:
        values, names, ts = eeg_filter.read_sample()
        # Record every timestamp from the 500 Hz stream
        original_timestamps.append(ts)
        
        # Only keep every `decimation_factor`-th sample
        if sample_counter % decimation_factor == 0:
            down_timestamps.append(ts)
            down_values.append(values)
        sample_counter += 1

    # 4) Calculate rates and perform sanity checks
    est_orig_rate, est_down_rate = _calculate_rates(original_timestamps, down_timestamps)
    _sanity_check_rates(orig_rate_expected, est_orig_rate, target_rate, est_down_rate)

    # 5) Create and return the final dictionary
    result = {
        'timestamps': down_timestamps,
        'values': down_values,
        'channels': filtered_channel_names,
        'metadata': {
            'original_rate': est_orig_rate,
            'downsampled_rate': est_down_rate,
            'target_rate': target_rate,
            'decimation_factor': decimation_factor,
            'num_samples': len(down_timestamps),
            'num_channels': len(filtered_channel_names)
        }
    }
    
    logger.info("Downsampled data collected successfully:")
    logger.info(f"Channels: {filtered_channel_names}")
    logger.info(f"Samples collected: {len(down_timestamps)}")
    
    return result

def _calculate_rates(original_timestamps, down_timestamps):
    """
    Calculate estimated sampling rates from timestamp arrays.
    
    Returns:
        tuple: (est_orig_rate, est_down_rate)
    """
    # Calculate original sampling rate
    orig_deltas = [
        t2 - t1
        for t1, t2 in zip(original_timestamps[:-1], original_timestamps[1:])
    ]
    avg_orig_delta = sum(orig_deltas) / len(orig_deltas)
    est_orig_rate = 1.0 / avg_orig_delta if avg_orig_delta > 0 else float('inf')

    # Calculate downsampled rate
    down_deltas = [
        t2 - t1
        for t1, t2 in zip(down_timestamps[:-1], down_timestamps[1:])
    ]
    avg_down_delta = sum(down_deltas) / len(down_deltas)
    est_down_rate = 1.0 / avg_down_delta if avg_down_delta > 0 else float('inf')
    
    return est_orig_rate, est_down_rate

def _sanity_check_channels(filtered_channel_names):
    """
    Verifies that the number of filtered channels matches expectations.
    
    Args:
        filtered_channel_names (list): List of channel names from the filter.
    
    Raises:
        RuntimeError: If channel count doesn't match expected value.
    """
    actual_channels = len(filtered_channel_names)
    if actual_channels != EXPECTED_CHANNELS:
        raise RuntimeError(f"Channel count mismatch: expected {EXPECTED_CHANNELS}, got {actual_channels}")
    logger.info(f"Channel count check: {actual_channels} channels → OK")

def _sanity_check_rates(orig_rate_expected, est_orig_rate, target_rate, est_down_rate):
    """
    Raises RuntimeError if estimated rates deviate beyond tolerances,
    and logs a summary of the checked rates.
    """
    # original‐rate sanity
    low_o, high_o = orig_rate_expected * (1 - ORIG_TOLERANCE), orig_rate_expected * (1 + ORIG_TOLERANCE)
    if not (low_o < est_orig_rate < high_o):
        raise RuntimeError(f"Original sampling rate check failed: estimated {est_orig_rate:.1f} Hz")
    # downsampled‐rate sanity
    low_d, high_d = target_rate * (1 - DS_TOLERANCE), target_rate * (1 + DS_TOLERANCE)
    if not (low_d < est_down_rate < high_d):
        raise RuntimeError(f"Downsampled rate check failed: estimated {est_down_rate:.1f} Hz, target {target_rate} Hz")
    
    logger.info(f"Estimated original rate: {est_orig_rate:.1f} Hz → OK")
    logger.info(f"Estimated downsampled rate: {est_down_rate:.1f} Hz (target {target_rate} Hz) → OK\n")

if __name__ == "__main__":
    try:
        result = downsample_lsl(STREAM_NAME, DEFAULT_TARGET_RATE, DEFAULT_MAX_DOWNSAMP)
        
        # Optional: log first few samples for verification
        logger.info("Sample data preview:")
        for i, (ts, vals) in enumerate(zip(result['timestamps'][:3], result['values'][:3])):
            vals_str = ", ".join(f"{v:.3f}" for v in vals)
            logger.info(f"  {ts:.6f} → [{vals_str}]")
        
    except RuntimeError as err:
        logger.error(f"Error during downsampling: {err}")
