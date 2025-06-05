import time
from read_lsl import EEGStreamFilter, STREAM_NAME

def downsample_lsl(stream_name, decimation_factor=5, max_downsamples=100):
    """
    Reads samples from an EEGStreamFilter at 500 Hz, downsamples to 100 Hz by
    taking every `decimation_factor`-th sample, and performs sanity checks on
    timestamps to confirm the effective sampling rate. Prints the downsampled
    data (timestamp + channel values) in the same channel‐order structure.
    
    Args:
        stream_name (str): Name of the LSL stream to resolve.
        decimation_factor (int): How many original samples to skip before taking one.
                                 For 500 → 100 Hz, use 5.
        max_downsamples (int): Number of downsampled points to collect before stopping.
    """
    # 1) Instantiate the filter (resolves stream, selects channels, etc.) once.
    eeg_filter = EEGStreamFilter(stream_name)
    filtered_channel_names = eeg_filter.filtered_channel_names  # list of 27 channels

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

    # 4) Sanity check: original sampling rate ≈ 500 Hz
    orig_deltas = [
        t2 - t1
        for t1, t2 in zip(original_timestamps[:-1], original_timestamps[1:])
    ]
    avg_orig_delta = sum(orig_deltas) / len(orig_deltas)
    est_orig_rate = 1.0 / avg_orig_delta if avg_orig_delta > 0 else float('inf')

    # 5) Sanity check: downsampled rate ≈ 100 Hz
    down_deltas = [
        t2 - t1
        for t1, t2 in zip(down_timestamps[:-1], down_timestamps[1:])
    ]
    avg_down_delta = sum(down_deltas) / len(down_deltas)
    est_down_rate = 1.0 / avg_down_delta if avg_down_delta > 0 else float('inf')

    # 6) Validate that we indeed went from ~500 Hz to ~100 Hz
    if not (480 < est_orig_rate < 520):
        raise RuntimeError(
            f"Original sampling rate check failed: estimated {est_orig_rate:.1f} Hz"
        )
    if not (95 < est_down_rate < 105):
        raise RuntimeError(
            f"Downsampled rate check failed: estimated {est_down_rate:.1f} Hz"
        )

    # 7) Print summary of sanity checks
    print(f"Estimated original rate: {est_orig_rate:.1f} Hz → OK")
    print(f"Estimated downsampled rate: {est_down_rate:.1f} Hz → OK\n")

    # 8) Print the final downsampled data
    # Print header of channel names once
    print("Downsampled data (timestamp + values):")
    print("Channels:", filtered_channel_names)
    print("----")
    
    for ts, vals in zip(down_timestamps, down_values):
        # Print timestamp and the list of values in the same order as filtered_channel_names
        vals_str = ", ".join(f"{v:.3f}" for v in vals)
        print(f"{ts:.6f} → [{vals_str}]")

if __name__ == "__main__":
    # Run the downsampling routine and print 100 downsampled samples
    try:
        downsample_lsl(STREAM_NAME, decimation_factor=5, max_downsamples=100)
    except RuntimeError as err:
        print("Error during downsampling:", err)
