


def verify_sampling_rate(stream_name=None, target_rate=100.0, num_samples=50):
    """
    Verify the actual sampling rate and timing consistency.
    
    Args:
        stream_name (str): Name of the LSL stream
        target_rate (float): Target sampling rate
        num_samples (int): Number of samples for verification
    """
    from realtime_downsample_lsl import RealTimeDownsampler
    
    downsampler = RealTimeDownsampler(stream_name, target_rate, debug=True)
    
    # Collect timestamps for rate verification - only from downsampled samples
    timestamps = []
    print(f"Collecting {num_samples} downsampled samples for rate verification...")
    
    for i in range(num_samples):
        values, timestamp = downsampler.get_next_sample()  # This already returns only downsampled samples
        timestamps.append(timestamp)
        
        if (i + 1) % 10 == 0:
            print(f"Collected {i + 1}/{num_samples} downsampled samples...")
    
    # Calculate actual sampling rate
    if len(timestamps) > 1:
        time_span = timestamps[-1] - timestamps[0]
        actual_rate = (len(timestamps) - 1) / time_span
        
        # Calculate inter-sample intervals for consistency check
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        avg_interval = sum(intervals) / len(intervals)
        expected_interval = 1.0 / target_rate
        
        print(f"\n{'='*60}")
        print(f"DOWNSAMPLING VERIFICATION RESULTS")
        print(f"{'='*60}")
        print(f"Target rate:           {target_rate} Hz")
        print(f"Actual rate:           {actual_rate:.2f} Hz")
        print(f"Expected interval:     {expected_interval*1000:.2f} ms")
        print(f"Actual avg interval:   {avg_interval*1000:.2f} ms")
        print(f"Time span:             {time_span:.3f} seconds")
        print(f"Samples collected:     {len(timestamps)}")
        print(f"Decimation factor:     {downsampler.decimation_factor}")
        print(f"Rate deviation:        {abs(actual_rate - target_rate) / target_rate * 100:.1f}%")
        
        # Check interval consistency
        interval_std = (sum([(x - avg_interval)**2 for x in intervals]) / len(intervals))**0.5
        print(f"Interval std dev:      {interval_std*1000:.2f} ms")
        
        if abs(actual_rate - target_rate) / target_rate < 0.05:  # Within 5%
            print("✓ Downsampling rate verification PASSED")
        else:
            print("✗ Downsampling rate verification FAILED")
        
        if interval_std < 0.01:  # Less than 10ms standard deviation
            print("✓ Timing consistency verification PASSED")
        else:
            print("✗ Timing consistency verification FAILED")
        print(f"{'='*60}\n")