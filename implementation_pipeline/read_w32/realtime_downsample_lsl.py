import time
import logging
from read_lsl import EEGStreamFilter, STREAM_NAME, ORIGINAL_RATE, DESIRED_CHANNELS, EXPECTED_CHANNELS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealTimeDownsampler:
    """
    Real-time EEG downsampler that processes samples immediately without batching delays.
    """
    
    def __init__(self, stream_name, target_rate, debug=False):
        """
        Initialize the real-time downsampler.
        
        Args:
            stream_name (str): Name of the LSL stream to resolve.
            target_rate (float): Desired downsample rate in Hz.
            debug (bool): Whether to print debug information.
        """
        self.eeg_filter = EEGStreamFilter(stream_name, debug=debug)
        self.target_rate = target_rate
        self.decimation_factor = int(round(ORIGINAL_RATE / target_rate))
        self.sample_counter = 0
        self.debug = debug
        
        # Verify channel count
        actual_channels = len(self.eeg_filter.filtered_channel_names)
        if actual_channels != EXPECTED_CHANNELS:
            raise RuntimeError(f"Channel count mismatch: expected {EXPECTED_CHANNELS}, got {actual_channels}")
        
        if self.debug:
            logger.info(f"Real-time downsampler initialized:")
            logger.info(f"Target rate: {target_rate} Hz, Decimation factor: {self.decimation_factor}")
            logger.info(f"Channels: {len(self.eeg_filter.filtered_channel_names)}")
    
    def get_next_sample(self):
        """
        Get the next downsampled sample in real-time.
        
        Returns:
            tuple: (values, timestamp) if downsampled sample available, None otherwise
        """
        while True:
            values, names, ts = self.eeg_filter.read_sample()
            
            # Only return every decimation_factor-th sample
            if self.sample_counter % self.decimation_factor == 0:
                self.sample_counter += 1
                return values, ts
            
            self.sample_counter += 1
    
    def stream_continuous(self, callback_func, max_samples=None):
        """
        Stream downsampled samples continuously and call callback for each.
        
        Args:
            callback_func: Function to call with (values, timestamp, channels) for each sample
            max_samples (int, optional): Stop after this many samples. None for infinite.
        """
        sample_count = 0
        
        try:
            while max_samples is None or sample_count < max_samples:
                values, timestamp = self.get_next_sample()
                callback_func(values, timestamp, self.eeg_filter.filtered_channel_names)
                sample_count += 1
                
        except KeyboardInterrupt:
            logger.info(f"Streaming stopped. Processed {sample_count} downsampled samples.")
