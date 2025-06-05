from pylsl import StreamInlet, resolve_byprop

DESIRED_CHANNELS = {
    "FP1","FZ","F3","F7","FC5","FC1","C3","T7",
    "CP5","CP1","PZ","P3","P7","O1","O2","P4",
    "P8","CP6","CP2","CZ","C4","T8","FC6","FC2",
    "F4","F8","FP2"
}
STREAM_NAME = "iReW32_39"

class EEGStreamFilter:
    """
    Initialize by resolving the named EEG stream and selecting only the desired channels
    (preserving their original stream order). Then use `read_sample()` to pull one filtered
    sample at a time without re-resolving or re-indexing.
    """

    def __init__(self, stream_name):
        # 1) Resolve the stream by name (only once)
        streams = resolve_byprop('name', stream_name)
        if not streams:
            raise RuntimeError(f"No EEG stream found with name: {stream_name}")
        self.inlet = StreamInlet(streams[0])

        # 2) Extract all channel names exactly in the order provided by the stream
        info = self.inlet.info()
        n_ch = info.channel_count()
        chan_xml = info.desc().child("channels").child("channel")

        self.stream_channel_names = []
        for _ in range(n_ch):
            self.stream_channel_names.append(chan_xml.child_value("label"))
            chan_xml = chan_xml.next_sibling()

        # 3) Define the set of 27 desired channels (uppercase for matching)
        desired = DESIRED_CHANNELS

        # 4) Build a lookup from uppercase stream label → index
        stream_upper_to_idx = {
            name.upper(): idx
            for idx, name in enumerate(self.stream_channel_names)
        }

        # 5) Walk through the original stream order and keep any channel that is in `desired`
        self.keep_idx = []
        self.filtered_channel_names = []
        for idx, name in enumerate(self.stream_channel_names):
            if name.upper() in desired:
                self.keep_idx.append(idx)
                self.filtered_channel_names.append(name)

        # 6) Verify that no desired channel is missing
        missing = [ch for ch in desired if ch not in stream_upper_to_idx]
        if missing:
            raise RuntimeError(f"Missing channels in the stream: {missing}")

        # Optional: print summary once at initialization
        print(f"Stream channel names (total {n_ch}): {self.stream_channel_names}")
        print(f"Sampling rate: {info.nominal_srate()}, Stream type: {info.type()}")
        print(f"Requested {len(desired)} channels, Found {len(self.keep_idx)}.")
        print("Streaming only these channels (original order):")
        print(self.filtered_channel_names)
        print("---- Initialization complete, calling `read_sample()` will return filtered data. ----\n")

    def read_sample(self):
        """
        Pull one sample from the inlet and return only the filtered channels.
        
        Returns:
            filtered_values: list of floats (in the same order as filtered_channel_names)
            filtered_channel_names: list of str
            timestamp: float
        """
        sample, timestamp = self.inlet.pull_sample()
        filtered_values = [sample[i] for i in self.keep_idx]
        return filtered_values, self.filtered_channel_names, timestamp


if __name__ == "__main__":
    try:
        # Instantiate once (resolves stream and selects channels)
        eeg_filter = EEGStreamFilter(STREAM_NAME)

        # Example: continue pulling samples in a loop
        while True:
            values, names, ts = eeg_filter.read_sample()
            print(f"Timestamp: {ts:.6f}")
            for ch, val in zip(names, values):
                print(f"  {ch}: {val}")
            print()
    except RuntimeError as e:
        print("Error:", e)
