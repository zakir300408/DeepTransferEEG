# write code to simply check available lsl streams and print their names and types

from pylsl import resolve_streams
import time

def check_lsl_streams():
    """Check and display all available LSL streams."""
    print("Searching for LSL streams...")
    print("-" * 50)
    
    try:
        # Wait a few seconds to discover streams
        streams = resolve_streams(wait_time=3.0)
        
        if not streams:
            print("No LSL streams found.")
            return
        
        print(f"Found {len(streams)} LSL stream(s):")
        print()
        
        for i, stream in enumerate(streams, 1):
            print(f"Stream {i}:")
            print(f"  Name: {stream.name()}")
            print(f"  Type: {stream.type()}")
            print(f"  Channel count: {stream.channel_count()}")
            print(f"  Sampling rate: {stream.nominal_srate()} Hz")
            print(f"  Source ID: {stream.source_id()}")
            print(f"  Hostname: {stream.hostname()}")
            print()
            
            # Try to get additional info from stream info XML
            try:
                info = stream.info()
                print(f"  Additional info: {info}")
                print()
            except Exception as e:
                print(f"  Could not retrieve additional info: {e}")
                print()
                
    except Exception as e:
        print(f"Error while searching for streams: {e}")

if __name__ == "__main__":
    check_lsl_streams()