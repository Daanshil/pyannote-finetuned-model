import os
import argparse
import torch
from tqdm import tqdm
from pyannote.database import registry,FileFinder
from pyannote.core import Segment, Timeline

def get_annotated(file):
    # Get the 'extent' (the start and end of all annotations)
    extent = file["annotation"].get_timeline().extent()
    # Wrap it in a Timeline object for the pyannote training loop
    return Timeline([Segment(0, extent.end)])


def main():
    # 1. Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Verify pyannote database.yml setup and count split metrics."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Absolute or relative path to the database.yml file."
    )
    parser.add_argument(
        "--wav-dir", 
        type=str, 
        required=True, 
        help="Absolute path to the directory containing the .wav files."
    )
    parser.add_argument(
        "--protocol-name", 
        type=str, 
        required=True, 
        help="Name of the protocol to analyze."
    )
    
    args = parser.parse_args()

    # 2. Check if the configuration file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at '{args.config}'")
        return

    # 3. Dynamically define the audio path preprocessor using the argument
    def absolute_audio_path(current_file):
        uri = current_file['uri']
        # Combines the user-provided directory with the URI and extension
        return os.path.join(args.wav_dir, f"{uri}.wav")

    # 4. Load the database configuration
    print(f"Loading database configuration from: {args.config}")
    registry.load_database(args.config)

    # 5. Extract the protocol
    protocol_name = args.protocol_name
    try:
        protocol = registry.get_protocol(
            protocol_name, 
            preprocessors={"audio": FileFinder(),"annotated": get_annotated}
        )
    except ValueError as e:
        print(f"Error loading protocol '{protocol_name}': {e}")
        print("Available protocols in registry:", registry.protocols)
        return

    # 6. Define splits to analyze
    splits = {
        "Train": protocol.train,
        "Development": protocol.development,
        "Test": protocol.test
    }

    print("\n" + "="*40)
    print("       DATABASE PROTOCOL METRICS       ")
    print("="*40)

    # 7. Gather and print metrics
    for split_name, split_generator in splits.items():
        try:
            # Convert generator to a list to count and analyze
            file_list = list(split_generator())
            
            audio_file_count = len(file_list)
            total_annotations = 0
            
            for file in file_list:
                if 'annotation' in file and file['annotation'] is not None:
                    total_annotations += len(file['annotation'])
            
            print(f"[{split_name} Split]")
            print(f"  - Number of audio files: {audio_file_count}")
            print(f"  - Total individual annotations (turns): {total_annotations}")
            print("-" * 40)
            
        except Exception as e:
            print(f"[{split_name} Split] Error processing: {e}")
            print("-" * 40)

if __name__ == "__main__":
    main()