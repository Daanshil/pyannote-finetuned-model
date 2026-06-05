import os
import argparse
from pyannote.database import registry, FileFinder
from pyannote.core import Segment, Timeline

def get_annotated(file):
    # Get the 'extent' (the start and end of all annotations)
    extent = file["annotation"].get_timeline().extent()
    # Wrap it in a Timeline object for the pyannote training loop
    return Timeline([Segment(0, extent.end)])

class CombinedProtocol:
    name = "Combined_Dataset"
    
    def __init__(self, protocols):
        self.protocols = protocols
        
    def train(self):
        for p in self.protocols:
            yield from p.train()
            
    def development(self):
        for p in self.protocols:
            yield from p.development()
            
    def test(self):
        for p in self.protocols:
            yield from p.test()

def main():
    parser = argparse.ArgumentParser(description="Initialize and verify a multi-protocol dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to multi.yml configuration file")
    
    # Accept a space-separated list of protocols from the CLI
    parser.add_argument(
        "--protocols", 
        type=str, 
        nargs="+", 
        required=True, 
        help="Space-separated list of pyannote protocol names to combine"
    )
    
    args = parser.parse_args()

    # 1. Validation check on configuration file
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
        return

    # 2. Load the unified YAML database definition
    print(f"Loading configuration database: {args.config}")
    registry.load_database(args.config)

    # 3. Dynamic initialization loop for multiple protocols
    loaded_protocols = []
    
    # Preprocessors definition using native FileFinder linked to multi.yml
    preprocessors = {
        "audio": FileFinder(), 
        "annotated": get_annotated
    }

    print("Initializing protocol sub-components...")
    for proto_name in args.protocols:
        try:
            # CORRECTION: Initializing protocol inside the loop scope where proto_name exists
            proto = registry.get_protocol(proto_name, preprocessors=preprocessors)
            loaded_protocols.append(proto)
            print(f"  Successfully registry-loaded: {proto_name}")
        except ValueError as e:
            print(f"Error loading individual protocol '{proto_name}': {e}")
            print(f"Available database protocols are: {registry.protocols}")
            return

    # 4. Wrap loaded generator layers inside your composite generator class
    print(f"Fusing {len(loaded_protocols)} protocols inside CombinedProtocol space...")
    dataset = CombinedProtocol(loaded_protocols)

    # 5. Pipeline sanity check verification step
    try:
        # Check training generation
        train_sample = next(dataset.train())
        print(f"Sanity Check Train item: {train_sample['uri']}")
        train_sample = next(dataset.train())
        print(f"Sanity Check Train item: {train_sample['a']}")
        
        # Check testing generation
        test_sample = next(dataset.test())
        print(f"Sanity Check Test item: {test_sample['uri']}")
        
        print("Success: All components mapped and verified properly.")
    except StopIteration:
        print("Error: CombinedProtocol loaded successfully, but one or more target split generators are empty.")
    except Exception as e:
        print(f"Verification Error: {e}")

if __name__ == "__main__":
    main()