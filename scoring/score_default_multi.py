import os
import argparse
from pathlib import Path
from tqdm import tqdm
import torch

from pyannote.database import registry, FileFinder
from pyannote.audio import Pipeline
from pyannote.core import Segment, Timeline

from pyannote.metrics.diarization import (
    DiarizationErrorRate, 
    JaccardErrorRate,
    DiarizationPurity,
    DiarizationCoverage
)

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

def get_annotated(file):
    extent = file["annotation"].get_timeline().extent()
    return Timeline([Segment(0, extent.end)])

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple pyannote metrics on a baseline pipeline with combined protocols.")
    parser.add_argument("--config", type=str, required=True, help="Path to multi.yml configuration file")
    parser.add_argument(
        "--protocols", 
        type=str, 
        nargs="+", 
        required=True, 
        help="Space-separated list of pyannote protocol names to combine"
    )
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API Token")
    parser.add_argument("--exp-label", type=str, required=True, help="Experiment label")
    parser.add_argument("--set", type=str, choices=["train", "development", "test"], default="test", help="Split to evaluate")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--output-dir", type=str, default="output", help="Root directory for saving evaluation results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    # 1. Output setup
    base_output_path = Path(args.output_dir)
    output_dir = base_output_path / args.exp_label
    split_dir = output_dir / "baseline" / args.set
    split_dir.mkdir(parents=True, exist_ok=True)

    # 2. Pyannote Database Setup via multi.yml
    if not os.path.exists(args.config):
        print(f"Error: {args.config} not found.")
        return

    print(f"Loading configuration database: {args.config}")
    registry.load_database(args.config)
    
    # Dynamic evaluation protocol instantiation
    loaded_protocols = []
    preprocessors = {
        "audio": FileFinder(),
        "annotated": get_annotated
    }

    print("Initializing protocol sub-components...")
    for proto_name in args.protocols:
        try:
            proto = registry.get_protocol(proto_name, preprocessors=preprocessors)
            loaded_protocols.append(proto)
            print(f"  Successfully loaded protocol component: {proto_name}")
        except ValueError as e:
            print(f"Error loading individual protocol '{proto_name}': {e}")
            return

    # Fuse dataset splits together
    protocol = CombinedProtocol(loaded_protocols)

    # 3. Load Baseline Pipeline
    print("Loading pyannote/speaker-diarization-3.1...")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=args.token)
        pipeline.to(device)
    except Exception as e:
        print(f"HF Error: {e}")
        return

    # 4. Extract targeted joint split sequence
    split_generators = {"train": protocol.train, "development": protocol.development, "test": protocol.test}
    dataset_split = list(split_generators[args.set]())

    if not dataset_split:
        print(f"No files found in combined '{args.set}' split.")
        return

    # 5. Initialize Metrics Collection
    metrics = {
        "Diarization Error Rate (DER)": DiarizationErrorRate(),
        "Jaccard Error Rate (JER)": JaccardErrorRate(),
        "Diarization Purity": DiarizationPurity(),
        "Diarization Coverage": DiarizationCoverage()
    }

    # 6. Process & Evaluate
    print(f"\nEvaluating combined {args.set.upper()} split ({len(dataset_split)} files)...")
    for file in tqdm(dataset_split, desc=f"Processing {args.set}"):
        uri = file["uri"]
        try:
            raw_output = pipeline(file, num_speakers=2)
            hypothesis = getattr(raw_output, "speaker_diarization", raw_output)

            # Save RTTM file
            with open(split_dir / f"{uri}.rttm", "w") as rttm_file:
                hypothesis.write_rttm(rttm_file)

            # Accumulate across all defined metrics
            reference = file["annotation"]
            uem = file.get("annotated")
            for metric in metrics.values():
                metric(reference, hypothesis, uem=uem)

        except Exception as e:
            print(f"\nError processing {uri}: {e}")

    # 7. Generate Reports & Summary Dashboard
    output_buffer = []
    output_buffer.append(f"\n{'='*20} {args.set.upper()} SPLIT DETAILED REPORTS {'='*20}\n")
    
    for name, metric in metrics.items():
        output_buffer.append(f"## {name} Report")
        output_buffer.append(metric.report().to_string())
        output_buffer.append("\n" + "="*50 + "\n")

    # Build the final concise dashboard panel
    output_buffer.append(f"## Summary Dashboard (Averages Across {args.set.upper()} Dataset)")
    for name, metric in metrics.items():
        output_buffer.append(f"Overall {name}: {abs(metric) * 100:.2f}%")

    # Combine string buffer
    final_output = "\n".join(output_buffer)
    
    for name, metric in metrics.items():
        print(f"Overall {name}: {abs(metric) * 100:.2f}%")

    # Save cleanly to the summary log file
    with open(output_dir / f"{args.exp_label}_summary.txt", "a") as summary:
        summary.write(f"\nSet: {args.set}\n" + final_output + "\n")

if __name__ == "__main__":
    main()