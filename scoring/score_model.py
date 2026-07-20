import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch

from pyannote.database import registry,FileFinder
from pyannote.audio import Pipeline, Model
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment, Timeline
from pyannote.metrics.diarization import (
    DiarizationErrorRate, 
    JaccardErrorRate,
    DiarizationPurity,
    DiarizationCoverage
)

def get_annotated(file):
    extent = file["annotation"].get_timeline().extent()
    return Timeline([Segment(0, extent.end)])

def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned pyannote model using saved thresholds.")
    parser.add_argument("--config", type=str, required=True, help="Path to database.yml")
    parser.add_argument("--wav-dir", type=str, required=False, help="Path to audio folder")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API Token")
    parser.add_argument("--model-path", type=str, required=True, help="Path to finetuned model (.ckpt)")
    parser.add_argument("--exp-label", type=str, required=True, help="Experiment label for output naming")
    parser.add_argument("--set", type=str, choices=["train", "development", "test"], default="test", help="Split to evaluate")
    parser.add_argument("--protocol-name", type=str, default="up_child.SpeakerDiarization.CustomProtocol", help="Database protocol name")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--output-dir", type=str, default="output", help="Root directory for saving evaluation results")
    parser.add_argument("--collar",type=float, default=0.0,help="Collar (seconds) applied around reference boundaries to forgive near-miss detections (default: 0.0)")
    parser.add_argument("--min_duration",type=float, default=0.0,help="Min duration off parameter to merge segments")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    # 1. Output Setup
    base_output_path = Path(args.output_dir)
    output_dir = base_output_path / args.exp_label
    split_dir = output_dir / "finetuned" / args.set
    split_dir.mkdir(parents=True, exist_ok=True)

    seg_threshold = 0.0
    min_duration_off = args.min_duration

    print(f"Loaded parameters -> threshold: {seg_threshold:.4f}, min_duration_off: {min_duration_off:.4f}")

    # 3. Setup Pyannote Database
    if not os.path.exists(args.config):
        print(f"Error: {args.config} not found.")
        return

    registry.load_database(args.config)
    try:
        dataset = registry.get_protocol(
            args.protocol_name, 
            preprocessors={
                "audio": FileFinder(), 
                "annotated": get_annotated
            }
        )
    except ValueError as e:
        print(f"Error loading protocol: {e}")
        return

    # 4. Load Base Pipeline Structure and Swap the Segmentation Model
    print("Initializing architecture using pyannote/speaker-diarization-3.1...")
    try:
        finetuned_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=args.token)
    except Exception as e:
        print(f"HF Error: {e}")
        return

    print(f"Injecting local finetuned model weights: {args.model_path}")
    finetuned_model = Model.from_pretrained(args.model_path)
    finetuned_model.eval()
    
    # Re-build pipeline targeting the local finetuned model architecture
    finetuned_pipeline = SpeakerDiarization(
        segmentation=finetuned_model,
        embedding=finetuned_pipeline.embedding,
        embedding_exclude_overlap=finetuned_pipeline.embedding_exclude_overlap,
        clustering=finetuned_pipeline.klustering,
    )
    
    # Apply parameters directly
    finetuned_pipeline.segmentation.threshold = seg_threshold
    finetuned_pipeline.segmentation.min_duration_off = min_duration_off
    
    # Initialize default clustering parameters
    finetuned_pipeline.instantiate({
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 20,
            "threshold": 0.8,
        },
    })
    finetuned_pipeline.to(device)

    # 5. Extract target dataset split items
    split_generators = {"train": dataset.train, "development": dataset.development, "test": dataset.test}
    dataset_split = list(split_generators[args.set]())

    if not dataset_split:
        print(f"No files found in '{args.set}' split.")
        return

    # 6. Initialize Multi-Metric Trackers

    print(f"Collar applied: {args.collar}")
    metrics = {
        "Diarization Error Rate (DER)": DiarizationErrorRate(collar=args.collar, skip_overlap=False),
        "Jaccard Error Rate (JER)": JaccardErrorRate(collar=args.collar),
        "Diarization Purity": DiarizationPurity(collar=args.collar),
        "Diarization Coverage": DiarizationCoverage(collar=args.collar)
    }

    # 7. Process & Evaluate
    print(f"\nEvaluating finetuned model on {args.set.upper()} split ({len(dataset_split)} files)...")
    for file in tqdm(dataset_split, desc=f"Scoring {args.set}"):
        uri = file["uri"]
        try:
            output = finetuned_pipeline(file)
            hypothesis = getattr(output, "speaker_diarization", output)

            # Save generated RTTM prediction files to disk
            with open(split_dir / f"{uri}.rttm", "w") as rttm_file:
                hypothesis.write_rttm(rttm_file)

            # Accumulate evaluation statistics across tracking classes
            reference = file["annotation"]
            uem = file.get("annotated")
            for metric in metrics.values():
                metric(reference, hypothesis, uem=uem)

        except Exception as e:
            print(f"\nError processing {uri}: {e}")

    # 8. Compile Reports and Summary Metrics
    output_buffer = []
    output_buffer.append(f"\n{'='*20} FINETUNED {args.set.upper()} PERFORMANCE REPORTS {'='*20}\n")
    
    for name, metric in metrics.items():
        output_buffer.append(f"## {name} Report")
        output_buffer.append(metric.report().to_string())
        output_buffer.append("\n" + "="*50 + "\n")

    output_buffer.append(f"## Summary Dashboard (Averages Across Finetuned {args.set.upper()} Dataset)")
    for name, metric in metrics.items():
        output_buffer.append(f"Overall {name}: {abs(metric) * 100:.2f}%")

    final_output = "\n".join(output_buffer)
    #print(final_output)
    for name, metric in metrics.items():
        print(f"Overall {name}: {abs(metric) * 100:.2f}%")

    # Append complete experiment data logs to the label run summary text file
    with open(output_dir / f"{args.exp_label}_summary.txt", "a") as summary:
        summary.write(f"\nModel: {Path(args.model_path).name} | Set: {args.set}\n" + final_output + "\n")

if __name__ == "__main__":
    main()