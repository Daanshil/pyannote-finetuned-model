import os
import argparse
from pathlib import Path
import torch
from lightning.pytorch import seed_everything

from pyannote.database import registry, FileFinder
from pyannote.database.protocol import SpeakerDiarizationProtocol  # <-- Added the proper base class
from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.core import Segment, Timeline

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

# Fix: Inherit from SpeakerDiarizationProtocol instead of SpeakerDiarization task
class CombinedProtocol(SpeakerDiarizationProtocol):
    
    def __init__(self, protocols):
        super().__init__()
        self.protocols = protocols
        self.name = "Combined_Dataset"
        
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
    # Get the 'extent' (the start and end of all annotations)
    extent = file["annotation"].get_timeline().extent()
    # Wrap it in a Timeline object for the pyannote training loop
    return Timeline([Segment(0, extent.end)])

def main():
    parser = argparse.ArgumentParser(description="Finetune Pyannote Segmentation Model on Multiple Protocols.")
    
    # Core configuration arguments
    parser.add_argument("--config", type=str, required=True, help="Path to multi.yml configuration file")
    parser.add_argument(
        "--protocols", 
        type=str, 
        nargs="+", 
        required=True, 
        help="Space-separated list of pyannote protocol names to combine (e.g., Babaloon.SpeakerDiarization.CustomProtocol up_child.SpeakerDiarization.CustomProtocol)"
    )
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API Token")
    parser.add_argument("--exp-label", type=str, required=True, help="Experiment label for logs/checkpoints")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA if available")
    
    # Tuning Arguments
    parser.add_argument("--duration", type=float, default=15.0, help="Length (seconds) of training chunks")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of audio chunks per batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimization")
    parser.add_argument("--log-dir", type=str, default="logs_15s", help="Root directory for logs and checkpoints")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of workers for data loading")
    
    args = parser.parse_args()

    seed_everything(42, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    accelerator = "gpu" if device.type == "cuda" else "cpu"
    print(f"Using accelerator: {accelerator}")

    # 1. Pyannote Database Setup via multi.yml
    if not os.path.exists(args.config):
        print(f"Error: {args.config} not found.")
        return

    print(f"Loading configuration database: {args.config}")
    registry.load_database(args.config)
    
    # 2. Dynamic initialization loop using FileFinder to respect multi.yml paths
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

    # Fuse protocols inside the combined space
    print(f"Fusing {len(loaded_protocols)} protocols inside CombinedProtocol layer...")
    dataset = CombinedProtocol(loaded_protocols)

    # 3. Sanity Check Training Set
    try:
        sample = next(dataset.train())
        print(f"Combined database loaded. Sample annotated type: {type(sample['annotated'])}")
        print(f"Sample absolute audio path mapped: {sample['audio']}")
    except Exception as e:
        print(f"Error accessing training data generator layer: {e}")
        return

    # 4. Load Pretrained Segmentation Model
    print("Loading pyannote/segmentation-3.0...")
    try:
        model = Model.from_pretrained("pyannote/segmentation-3.0", token=args.token)
        print(f"Base model loaded ({type(model).__name__})")
    except Exception as e:
        print(f"HF Error loading model: {e}")
        return

    # 5. Define Speaker Diarization Task
    task = SpeakerDiarization(
        dataset,
        duration=args.duration,
        max_speakers_per_frame=2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model.task = task
    model.learning_rate = args.lr
    
    print(f"Task attached. Duration: {task.duration}s | Batch Size: {task.batch_size} | Learning Rate: {model.learning_rate}")

    # 6. Lightning Callbacks & Loggers
    base_log_path = Path(args.log_dir)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="loss/val",
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
        save_last=False,
        filename=f"{args.exp_label}-" + "{epoch}-{loss/val:.4f}",
    )
    
    early_stopping = EarlyStopping(
        monitor="loss/val",
        mode="min",
        patience=args.patience,
        strict=True,
    )

    logger = TensorBoardLogger(save_dir=str(base_log_path), name=args.exp_label)
    callbacks = [TQDMProgressBar(), checkpoint_callback, early_stopping, LearningRateMonitor(logging_interval='step')]

    # 7. Initialize Trainer & Fit Model
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=0.5,
    )

    print(f"\nStarting training for multi-dataset experiment: {args.exp_label}...")
    trainer.fit(model)

    print(f"\nTraining Complete!")
    print(f"Best model checkpoint saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()