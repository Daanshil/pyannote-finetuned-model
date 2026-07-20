import os
import argparse
from pathlib import Path
import torch
import types

from pyannote.database import FileFinder, registry
from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.core import Segment, Timeline

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch import seed_everything
from torch.optim import Adam

def get_annotated(file):
    # 1. Get the 'extent' (the start and end of all annotations)
    extent = file["annotation"].get_timeline().extent()
    
    # 2. Wrap it in a Timeline object
    # This is what the pyannote training loop is specifically looking for
    return Timeline([Segment(0, extent.end)])

def main():
    parser = argparse.ArgumentParser(description="Finetune Pyannote Segmentation Model.")
    
    # Core configuration arguments
    parser.add_argument("--config", type=str, required=True, help="Path to database.yml")
    parser.add_argument("--wav-dir", type=str, required=False, help="Path to audio folder")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API Token")
    parser.add_argument("--exp-label", type=str, required=True, help="Experiment label for logs/checkpoints")
    parser.add_argument("--protocol-name", type=str, default="up_child.SpeakerDiarization.CustomProtocol", help="Name of the pyannote protocol to use")
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

    # 1. Pyannote Database Setup
    if not os.path.exists(args.config):
        print(f"Error: {args.config} not found.")
        return

    registry.load_database(args.config)
    
    # Re-apply the protocol with the corrected preprocessors using lambda mapping
    try:
        dataset = registry.get_protocol(
            args.protocol_name, 
            preprocessors={
                "audio": FileFinder(),
                "annotated": get_annotated
            }
        )
    except ValueError as e:
        print(f"Error loading protocol '{args.protocol_name}': {e}")
        return

    # 2. Sanity Check Training Set
    try:
        sample = next(dataset.train())
        print(f"Protocol loaded. Sample annotated type: {type(sample['annotated'])}")
    except Exception as e:
        print(f"Error accessing training data: {e}")
        return

    # 3. Load Pretrained Segmentation Model
    print("Loading pyannote/segmentation-3.0...")
    try:
        model = Model.from_pretrained("pyannote/segmentation-3.0", token=args.token)
        print(f"Base model loaded ({type(model).__name__})")
    except Exception as e:
        print(f"HF Error loading model: {e}")
        return

    # 4. Define Speaker Diarization Task
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

    # 5. Lightning Callbacks & Loggers
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

    # 6. Initialize Trainer & Fit Model
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=0.5,
    )
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=args.lr)

    print(f"\nStarting training for experiment: {args.exp_label}...")
    opt = model.configure_optimizers()
    print(opt)
    optimizer = opt[0][0] if isinstance(opt, tuple) else opt
    print(type(optimizer).__name__)
    print(optimizer.defaults)  # lr, betas, eps, weight_decay, etc.

    
    
    trainer.fit(model)

    print(f"\nTraining Complete!")
    print(f"Best model checkpoint saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()