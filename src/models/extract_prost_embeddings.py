"""
Extract ProstT5 embeddings for all protein sequences

Uses Rostlab/ProstT5 for structural information
Embeddings shape: (seq_length, 1024)
"""
# IMPORTANT: Set env vars BEFORE any HuggingFace imports
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import pandas as pd
import torch
from pathlib import Path
from transformers import T5Tokenizer, T5EncoderModel
from transformers.utils import logging as hf_logging
from tqdm import tqdm
import logging
import argparse
import json
import re
import time
import sys
import threading

# Enable HuggingFace progress bars
hf_logging.set_verbosity_info()

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Force flush stdout for real-time logging
sys.stdout.reconfigure(line_buffering=True)


class LoadingMonitor:
    """Background thread to monitor loading progress"""

    def __init__(self, interval=5):
        self.interval = interval
        self.running = False
        self.thread = None
        self.start_time = None
        self.stage = "unknown"

    def _monitor_loop(self):
        while self.running:
            elapsed = time.time() - self.start_time

            # Check memory usage
            ram_info = ""
            try:
                import psutil
                process = psutil.Process()
                ram_gb = process.memory_info().rss / 1024**3
                ram_info = f", RAM: {ram_gb:.2f}GB"
            except ImportError:
                pass

            gpu_info = ""
            if torch.cuda.is_available():
                try:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    gpu_info = f", GPU: {allocated:.2f}GB"
                except:
                    pass

            logger.info(f"  ... still {self.stage} ({elapsed:.0f}s elapsed{ram_info}{gpu_info})")
            time.sleep(self.interval)

    def start(self, stage="loading"):
        self.stage = stage
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)


def log_gpu_memory():
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
    else:
        logger.info("CUDA not available")


class ProstT5EmbeddingExtractor:
    """
    Extract ProstT5 embeddings for protein sequences
    """

    def __init__(self, model_name="Rostlab/ProstT5", device="cpu"):
        """
        Initialize ProstT5 model

        Args:
            model_name: HuggingFace model name
            device: 'cpu' or 'cuda'
        """
        logger.info(f"="*50)
        logger.info(f"Starting model initialization")
        logger.info(f"Model: {model_name}")
        logger.info(f"Target device: {device}")
        logger.info(f"="*50)

        self.device = device

        # Check CUDA availability
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            log_gpu_memory()

        # Start loading monitor
        monitor = LoadingMonitor(interval=10)

        # Load tokenizer
        logger.info("[1/4] Loading tokenizer...")
        start_time = time.time()
        monitor.start("loading tokenizer")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        monitor.stop()
        logger.info(f"[1/4] Tokenizer loaded in {time.time() - start_time:.2f}s")

        # Load model
        logger.info("[2/4] Loading model weights...")
        logger.info("      ProstT5 is ~6GB - this will take 1-3 minutes from cache")
        logger.info("      (If downloading for first time: ~10-15 minutes)")
        start_time = time.time()

        monitor.start("loading model weights from disk")

        if device == "cuda":
            logger.info("      Using half precision (FP16) for GPU")
            self.model = T5EncoderModel.from_pretrained(
                model_name,
                dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        else:
            logger.info("      Using full precision (FP32) for CPU")
            self.model = T5EncoderModel.from_pretrained(
                model_name,
                low_cpu_mem_usage=True
            )

        monitor.stop()
        logger.info(f"[2/4] Model weights loaded in {time.time() - start_time:.2f}s")

        # Move to device
        logger.info(f"[3/4] Moving model to {device}...")
        start_time = time.time()
        monitor.start(f"moving model to {device}")
        self.model.to(device)
        monitor.stop()
        logger.info(f"[3/4] Model moved to {device} in {time.time() - start_time:.2f}s")

        if torch.cuda.is_available():
            log_gpu_memory()

        # Set eval mode
        logger.info("[4/4] Setting model to eval mode...")
        self.model.eval()

        logger.info(f"="*50)
        logger.info(f"Model initialization complete!")
        logger.info(f"Embedding dimension: {self.model.config.d_model}")
        logger.info(f"="*50)

    def preprocess_sequence(self, sequence):
        """
        Preprocess protein sequence for ProstT5

        Args:
            sequence: Protein sequence (string)

        Returns:
            Preprocessed sequence with spaces between residues
        """
        # Add spaces between amino acids for T5 tokenizer
        sequence = " ".join(list(sequence))

        # Replace rare/ambiguous amino acids
        sequence = re.sub(r"[UZOB]", "X", sequence)

        return sequence

    def extract_embeddings(self, sequence, sequence_id="unknown"):
        """
        Extract per-residue embeddings for a sequence

        Args:
            sequence: Protein sequence (string)
            sequence_id: ID for logging

        Returns:
            torch.Tensor of shape (seq_length, 1024)
        """
        seq_len = len(sequence)
        logger.info(f"  Processing {sequence_id}: {seq_len} residues")

        # Preprocess
        logger.info(f"    [1/5] Preprocessing sequence...")
        sequence_processed = self.preprocess_sequence(sequence)

        # Tokenize
        logger.info(f"    [2/5] Tokenizing...")
        inputs = self.tokenizer(
            sequence_processed,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=True
        )
        logger.info(f"    [3/5] Moving to {self.device}...")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Log memory before inference
        if torch.cuda.is_available() and self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"    GPU memory before inference: {allocated:.2f}GB")

        # Extract embeddings
        logger.info(f"    [4/5] Running model inference (long sequences may take minutes)...")
        start_time = time.time()

        # Start monitor for long sequences
        monitor = None
        if seq_len > 1000:
            monitor = LoadingMonitor(interval=15)
            monitor.start(f"running inference on {seq_len} residues")

        with torch.no_grad():
            outputs = self.model(**inputs)

        if monitor:
            monitor.stop()

        inference_time = time.time() - start_time
        logger.info(f"    [4/5] Inference done in {inference_time:.2f}s")

        # Get last hidden state
        logger.info(f"    [5/5] Extracting embeddings...")
        embeddings = outputs.last_hidden_state.squeeze(0)

        # Remove special tokens (start, end)
        embeddings = embeddings[:-1]  # Remove EOS token

        # Clear GPU cache for long sequences
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"    Done! Shape: {embeddings.shape}")
        return embeddings.cpu().float()

    def extract_and_save(self, sequence_id, sequence, output_dir):
        """
        Extract embeddings and save to file

        Args:
            sequence_id: Unique sequence identifier
            sequence: Protein sequence
            output_dir: Directory to save embeddings
        """
        embeddings = self.extract_embeddings(sequence, sequence_id)

        output_file = output_dir / f"{sequence_id}.pt"
        logger.info(f"    Saving to {output_file}...")
        torch.save(embeddings, output_file)

        return embeddings.shape


def load_checkpoint(checkpoint_file):
    """
    Load checkpoint to resume processing

    Args:
        checkpoint_file: Path to checkpoint JSON

    Returns:
        Set of processed sequence IDs
    """
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        logger.info(f"Loaded checkpoint: {len(checkpoint['processed'])} sequences already processed")
        return set(checkpoint['processed'])
    return set()


def save_checkpoint(checkpoint_file, processed_ids):
    """
    Save checkpoint

    Args:
        checkpoint_file: Path to checkpoint JSON
        processed_ids: Set of processed sequence IDs
    """
    with open(checkpoint_file, 'w') as f:
        json.dump({'processed': list(processed_ids)}, f)


def main():
    """
    Main pipeline for extracting ProstT5 embeddings
    """
    parser = argparse.ArgumentParser(description='Extract ProstT5 embeddings')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file with sequences (absolute or relative path)')
    parser.add_argument('--output_dir', type=str, default='data/embeddings/prostt5',
                        help='Output directory for embeddings')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--checkpoint_interval', type=int, default=500,
                        help='Save checkpoint every N sequences')
    args = parser.parse_args()

    logger.info("Starting ProstT5 embedding extraction")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Device: {args.device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint file
    checkpoint_file = output_dir / "checkpoint.json"

    # Load checkpoint
    processed_ids = load_checkpoint(checkpoint_file)

    # Load dataset
    logger.info(f"Loading dataset from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Total sequences: {len(df)}")

    # Filter already processed
    if processed_ids:
        df_remaining = df[~df['sequence_id'].isin(processed_ids)]
        logger.info(f"Remaining sequences: {len(df_remaining)}")
    else:
        df_remaining = df

    if len(df_remaining) == 0:
        logger.info("All sequences already processed")
        return

    # Sort by sequence length (process shorter ones first)
    df_remaining = df_remaining.copy()
    df_remaining['seq_len'] = df_remaining['sequence'].str.len()
    df_remaining = df_remaining.sort_values('seq_len')
    logger.info(f"Sequences sorted by length: {df_remaining['seq_len'].min()} to {df_remaining['seq_len'].max()} residues")

    # Show what we're about to process
    logger.info("="*50)
    logger.info("Sequences to process:")
    for _, row in df_remaining.iterrows():
        logger.info(f"  - {row['sequence_id']}: {len(row['sequence'])} residues")
    logger.info("="*50)

    # Initialize extractor
    extractor = ProstT5EmbeddingExtractor(device=args.device)

    # Extract embeddings
    logger.info("="*50)
    logger.info("Starting embedding extraction...")
    logger.info("="*50)

    total = len(df_remaining)
    for i, (idx, row) in enumerate(df_remaining.iterrows()):
        sequence_id = row['sequence_id']
        sequence = row['sequence']

        logger.info(f"\n[{i+1}/{total}] Processing sequence {sequence_id}")
        logger.info(f"    Length: {len(sequence)} residues")

        try:
            start_time = time.time()

            # Extract and save
            shape = extractor.extract_and_save(sequence_id, sequence, output_dir)

            elapsed = time.time() - start_time
            logger.info(f"[{i+1}/{total}] SUCCESS: {sequence_id} processed in {elapsed:.2f}s")

            # Add to processed
            processed_ids.add(sequence_id)

            # Checkpoint
            if len(processed_ids) % args.checkpoint_interval == 0:
                save_checkpoint(checkpoint_file, processed_ids)
                logger.info(f"Checkpoint saved: {len(processed_ids)} sequences processed")

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"[{i+1}/{total}] GPU OUT OF MEMORY for {sequence_id} ({len(sequence)} residues)")
            logger.error(f"    Try running with --device cpu for long sequences")
            torch.cuda.empty_cache()
            continue

        except Exception as e:
            logger.error(f"[{i+1}/{total}] ERROR processing {sequence_id}: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Final checkpoint
    save_checkpoint(checkpoint_file, processed_ids)

    logger.info(f"Extraction complete: {len(processed_ids)} embeddings saved")
    logger.info(f"Output directory: {output_dir}")

    # Statistics
    logger.info("\n=== Statistics ===")
    logger.info(f"Total sequences in dataset: {len(df)}")
    logger.info(f"Successfully processed: {len(processed_ids)}")

    if len(processed_ids) < len(df):
        missing = set(df['sequence_id']) - processed_ids
        logger.warning(f"Missing embeddings: {len(missing)}")
        logger.warning(f"Missing IDs: {list(missing)[:10]}...")


if __name__ == "__main__":
    main()
