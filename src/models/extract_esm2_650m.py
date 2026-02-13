"""
Extract ESM-2 650M embeddings for all protein sequences

Uses facebook/esm2_t33_650M_UR50D (650M parameters)
Embeddings shape: (seq_length, 1280)

This is a significant upgrade from the 35M model (480-D)
"""
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import pandas as pd
import torch
import esm
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
import json
import time
import gc
import zlib
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def stable_shard(sequence_id: str, num_shards: int) -> int:
    """Deterministic shard assignment based on sequence_id."""
    return zlib.crc32(sequence_id.encode("utf-8")) % num_shards


class ESM2Extractor:
    """Extract embeddings using ESM-2 650M model"""

    def __init__(self, device="cuda:0", use_half=True):
        """
        Args:
            device: 'cuda:0' / 'cuda:1' / 'cpu'
            use_half: Use FP16 for GPU (saves memory)
        """
        self.device = device
        self.use_half = use_half and str(device).startswith("cuda")

        logger.info("Loading ESM-2 650M model...")
        logger.info("  Model: esm2_t33_650M_UR50D")
        logger.info("  Parameters: 650M")
        logger.info("  Embedding dim: 1280")

        start_time = time.time()

        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()

        self.model = self.model.to(device)
        if self.use_half:
            self.model = self.model.half()
            logger.info("  Using FP16 precision")

        self.model.eval()

        load_time = time.time() - start_time
        logger.info(f"  Model loaded in {load_time:.1f}s")

        if str(device).startswith("cuda"):
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"  GPU memory: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")

    def extract_embeddings(self, sequence: str, sequence_id: str):
        """
        Extract per-residue embeddings for a sequence.

        Returns:
            torch.Tensor of shape (seq_length, 1280) on CPU (float32)
        """
        seq_len = len(sequence)
        data = [(sequence_id, sequence)]
        _, _, batch_tokens = self.batch_converter(data)

        # tokens must stay as long
        batch_tokens = batch_tokens.to(self.device, non_blocking=True)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)

        embeddings = results["representations"][33]  # (1, T, 1280) with BOS/EOS
        embeddings = embeddings[0, 1:seq_len + 1, :]  # remove special tokens
        embeddings = embeddings.float().cpu()  # store float32 on CPU

        # cleanup
        del results, batch_tokens
        return embeddings


def load_checkpoint_ids(checkpoint_file: Path) -> set:
    if not checkpoint_file.exists():
        return set()
    try:
        with open(checkpoint_file, "r") as f:
            ckpt = json.load(f)
        return set(map(str, ckpt.get("processed_ids", [])))
    except Exception as e:
        logger.warning(f"Could not read checkpoint {checkpoint_file}: {e}")
        return set()


def save_checkpoint(checkpoint_file: Path, processed_ids: set, meta: Optional[Dict[str, Any]] = None):
    payload = {"processed_ids": sorted(processed_ids)}
    if meta:
        payload.update(meta)
    tmp = checkpoint_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(checkpoint_file)


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-2 650M embeddings")
    parser.add_argument("--input", type=str, default="data/processed/dataset_with_folds.csv",
                        help="Input CSV file")
    parser.add_argument("--output_dir", type=str, default="data/embeddings/esm2_650m",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device base (cuda or cpu)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="CUDA device index to use (only for device selection)")
    parser.add_argument("--use_half", action="store_true", default=True,
                        help="Use FP16 on GPU (default: enabled)")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                        help="Save checkpoint every N sequences")
    parser.add_argument("--max_length", type=int, default=0,
                        help="If >0, skip sequences longer than this length (0 = disabled)")
    parser.add_argument("--cpu_fallback", action="store_true",
                        help="If OOM on GPU, retry extraction on CPU")
    parser.add_argument("--cpu_max_len", type=int, default=0,
                        help="If >0 and seq_len > cpu_max_len, run directly on CPU (avoids OOM)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Number of shards for multi-process extraction (default: 1)")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard index in [0, num_shards-1] (default: 0)")
    args = parser.parse_args()

    # Resolve device string
    if args.device == "cuda":
        torch.cuda.set_device(args.gpu_id)
        device = f"cuda:{args.gpu_id}"
        logger.info(f"Using CUDA device: {device}")
    else:
        device = "cpu"
        logger.info("Using CPU")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset from {args.input}")
    df = pd.read_csv(args.input)
    if "sequence_id" not in df.columns or "sequence" not in df.columns:
        raise ValueError("CSV must contain columns: sequence_id, sequence")

    df["sequence_id"] = df["sequence_id"].astype(str)
    df["sequence"] = df["sequence"].astype(str)
    df["seq_len"] = df["sequence"].str.len()

    logger.info(f"Total sequences: {len(df)}")

    # Sharding (deterministic on sequence_id)
    if args.num_shards > 1:
        if not (0 <= args.shard_id < args.num_shards):
            raise ValueError("--shard_id must be in [0, num_shards-1]")
        df = df[df["sequence_id"].apply(lambda s: stable_shard(s, args.num_shards) == args.shard_id)].copy()
        logger.info(f"Shard {args.shard_id}/{args.num_shards}: {len(df)} sequences")

    # Existing .pt files are the source of truth
    existing_files = set(p.stem for p in output_dir.glob("*.pt"))

    checkpoint_file = output_dir / "checkpoint.json"
    checkpoint_ids = load_checkpoint_ids(checkpoint_file)

    # Self-heal: keep only checkpoint ids that actually have a .pt
    checkpoint_real = {sid for sid in checkpoint_ids if sid in existing_files}
    phantom = checkpoint_ids - checkpoint_real
    if phantom:
        logger.warning(f"Checkpoint contains {len(phantom)} IDs without .pt files (will be re-processed).")

    processed_ids = set(existing_files)  # definitive
    logger.info(f"Already have .pt files: {len(processed_ids)}")

    # Filter sequences to process = those missing on disk
    to_process = df[~df["sequence_id"].isin(processed_ids)].copy()

    # Optional skip by length
    if args.max_length and args.max_length > 0:
        before = len(to_process)
        to_process = to_process[to_process["seq_len"] <= args.max_length].copy()
        skipped = before - len(to_process)
        if skipped > 0:
            logger.warning(f"Skipped {skipped} sequences with length > {args.max_length}")

    logger.info(f"Sequences to process: {len(to_process)}")
    if len(to_process) == 0:
        logger.info("All sequences already processed!")
        # still rewrite a clean checkpoint if needed
        save_checkpoint(checkpoint_file, processed_ids, meta={"updated_at": time.time()})
        return

    # Process shorter first
    to_process = to_process.sort_values("seq_len", ascending=True)

    # Initialize GPU extractor (if requested)
    extractor_gpu = None
    extractor_cpu = None

    if device != "cpu":
        extractor_gpu = ESM2Extractor(device=device, use_half=bool(args.use_half))
    else:
        extractor_cpu = ESM2Extractor(device="cpu", use_half=False)

    failed = []
    newly_processed = 0

    logger.info("\nExtracting embeddings...")
    pbar = tqdm(to_process.itertuples(index=False), total=len(to_process), desc="Extracting")

    for n, row in enumerate(pbar, 1):
        sequence_id = str(getattr(row, "sequence_id"))
        sequence = str(getattr(row, "sequence"))
        seq_len = int(getattr(row, "seq_len"))

        output_file = output_dir / f"{sequence_id}.pt"

        # safety: if file appeared meanwhile, skip
        if output_file.exists():
            continue

        try:
            # Decide CPU vs GPU
            run_on_cpu = (device == "cpu") or (args.cpu_max_len > 0 and seq_len > args.cpu_max_len)

            if run_on_cpu:
                if extractor_cpu is None:
                    logger.warning("Initializing CPU extractor (first CPU job). This is slower but OOM-safe.")
                    extractor_cpu = ESM2Extractor(device="cpu", use_half=False)
                embeddings = extractor_cpu.extract_embeddings(sequence, sequence_id)
            else:
                embeddings = extractor_gpu.extract_embeddings(sequence, sequence_id)

            # Basic shape sanity
            if embeddings.ndim != 2 or embeddings.shape[0] != seq_len or embeddings.shape[1] != 1280:
                logger.warning(
                    f"{sequence_id}: unexpected embedding shape {tuple(embeddings.shape)} "
                    f"(expected ({seq_len}, 1280))"
                )

            torch.save(embeddings, output_file)
            processed_ids.add(sequence_id)
            newly_processed += 1

        except torch.OutOfMemoryError:
            # GPU OOM handling
            if str(device).startswith("cuda"):
                logger.error(f"OOM on GPU for {sequence_id} (len={seq_len}).")
                torch.cuda.empty_cache()
                gc.collect()

            if args.cpu_fallback:
                try:
                    if extractor_cpu is None:
                        logger.warning("Initializing CPU extractor for OOM fallback. This is slower but stable.")
                        extractor_cpu = ESM2Extractor(device="cpu", use_half=False)
                    embeddings = extractor_cpu.extract_embeddings(sequence, sequence_id)
                    torch.save(embeddings, output_file)
                    processed_ids.add(sequence_id)
                    newly_processed += 1
                    logger.warning(f"Recovered via CPU fallback: {sequence_id} (len={seq_len})")
                except Exception as e2:
                    failed.append((sequence_id, seq_len, f"CPU fallback failed: {e2}"))
                    logger.error(f"CPU fallback failed for {sequence_id}: {e2}")
            else:
                failed.append((sequence_id, seq_len, "GPU OOM (cpu_fallback disabled)"))

        except Exception as e:
            failed.append((sequence_id, seq_len, str(e)))
            logger.error(f"Error processing {sequence_id}: {e}")

        # Periodic checkpoint (cheap and safe)
        if newly_processed > 0 and newly_processed % args.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_file,
                processed_ids,
                meta={"updated_at": time.time(), "processed_count": len(processed_ids)}
            )
            logger.info(f"Checkpoint saved: {len(processed_ids)} total .pt files")

            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

    # Final checkpoint
    save_checkpoint(
        checkpoint_file,
        processed_ids,
        meta={"updated_at": time.time(), "processed_count": len(processed_ids)}
    )

    if failed:
        fail_path = output_dir / "failed_ids.txt"
        fail_path.write_text("\n".join([f"{sid}\t{L}\t{msg}" for sid, L, msg in failed]) + "\n")
        logger.warning(f"Some sequences failed ({len(failed)}). See: {fail_path}")

    logger.info(f"\nDone! Processed {newly_processed} new sequences")
    logger.info(f"Total .pt files now: {len(processed_ids)}")
    logger.info(f"Embeddings saved to: {output_dir}")


if __name__ == "__main__":
    main()

# """
# Extract ESM-2 650M embeddings for all protein sequences
# 
# Uses facebook/esm2_t33_650M_UR50D (650M parameters)
# Embeddings shape: (seq_length, 1280)
# 
# This is a significant upgrade from the 35M model (480-D)
# """
# import os
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# 
# import pandas as pd
# import torch
# import esm
# from pathlib import Path
# from tqdm import tqdm
# import logging
# import argparse
# import json
# import time
# import gc
# 
# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(levelname)s] %(asctime)s - %(message)s',
#     datefmt='%H:%M:%S'
# )
# logger = logging.getLogger(__name__)
# 
# 
# class ESM2Extractor:
#     """Extract embeddings using ESM-2 650M model"""
# 
#     def __init__(self, device="cuda", use_half=True):
#         """
#         Args:
#             device: 'cuda' or 'cpu'
#             use_half: Use FP16 for GPU (saves memory)
#         """
#         self.device = device
#         self.use_half = use_half and device == "cuda"
# 
#         logger.info("Loading ESM-2 650M model...")
#         logger.info("  Model: esm2_t33_650M_UR50D")
#         logger.info("  Parameters: 650M")
#         logger.info("  Embedding dim: 1280")
# 
#         start_time = time.time()
# 
#         # Load model and alphabet
#         self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#         self.batch_converter = self.alphabet.get_batch_converter()
# 
#         # Move to device
#         self.model = self.model.to(device)
#         if self.use_half:
#             self.model = self.model.half()
#             logger.info("  Using FP16 precision")
# 
#         self.model.eval()
# 
#         load_time = time.time() - start_time
#         logger.info(f"  Model loaded in {load_time:.1f}s")
# 
#         # Log GPU memory
#         if device == "cuda":
#             allocated = torch.cuda.memory_allocated() / 1024**3
#             logger.info(f"  GPU memory: {allocated:.2f} GB")
# 
#     def extract_embeddings(self, sequence, sequence_id="unknown"):
#         """
#         Extract per-residue embeddings for a sequence
# 
#         Args:
#             sequence: Protein sequence (string)
#             sequence_id: ID for logging
# 
#         Returns:
#             torch.Tensor of shape (seq_length, 1280)
#         """
#         seq_len = len(sequence)
# 
#         # Prepare data
#         data = [(sequence_id, sequence)]
#         batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
#         batch_tokens = batch_tokens.to(self.device)
# 
#         if self.use_half:
#             batch_tokens = batch_tokens.long()  # Tokens must stay as long
# 
#         # Extract embeddings
#         with torch.no_grad():
#             results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
# 
#         # Get embeddings from layer 33 (last layer)
#         embeddings = results["representations"][33]
# 
#         # Remove batch dimension and special tokens (CLS at start, EOS at end)
#         embeddings = embeddings[0, 1:seq_len+1, :]
# 
#         # Convert to float32 for storage
#         embeddings = embeddings.float().cpu()
# 
#         return embeddings
# 
# 
# def main():
#     parser = argparse.ArgumentParser(description='Extract ESM-2 650M embeddings')
#     parser.add_argument('--input', type=str, default='data/processed/dataset_with_folds.csv',
#                         help='Input CSV file')
#     parser.add_argument('--output_dir', type=str, default='data/embeddings/esm2_650m',
#                         help='Output directory')
#     parser.add_argument('--device', type=str, default='cuda',
#                         help='Device (cuda or cpu)')
#     parser.add_argument('--batch_size', type=int, default=1,
#                         help='Batch size (1 recommended for long sequences)')
#     parser.add_argument('--checkpoint_every', type=int, default=100,
#                         help='Save checkpoint every N sequences')
#     parser.add_argument('--max_length', type=int, default=2000,
#                         help='Max sequence length (skip longer)')
#     parser.add_argument('--gpu_id', type=int, default=0,
#                         help='GPU ID to use')
#     args = parser.parse_args()
# 
#     # Set GPU
#     if args.device == "cuda":
#         torch.cuda.set_device(args.gpu_id)
#         logger.info(f"Using GPU {args.gpu_id}")
# 
#     # Create output directory
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
# 
#     # Load dataset
#     logger.info(f"Loading dataset from {args.input}")
#     df = pd.read_csv(args.input)
#     logger.info(f"Total sequences: {len(df)}")
# 
#     # Check for already processed sequences
#     checkpoint_file = output_dir / "checkpoint.json"
#     processed_ids = set()
# 
#     if checkpoint_file.exists():
#         with open(checkpoint_file, 'r') as f:
#             checkpoint = json.load(f)
#             processed_ids = set(checkpoint.get('processed_ids', []))
#         logger.info(f"Resuming from checkpoint: {len(processed_ids)} already processed")
# 
#     # Also check existing .pt files
#     existing_files = set(p.stem for p in output_dir.glob("*.pt"))
#     processed_ids = processed_ids.union(existing_files)
#     logger.info(f"Total already processed: {len(processed_ids)}")
# 
#     # Filter sequences to process
#     to_process = df[~df['sequence_id'].isin(processed_ids)].copy()
# 
#     # Filter by length
#     to_process = to_process[to_process['sequence'].str.len() <= args.max_length]
#     logger.info(f"Sequences to process: {len(to_process)}")
# 
#     if len(to_process) == 0:
#         logger.info("All sequences already processed!")
#         return
# 
#     # Sort by length (process shorter first for faster progress)
#     to_process = to_process.sort_values('seq_length')
# 
#     # Initialize extractor
#     extractor = ESM2Extractor(device=args.device, use_half=True)
# 
#     # Process sequences
#     logger.info("\nExtracting embeddings...")
#     newly_processed = []
# 
#     for idx, row in tqdm(to_process.iterrows(), total=len(to_process), desc="Extracting"):
#         sequence_id = row['sequence_id']
#         sequence = row['sequence']
# 
#         try:
#             # Extract embeddings
#             embeddings = extractor.extract_embeddings(sequence, sequence_id)
# 
#             # Save
#             output_file = output_dir / f"{sequence_id}.pt"
#             torch.save(embeddings, output_file)
# 
#             newly_processed.append(sequence_id)
# 
#             # Checkpoint
#             if len(newly_processed) % args.checkpoint_every == 0:
#                 all_processed = list(processed_ids.union(set(newly_processed)))
#                 with open(checkpoint_file, 'w') as f:
#                     json.dump({'processed_ids': all_processed}, f)
#                 logger.info(f"Checkpoint saved: {len(all_processed)} total")
# 
#                 # Clear GPU cache
#                 if args.device == "cuda":
#                     torch.cuda.empty_cache()
#                     gc.collect()
# 
#         except Exception as e:
#             logger.error(f"Error processing {sequence_id}: {str(e)}")
#             continue
# 
#     # Final checkpoint
#     all_processed = list(processed_ids.union(set(newly_processed)))
#     with open(checkpoint_file, 'w') as f:
#         json.dump({'processed_ids': all_processed}, f)
# 
#     logger.info(f"\nDone! Processed {len(newly_processed)} new sequences")
#     logger.info(f"Total processed: {len(all_processed)}")
#     logger.info(f"Embeddings saved to: {output_dir}")
# 
# 
# if __name__ == "__main__":
#     main()
