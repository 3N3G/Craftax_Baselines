"""
Compute normalization statistics for hidden states from training data
"""
import numpy as np
import glob
import argparse
from pathlib import Path

def compute_stats(data_dir, output_path, num_files=None):
    """
    Compute mean and std of hidden states from training data

    Args:
        data_dir: Directory containing trajectories_batch_*.npz files
        output_path: Where to save hidden_state_stats.npz
        num_files: Number of files to use (None = all files)
    """
    print(f"Searching for training data in: {data_dir}")

    search_pattern = str(Path(data_dir) / "trajectories_batch_*.npz")
    files = sorted(glob.glob(search_pattern))

    if not files:
        raise ValueError(f"No training files found matching: {search_pattern}")

    print(f"Found {len(files)} files")

    if num_files is not None:
        files = files[:num_files]
        print(f"Using first {num_files} files")

    # Collect hidden states
    all_hidden_states = []

    for i, filepath in enumerate(files):
        print(f"Loading {i+1}/{len(files)}: {Path(filepath).name}")
        try:
            with np.load(filepath) as data:
                if 'hidden_state' not in data:
                    print(f"  Warning: 'hidden_state' not found in {filepath}, skipping")
                    continue

                raw_hidden = data['hidden_state']  # Expected shape: (N, 80, 2560) or (N, 2560)

                # Mean pool if needed
                if raw_hidden.ndim == 3:
                    pooled_hidden = np.mean(raw_hidden, axis=1)  # (N, 2560)
                else:
                    pooled_hidden = raw_hidden

                all_hidden_states.append(pooled_hidden)
                print(f"  Loaded {pooled_hidden.shape[0]} samples")

        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
            continue

    if not all_hidden_states:
        raise ValueError("No hidden states successfully loaded!")

    # Concatenate and compute statistics
    print("\nComputing statistics...")
    all_hidden = np.concatenate(all_hidden_states, axis=0)
    print(f"Total samples: {all_hidden.shape[0]}")
    print(f"Hidden state dimension: {all_hidden.shape[1]}")

    hidden_mean = np.mean(all_hidden, axis=0)
    hidden_std = np.std(all_hidden, axis=0)

    # Prevent division by zero
    hidden_std = np.where(hidden_std < 1e-6, 1.0, hidden_std)

    print(f"\nStatistics:")
    print(f"  Mean - min: {hidden_mean.min():.6f}, max: {hidden_mean.max():.6f}, avg: {hidden_mean.mean():.6f}")
    print(f"  Std  - min: {hidden_std.min():.6f}, max: {hidden_std.max():.6f}, avg: {hidden_std.mean():.6f}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        mean=hidden_mean,
        std=hidden_std,
        num_samples=all_hidden.shape[0],
        num_files=len(files)
    )

    print(f"\n✓ Saved to: {output_path}")
    print(f"  Shape: mean={hidden_mean.shape}, std={hidden_std.shape}")

def main():
    parser = argparse.ArgumentParser(description="Compute hidden state normalization statistics")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/data/group_data/rl/geney/craftax_labelled_results_with_returns',
        help='Directory containing training data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/data/group_data/rl/geney/checkpoints/awr_augmented/hidden_state_stats.npz',
        help='Output path for stats file'
    )
    parser.add_argument(
        '--num_files',
        type=int,
        default=None,
        help='Number of files to use (default: all files)'
    )

    args = parser.parse_args()
    compute_stats(args.data_dir, args.output, args.num_files)

if __name__ == "__main__":
    main()
