#!/usr/bin/env python3
"""
Parallel tar extraction script
Extracts all tar files in a specified directory using multiple CPU cores
"""

import os
import tarfile
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def extract_tar(args):
    """
    Extract a single tar file to its parent directory

    Args:
        args: tuple of (tar_path, keep_tar)

    Returns:
        tuple: (success: bool, tar_path: str, message: str)
    """
    tar_path, keep_tar = args
    tar_name = os.path.basename(tar_path)
    extract_dir = os.path.dirname(tar_path)

    try:
        # Extract tar file to its parent directory
        with tarfile.open(tar_path, 'r:*') as tar:
            tar.extractall(path=extract_dir)

        # Optionally remove tar file after successful extraction
        if not keep_tar:
            os.remove(tar_path)
            return (True, tar_name, f"Extracted and removed")
        else:
            return (True, tar_name, f"Extracted successfully")

    except Exception as e:
        return (False, tar_name, f"Error: {str(e)}")


def find_tar_files(directory):
    """
    Find all tar files in the specified directory

    Args:
        directory: Path to search for tar files

    Returns:
        list: List of tar file paths
    """
    tar_extensions = ['.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tar.xz']
    tar_files = []

    directory = Path(directory)
    for ext in tar_extensions:
        tar_files.extend(directory.glob(f'*{ext}'))

    return [str(f) for f in sorted(tar_files)]


def main():
    parser = argparse.ArgumentParser(
        description='Extract tar files in parallel using multiple CPU cores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all tars in data/waymo/tar using all available CPUs (extracts in-place)
  python extract_tars.py data/waymo/tar

  # Extract using 64 CPU cores and keep original tar files
  python extract_tars.py data/waymo/tar --num-workers 64 --keep-tar
        """
    )

    parser.add_argument(
        'tar_dir',
        type=str,
        help='Directory containing tar files to extract'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help=f'Number of parallel workers (default: all {cpu_count()} CPUs)'
    )

    parser.add_argument(
        '--keep-tar',
        action='store_true',
        help='Keep original tar files after extraction (default: remove them)'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.tar_dir):
        print(f"Error: Directory '{args.tar_dir}' does not exist!")
        return 1

    # Set number of workers
    num_workers = args.num_workers if args.num_workers else cpu_count()

    # Find all tar files
    print(f"Searching for tar files in '{args.tar_dir}'...")
    tar_files = find_tar_files(args.tar_dir)

    if not tar_files:
        print(f"No tar files found in '{args.tar_dir}'")
        return 0

    print(f"Found {len(tar_files)} tar file(s)")
    print(f"Using {num_workers} CPU core(s)")
    print(f"Extracting in-place to: {args.tar_dir}")
    print(f"Keep original tars: {args.keep_tar}")
    print("-" * 60)

    # Prepare arguments for parallel processing
    extract_args = [(tar_path, args.keep_tar) for tar_path in tar_files]

    # Extract files in parallel
    success_count = 0
    failed_count = 0

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(extract_tar, extract_args),
            total=len(tar_files),
            desc="Extracting",
            unit="file"
        ))

    # Print results
    print("-" * 60)
    for success, tar_name, message in results:
        if success:
            success_count += 1
            print(f"✓ {tar_name}: {message}")
        else:
            failed_count += 1
            print(f"✗ {tar_name}: {message}")

    # Print summary
    print("-" * 60)
    print(f"Completed: {success_count} successful, {failed_count} failed")

    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    exit(main())