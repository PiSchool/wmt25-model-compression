#!/usr/bin/env python
"""
Data Manager for WMT25 Model Compression Constrained Task

Handles downloading, preprocessing, and organizing training and test data
for the three constrained language pairs.
"""

import argparse
import json
import subprocess as sp
import logging as LOG
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from modelzip.constrained_config import (
    CONSTRAINED_TASK, DATA_DIR, WORK_DIR, LANGS_MAP,
    get_constrained_pairs, setup_directories
)

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataManager:
    """Manages data collection, preprocessing and organization for constrained task"""
    
    def __init__(self, work_dir: Path = WORK_DIR):
        self.work_dir = Path(work_dir)
        self.data_dir = self.work_dir / "data"
        self.tests_dir = self.work_dir / "tests"
        setup_directories()
        self.data_dir.mkdir(exist_ok=True)
        self.tests_dir.mkdir(exist_ok=True)
        
    def download_test_data(self, lang_pairs: Optional[List[str]] = None) -> Dict[str, Dict[str, Path]]:
        """Download test data for specified language pairs"""
        lang_pairs = lang_pairs or get_constrained_pairs()
        test_files = {}
        
        for pair in lang_pairs:
            if pair not in CONSTRAINED_TASK["data_sources"]:
                LOG.warning(f"No data source configured for {pair}")
                continue
                
            test_files[pair] = {}
            src, tgt = pair.split("-")
            pair_dir = self.tests_dir / pair
            pair_dir.mkdir(exist_ok=True)
            
            test_sources = CONSTRAINED_TASK["data_sources"][pair]["test"]
            for test_name, get_cmd in test_sources.items():
                src_file = pair_dir / f"{test_name}.{src}-{tgt}.{src}"
                ref_file = pair_dir / f"{test_name}.{src}-{tgt}.{tgt}"
                
                if self._files_exist_and_valid(src_file, ref_file):
                    LOG.info(f"Test files exist for {pair}:{test_name}")
                    test_files[pair][test_name] = {"src": src_file, "ref": ref_file}
                    continue
                
                try:
                    LOG.info(f"Downloading {test_name} for {pair} via: {get_cmd}")
                    lines = sp.check_output(get_cmd, shell=True, text=True).strip().replace("\r", "").split("\n")
                    lines = [x.strip().split("\t") for x in lines]
                    
                    if "mtdata" in get_cmd:
                        lines = [x[:2] for x in lines]
                    
                    self._validate_parallel_data(lines, test_name)
                    srcs, refs = zip(*lines)
                    
                    src_file.write_text("\n".join(srcs), encoding="utf-8")
                    ref_file.write_text("\n".join(refs), encoding="utf-8")
                    
                    test_files[pair][test_name] = {"src": src_file, "ref": ref_file}
                    LOG.info(f"Created test files: {src_file}, {ref_file}")
                    
                except Exception as e:
                    LOG.error(f"Failed to download {test_name} for {pair}: {e}")
                    
        return test_files
    
    def download_training_data(self, lang_pairs: Optional[List[str]] = None) -> Dict[str, Path]:
        """Download training data using mtdata for specified language pairs"""
        lang_pairs = lang_pairs or get_constrained_pairs()
        train_files = {}
        
        for pair in lang_pairs:
            if pair not in CONSTRAINED_TASK["data_sources"]:
                LOG.warning(f"No training data source configured for {pair}")
                continue
                
            pair_dir = self.data_dir / "train" / pair
            pair_dir.mkdir(parents=True, exist_ok=True)
            
            train_file = pair_dir / f"train.{pair}.tsv"
            if train_file.exists() and train_file.stat().st_size > 0:
                LOG.info(f"Training data exists for {pair}: {train_file}")
                train_files[pair] = train_file
                continue
            
            try:
                # Get the dataset name from config
                dataset_name = CONSTRAINED_TASK["data_sources"][pair]["train"]
                LOG.info(f"Downloading training data for {pair} using dataset: {dataset_name}")
                
                # Use mtdata to download specific dataset
                src, tgt = pair.split("-")
                cmd = f"mtdata get -tr {dataset_name} -l {src}-{tgt} -o {pair_dir} --merge"
                LOG.info(f"Running command: {cmd}")
                sp.check_call(cmd, shell=True)
                
                # Find the downloaded files (mtdata creates separate src/tgt files)
                src, tgt = pair.split("-")
                src_file = pair_dir / f"train.{src}"
                tgt_file = pair_dir / f"train.{tgt}"
                
                if src_file.exists() and tgt_file.exists():
                    # Combine source and target files into TSV format
                    LOG.info(f"Combining {src_file} and {tgt_file} into {train_file}")
                    with open(src_file, 'r', encoding='utf-8') as sf, \
                         open(tgt_file, 'r', encoding='utf-8') as tf, \
                         open(train_file, 'w', encoding='utf-8') as outf:
                        for src_line, tgt_line in zip(sf, tf):
                            src_line = src_line.strip()
                            tgt_line = tgt_line.strip()
                            if src_line and tgt_line:  # Skip empty lines
                                outf.write(f"{src_line}\t{tgt_line}\n")
                    
                    train_files[pair] = train_file
                    LOG.info(f"Created training data: {train_file}")
                else:
                    LOG.error(f"Source or target files not found: {src_file}, {tgt_file}")
                    
            except Exception as e:
                LOG.error(f"Failed to download training data for {pair}: {e}")
                # Try alternative approach - download any available dataset for this language pair
                try:
                    LOG.info(f"Trying alternative download for {pair}...")
                    src, tgt = pair.split("-")
                    cmd = f"mtdata get -l {src}-{tgt} -tr Statmt-news_commentary-17-{src}-{tgt} -o {pair_dir} --merge"
                    sp.check_call(cmd, shell=True)
                    
                    # Check for downloaded files again (mtdata format)
                    src, tgt = pair.split("-")
                    src_file = pair_dir / f"train.{src}"
                    tgt_file = pair_dir / f"train.{tgt}"
                    
                    if src_file.exists() and tgt_file.exists():
                        # Combine source and target files into TSV format
                        with open(src_file, 'r', encoding='utf-8') as sf, \
                             open(tgt_file, 'r', encoding='utf-8') as tf, \
                             open(train_file, 'w', encoding='utf-8') as outf:
                            for src_line, tgt_line in zip(sf, tf):
                                src_line = src_line.strip()
                                tgt_line = tgt_line.strip()
                                if src_line and tgt_line:
                                    outf.write(f"{src_line}\t{tgt_line}\n")
                        
                        train_files[pair] = train_file
                        LOG.info(f"Created training data (alternative): {train_file}")
                except Exception as e2:
                    LOG.error(f"Alternative download also failed for {pair}: {e2}")
                
        return train_files
    
    def preprocess_training_data(self, train_files: Dict[str, Path], 
                               max_length: int = 512, min_length: int = 5) -> Dict[str, Path]:
        """Preprocess training data: filter, clean, deduplicate"""
        processed_files = {}
        
        for pair, train_file in train_files.items():
            if not train_file.exists():
                LOG.warning(f"Training file does not exist: {train_file}")
                continue
                
            processed_file = train_file.with_suffix(".processed.tsv")
            if processed_file.exists():
                LOG.info(f"Processed file exists: {processed_file}")
                processed_files[pair] = processed_file
                continue
            
            LOG.info(f"Preprocessing training data for {pair}")
            src, tgt = pair.split("-")
            
            # Read data
            df = pd.read_csv(train_file, sep="\t", header=None, names=["source", "target"])
            original_size = len(df)
            
            # Clean and filter
            df = self._clean_parallel_data(df, min_length, max_length)
            
            # Deduplicate
            df = df.drop_duplicates()
            
            # Save processed data
            df.to_csv(processed_file, sep="\t", index=False, header=False)
            processed_files[pair] = processed_file
            
            LOG.info(f"Processed {pair}: {original_size} -> {len(df)} sentences")
            
        return processed_files
    
    def create_data_splits(self, processed_files: Dict[str, Path], 
                          dev_size: int = 2000) -> Dict[str, Dict[str, Path]]:
        """Create train/dev splits from processed training data"""
        splits = {}
        
        for pair, processed_file in processed_files.items():
            pair_splits = {}
            pair_dir = processed_file.parent
            
            # Read processed data
            df = pd.read_csv(processed_file, sep="\t", header=None, names=["source", "target"])
            
            # Create dev split
            if len(df) > dev_size:
                dev_df = df.sample(n=dev_size, random_state=42)
                train_df = df.drop(dev_df.index)
            else:
                dev_df = df.sample(n=min(1000, len(df)//10), random_state=42)
                train_df = df.drop(dev_df.index)
            
            # Save splits
            train_file = pair_dir / f"train.{pair}.final.tsv"
            dev_file = pair_dir / f"dev.{pair}.tsv"
            
            train_df.to_csv(train_file, sep="\t", index=False, header=False)
            dev_df.to_csv(dev_file, sep="\t", index=False, header=False)
            
            pair_splits = {"train": train_file, "dev": dev_file}
            splits[pair] = pair_splits
            
            LOG.info(f"Created splits for {pair}: train={len(train_df)}, dev={len(dev_df)}")
            
        return splits
    
    def _files_exist_and_valid(self, *files: Path) -> bool:
        """Check if files exist and are non-empty"""
        return all(f.exists() and f.stat().st_size > 0 for f in files)
    
    def _validate_parallel_data(self, lines: List[List[str]], name: str):
        """Validate parallel data format"""
        n_errs = sum(1 for x in lines if len(x) != 2)
        if n_errs:
            raise ValueError(f"Invalid data format in {name}: {n_errs} errors")
    
    def _clean_parallel_data(self, df: pd.DataFrame, min_length: int, max_length: int) -> pd.DataFrame:
        """Clean parallel data: remove empty, too short/long, invalid sentences"""
        # Ensure columns are strings
        df["source"] = df["source"].astype(str)
        df["target"] = df["target"].astype(str)
        
        # Remove empty or null
        df = df.dropna()
        df = df[df["source"].str.strip() != ""]
        df = df[df["target"].str.strip() != ""]
        df = df[df["source"] != "nan"]
        df = df[df["target"] != "nan"]
        
        # Length filtering
        df = df[df["source"].str.len() >= min_length]
        df = df[df["target"].str.len() >= min_length]
        df = df[df["source"].str.len() <= max_length]
        df = df[df["target"].str.len() <= max_length]
        
        # Remove sentences with extreme length ratios (avoid division by zero)
        src_len = df["source"].str.len()
        tgt_len = df["target"].str.len()
        df = df[tgt_len > 0]  # Remove zero-length targets
        ratio = src_len / tgt_len
        df = df[(ratio >= 0.3) & (ratio <= 3.0)]
        
        return df
    
    def get_data_statistics(self) -> Dict:
        """Get statistics about downloaded and processed data"""
        stats = {"test_data": {}, "train_data": {}}
        
        # Test data statistics
        for pair in get_constrained_pairs():
            pair_dir = self.tests_dir / pair
            if pair_dir.exists():
                stats["test_data"][pair] = {}
                src, tgt = pair.split("-")
                # Look for test files with proper naming pattern
                for test_file in pair_dir.glob(f"*.{pair}.{src}"):
                    test_name = test_file.name.split(".")[0]
                    if test_file.exists():
                        stats["test_data"][pair][test_name] = {
                            "sentences": len(test_file.read_text().splitlines())
                        }
        
        # Training data statistics
        for pair in get_constrained_pairs():
            train_dir = self.data_dir / "train" / pair
            if train_dir.exists():
                stats["train_data"][pair] = {}
                
                # Look for the original training file (before processing)
                original_file = train_dir / f"train.{pair}.tsv"
                if original_file.exists():
                    try:
                        df = pd.read_csv(original_file, sep="\t", header=None)
                        stats["train_data"][pair]["original"] = len(df)
                    except Exception as e:
                        LOG.warning(f"Could not read original file {original_file}: {e}")
                
                # Look for the processed training file
                processed_file = train_dir / f"train.{pair}.processed.tsv"
                if processed_file.exists():
                    try:
                        df = pd.read_csv(processed_file, sep="\t", header=None)
                        stats["train_data"][pair]["processed"] = len(df)
                    except Exception as e:
                        LOG.warning(f"Could not read processed file {processed_file}: {e}")
        
        return stats
    
    def _load_test_data_for_pair(self, lang_pair: str) -> List[Dict[str, str]]:
        """Load test data for a specific language pair in the format expected by evaluator
        
        Args:
            lang_pair: Language pair (e.g., "ces-deu")
            
        Returns:
            List of dictionaries with 'source' and 'target' keys
        """
        test_data = []
        pair_dir = self.tests_dir / lang_pair
        
        if not pair_dir.exists():
            LOG.warning(f"No test directory for {lang_pair}")
            return test_data
        
        src, tgt = lang_pair.split("-")
        
        # Look for test files with the expected pattern
        for test_file in pair_dir.glob(f"*.{lang_pair}.{src}"):
            test_name = test_file.name.split(".")[0]
            ref_file = pair_dir / f"{test_name}.{lang_pair}.{tgt}"
            
            if test_file.exists() and ref_file.exists():
                try:
                    sources = test_file.read_text(encoding='utf-8').strip().split('\n')
                    targets = ref_file.read_text(encoding='utf-8').strip().split('\n')
                    
                    for src_line, tgt_line in zip(sources, targets):
                        if src_line.strip() and tgt_line.strip():
                            test_data.append({
                                "source": src_line.strip(),
                                "target": tgt_line.strip()
                            })
                    
                    LOG.info(f"Loaded {len(test_data)} test samples from {test_name} for {lang_pair}")
                    
                except Exception as e:
                    LOG.error(f"Failed to load test data from {test_file}: {e}")
        
        return test_data
    
    def get_data_statistics(self, lang_pair: str = None) -> Dict:
        """Get statistics about data for a specific language pair or all pairs
        
        Args:
            lang_pair: Optional specific language pair, if None returns stats for all pairs
            
        Returns:
            Dictionary with data statistics
        """
        if lang_pair:
            # Statistics for specific language pair
            stats = {"train_samples": 0, "test_samples": 0}
            
            # Training data count
            train_dir = self.data_dir / "train" / lang_pair
            train_file = train_dir / f"train.{lang_pair}.tsv"
            if train_file.exists():
                try:
                    df = pd.read_csv(train_file, sep="\t", header=None)
                    stats["train_samples"] = len(df)
                except Exception as e:
                    LOG.warning(f"Could not read {train_file}: {e}")
            
            # Test data count  
            test_data = self._load_test_data_for_pair(lang_pair)
            stats["test_samples"] = len(test_data)
            
            return stats
        else:
            # Statistics for all language pairs (existing method)
            return self._get_all_data_statistics()
    
    def _get_all_data_statistics(self) -> Dict:
        """Get statistics about downloaded and processed data for all pairs"""
        stats = {"test_data": {}, "train_data": {}}
        
        # Test data statistics
        for pair in get_constrained_pairs():
            pair_dir = self.tests_dir / pair
            if pair_dir.exists():
                stats["test_data"][pair] = {}
                src, tgt = pair.split("-")
                # Look for test files with proper naming pattern
                for test_file in pair_dir.glob(f"*.{pair}.{src}"):
                    test_name = test_file.name.split(".")[0]
                    if test_file.exists():
                        stats["test_data"][pair][test_name] = {
                            "sentences": len(test_file.read_text().splitlines())
                        }
        
        # Training data statistics
        for pair in get_constrained_pairs():
            train_dir = self.data_dir / "train" / pair
            if train_dir.exists():
                stats["train_data"][pair] = {}
                
                # Look for the original training file (before processing)
                original_file = train_dir / f"train.{pair}.tsv"
                if original_file.exists():
                    try:
                        df = pd.read_csv(original_file, sep="\t", header=None)
                        stats["train_data"][pair]["original"] = len(df)
                    except Exception as e:
                        LOG.warning(f"Could not read original file {original_file}: {e}")
                
                # Look for the processed training file
                processed_file = train_dir / f"train.{pair}.processed.tsv"
                if processed_file.exists():
                    try:
                        df = pd.read_csv(processed_file, sep="\t", header=None)
                        stats["train_data"][pair]["processed"] = len(df)
                    except Exception as e:
                        LOG.warning(f"Could not read processed file {processed_file}: {e}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Data Manager for WMT25 Constrained Task")
    parser.add_argument("-w", "--work-dir", type=Path, default=WORK_DIR, 
                       help="Working directory")
    parser.add_argument("-l", "--lang-pairs", nargs="+", default=get_constrained_pairs(),
                       help="Language pairs to process") 
    parser.add_argument("--test-only", action="store_true", 
                       help="Download test data only")
    parser.add_argument("--train-only", action="store_true",
                       help="Download training data only")
    parser.add_argument("--stats", action="store_true",
                       help="Show data statistics")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sentence length for filtering")
    parser.add_argument("--dev-size", type=int, default=2000,
                       help="Development set size")
    
    args = parser.parse_args()
    
    # Initialize data manager
    data_manager = DataManager(args.work_dir)
    
    if args.stats:
        stats = data_manager.get_data_statistics()
        print(json.dumps(stats, indent=2))
        return
    
    # Download data
    if not args.train_only:
        LOG.info("Downloading test data...")
        test_files = data_manager.download_test_data(args.lang_pairs)
        LOG.info(f"Downloaded test data for {len(test_files)} language pairs")
    
    if not args.test_only:
        LOG.info("Downloading training data...")
        train_files = data_manager.download_training_data(args.lang_pairs)
        
        if train_files:
            LOG.info("Preprocessing training data...")
            processed_files = data_manager.preprocess_training_data(
                train_files, max_length=args.max_length
            )
            
            LOG.info("Creating data splits...")
            splits = data_manager.create_data_splits(processed_files, dev_size=args.dev_size)
            
            LOG.info(f"Processed training data for {len(splits)} language pairs")
    
    LOG.info("Data management completed!")


if __name__ == "__main__":
    main() 