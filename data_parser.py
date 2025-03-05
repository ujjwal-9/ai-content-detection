import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class TextSample:
    """Class to hold a single text sample and its metadata"""

    text: str
    source: str  # 'human', 'gpt', 'claude', etc.
    dataset: str  # 'wp', 'reuter', 'essay', etc.
    id: str
    logprobs: Dict = None


class GhostbusterDataset:
    def __init__(self, data_dir: str = "ghostbuster-data"):
        """Initialize the dataset parser

        Args:
            data_dir: Path to the ghostbuster-data directory
        """
        self.data_dir = Path(data_dir)
        self.datasets = ["wp", "reuter", "essay", "other"]
        self.sources = {
            "human": "human",
            "gpt": "gpt",
            "gpt_writing": "gpt",
            "gpt_semantic": "gpt",
            "gpt_prompt1": "gpt",
            "gpt_prompt2": "gpt",
            "claude": "claude",
        }

    def load_text(self, file_path: Path) -> str:
        """Load text from a file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except UnicodeDecodeError:
            # Try different encoding if utf-8 fails
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read().strip()

    def load_logprobs(self, logprob_path: Path) -> Dict:
        """Load logprobs from a json file if it exists"""
        if logprob_path.exists():
            with open(logprob_path, "r") as f:
                return json.load(f)
        return None

    def get_samples(
        self, dataset_type: str = None, source_type: str = None
    ) -> List[TextSample]:
        """Get text samples from the dataset

        Args:
            dataset_type: Type of dataset to load ('wp', 'reuter', 'essay', 'other'). If None, load all.
            source_type: Type of source to load ('human', 'gpt', 'claude'). If None, load all.

        Returns:
            List of TextSample objects
        """
        samples = []

        # Determine which datasets to process
        datasets_to_process = [dataset_type] if dataset_type else self.datasets

        for dataset in datasets_to_process:
            dataset_path = self.data_dir / dataset
            if not dataset_path.exists():
                continue

            # Process each source directory
            for source_dir in dataset_path.iterdir():
                if not source_dir.is_dir() or source_dir.name == "prompts":
                    continue

                # Check if we should process this source
                source = self.sources.get(source_dir.name)
                if source_type and source != source_type:
                    continue

                # Process text files
                for text_file in source_dir.glob("*.txt"):
                    if text_file.name == "README.txt":
                        continue

                    # Load text
                    text = self.load_text(text_file)

                    # Check for logprobs
                    logprob_path = source_dir / "logprobs" / f"{text_file.stem}.json"
                    logprobs = self.load_logprobs(logprob_path)

                    # Create sample
                    sample = TextSample(
                        text=text,
                        source=source,
                        dataset=dataset,
                        id=text_file.stem,
                        logprobs=logprobs,
                    )
                    samples.append(sample)

        return samples

    def get_dataset_stats(self) -> Dict:
        """Get statistics about the dataset"""
        stats = {
            "total_samples": 0,
            "by_dataset": {},
            "by_source": {},
            "by_dataset_and_source": {},
        }

        samples = self.get_samples()
        stats["total_samples"] = len(samples)

        # Count by dataset
        for dataset in self.datasets:
            dataset_samples = [s for s in samples if s.dataset == dataset]
            stats["by_dataset"][dataset] = len(dataset_samples)

        # Count by source
        for source in set(self.sources.values()):
            source_samples = [s for s in samples if s.source == source]
            stats["by_source"][source] = len(source_samples)

        # Count by dataset and source
        for dataset in self.datasets:
            stats["by_dataset_and_source"][dataset] = {}
            for source in set(self.sources.values()):
                count = len(
                    [s for s in samples if s.dataset == dataset and s.source == source]
                )
                stats["by_dataset_and_source"][dataset][source] = count

        return stats


def main():
    """Example usage of the dataset parser"""
    dataset = GhostbusterDataset()

    # Print dataset statistics
    stats = dataset.get_dataset_stats()
    print("\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")

    print("\nSamples by dataset:")
    for dataset, count in stats["by_dataset"].items():
        print(f"{dataset}: {count}")

    print("\nSamples by source:")
    for source, count in stats["by_source"].items():
        print(f"{source}: {count}")

    # Example: Load only human samples from wp dataset
    wp_human_samples = dataset.get_samples(dataset_type="wp", source_type="human")
    print(f"\nNumber of human WP samples: {len(wp_human_samples)}")

    if wp_human_samples:
        # Print first sample
        sample = wp_human_samples[0]
        print("\nExample sample:")
        print(f"ID: {sample.id}")
        print(f"Source: {sample.source}")
        print(f"Dataset: {sample.dataset}")
        print(f"Text preview: {sample.text[:200]}...")


if __name__ == "__main__":
    main()
