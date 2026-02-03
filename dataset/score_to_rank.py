#!/usr/bin/env python3
"""
Score to rank converter helper.

Converts response scores to preference rankings with tie handling.
For example: A:7.0 B:7.0 C:8.0 D:7.5 [OK]ranks [3, 3, 1, 2]
where 1 is best, and tied responses get the same rank.

This module is used by prepare_data.py during dataset preparation
to convert raw scores into preference rankings.

Usage:
    from dataset.score_to_rank import process_dataset, compute_preference_ranks
    
    # Process entire dataset
    dataset = process_dataset(raw_dataset)
    
    # Or compute ranks for a single example
    ranks = compute_preference_ranks({'A': 7.0, 'B': 7.0, 'C': 8.0, 'D': 7.5})
"""

from typing import Dict, List, Tuple
from collections import defaultdict
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from cache_utils import get_dataset_cache_dir


def compute_preference_ranks(scores: Dict[str, float]) -> List[int]:
    """
    Compute preference rankings from response scores.
    
    Args:
        scores: Dictionary mapping response labels to scores
                e.g., {'A': 7.0, 'B': 7.0, 'C': 8.0, 'D': 7.5}
    
    Returns:
        List of ranks [rank_A, rank_B, rank_C, rank_D]
        where 1 is best
        Tied scores get the same rank
        Zero scores are ranked last (worst)
        
        e.g., [3, 3, 1, 2] for A:7.0, B:7.0, C:8.0, D:7.5
        e.g., [2, 4, 1, 2] for A:6.0, B:0.0, C:8.0, D:6.0
    
    Example:
        >>> compute_preference_ranks({'A': 7.0, 'B': 7.0, 'C': 8.0, 'D': 7.5})
        [3, 3, 1, 2]
        >>> compute_preference_ranks({'A': 6.0, 'B': 0.0, 'C': 8.0, 'D': 6.0})
        [2, 4, 1, 2]
    """
    # Initialize ranks as 0
    ranks = [0, 0, 0, 0]
    label_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    # Separate valid scores (>0) from zero scores
    valid_score_buckets = defaultdict(list)
    zero_score_labels = []
    
    for label, score in scores.items():
        if score > 0:  # Valid response
            valid_score_buckets[score].append(label)
        else:  # Zero score - invalid/worst
            zero_score_labels.append(label)
    
    if not valid_score_buckets and not zero_score_labels:
        return ranks
    
    # Sort valid buckets by score (descending)
    sorted_scores = sorted(valid_score_buckets.keys(), reverse=True)
    
    # Assign ranks to valid responses (1 is best)
    current_rank = 1
    for score in sorted_scores:
        labels = valid_score_buckets[score]
        for label in labels:
            ranks[label_to_idx[label]] = current_rank
        # Next rank accounts for number of items in this tier
        current_rank += len(labels)
    
    # Assign worst rank to zero-score responses
    # They all get the same rank (last place)
    if zero_score_labels:
        worst_rank = current_rank
        for label in zero_score_labels:
            ranks[label_to_idx[label]] = worst_rank
    
    return ranks


def extract_scores_from_example(example: dict) -> Dict[str, float]:
    """
    Extract scores from a dataset example.
    
    Args:
        example: Dataset example with score_A, score_B, score_C, score_D fields
    
    Returns:
        Dictionary mapping labels to scores
    """
    scores = {}
    for label in ['A', 'B', 'C', 'D']:
        score_key = f'score_{label}'
        if score_key in example:
            try:
                scores[label] = float(example[score_key])
            except (ValueError, TypeError):
                scores[label] = 0.0
    
    return scores


def add_preference_ranking(example: dict) -> dict:
    """
    Add preference_ranks field to dataset example.
    
    Args:
        example: Dataset example
    
    Returns:
        Example with added 'rank_A', 'rank_B', 'rank_C', 'rank_D' fields
    """
    scores = extract_scores_from_example(example)
    ranks = compute_preference_ranks(scores)
    
    return {
        **example,
        'rank_A': ranks[0],
        'rank_B': ranks[1],
        'rank_C': ranks[2],
        'rank_D': ranks[3],
    }


def get_preference_pairs(ranks: List[int]) -> List[Tuple[str, str]]:
    """
    Extract all pairwise preferences from preference rankings.
    
    Args:
        ranks: List of ranks [rank_A, rank_B, rank_C, rank_D]
               where lower rank number means better (1 is best)
    
    Returns:
        List of (preferred, dispreferred) tuples
        e.g., [('C', 'D'), ('C', 'A'), ('C', 'B'), ('D', 'A'), ('D', 'B')]
    
    Example:
        >>> get_preference_pairs([3, 3, 1, 2])  # C:8.0, D:7.5, A=B:7.0
        [('C', 'D'), ('C', 'A'), ('C', 'B'), ('D', 'A'), ('D', 'B')]
        >>> get_preference_pairs([2, 4, 1, 2])  # C:8.0, A=D:6.0, B:0.0
        [('C', 'A'), ('C', 'D'), ('C', 'B'), ('A', 'B'), ('D', 'B')]
    """
    labels = ['A', 'B', 'C', 'D']
    pairs = []
    
    for i in range(4):
        if ranks[i] == 0:  # Skip if no rank assigned (shouldn't happen now)
            continue
        for j in range(4):
            if ranks[j] == 0:  # Skip if no rank assigned
                continue
            # i is better than j if rank[i] < rank[j] (lower rank = better)
            if ranks[i] < ranks[j]:
                pairs.append((labels[i], labels[j]))
    
    return pairs


def count_preference_types(ranks: List[int]) -> Dict[str, int]:
    """
    Count the number of responses in each preference tier.
    
    Args:
        ranks: List of ranks [rank_A, rank_B, rank_C, rank_D]
    
    Returns:
        Dictionary with statistics about the preference structure
    """
    valid_ranks = [r for r in ranks if r > 0]
    
    if not valid_ranks:
        return {
            'num_tiers': 0,
            'num_responses': 0,
            'num_pairs': 0,
            'max_tier_size': 0
        }
    
    unique_ranks = len(set(valid_ranks))
    rank_counts = defaultdict(int)
    for r in valid_ranks:
        rank_counts[r] += 1
    
    return {
        'num_tiers': unique_ranks,
        'num_responses': len(valid_ranks),
        'num_pairs': len(get_preference_pairs(ranks)),
        'max_tier_size': max(rank_counts.values())
    }


def process_dataset(dataset, use_cache=True):
    """
    Process entire dataset to add preference rankings.
    
    Args:
        dataset: HuggingFace Dataset or DatasetDict
        use_cache: Whether to use HuggingFace dataset caching (default: True)
    
    Returns:
        Processed dataset with rank_A, rank_B, rank_C, rank_D columns
        
    Note:
        Caching uses the dataset's built-in cache mechanism which stores
        processed data in the HuggingFace cache directory (from cache_utils).
    """
    cache_dir = get_dataset_cache_dir() if use_cache else None
    
    return dataset.map(
        add_preference_ranking,
        desc="Computing preference rankings",
        load_from_cache_file=use_cache,
    )


if __name__ == '__main__':
    # Test examples
    print("Testing Score to Rank Converter\n" + "=" * 60)
    
    # Test 1: Example from requirements (A:7.0, B:7.0, C:8.0, D:7.5)
    test_scores_1 = {'A': 7.0, 'B': 7.0, 'C': 8.0, 'D': 7.5}
    ranks_1 = compute_preference_ranks(test_scores_1)
    print(f"Test 1: {test_scores_1}")
    print(f"  Ranks: {ranks_1}")
    print(f"  Pairs: {get_preference_pairs(ranks_1)}")
    print(f"  Stats: {count_preference_types(ranks_1)}\n")
    
    # Test 2: All different scores
    test_scores_2 = {'A': 3.0, 'B': 5.0, 'C': 7.0, 'D': 9.0}
    ranks_2 = compute_preference_ranks(test_scores_2)
    print(f"Test 2: {test_scores_2}")
    print(f"  Ranks: {ranks_2}")
    print(f"  Pairs: {get_preference_pairs(ranks_2)}")
    print(f"  Stats: {count_preference_types(ranks_2)}\n")
    
    # Test 3: All same scores
    test_scores_3 = {'A': 5.0, 'B': 5.0, 'C': 5.0, 'D': 5.0}
    ranks_3 = compute_preference_ranks(test_scores_3)
    print(f"Test 3: {test_scores_3}")
    print(f"  Ranks: {ranks_3}")
    print(f"  Pairs: {get_preference_pairs(ranks_3)}")
    print(f"  Stats: {count_preference_types(ranks_3)}\n")
    
    # Test 4: Some zero scores (missing responses)
    test_scores_4 = {'A': 6.0, 'B': 0.0, 'C': 8.0, 'D': 6.0}
    ranks_4 = compute_preference_ranks(test_scores_4)
    print(f"Test 4: {test_scores_4}")
    print(f"  Ranks: {ranks_4}")
    print(f"  Pairs: {get_preference_pairs(ranks_4)}")
    print(f"  Stats: {count_preference_types(ranks_4)}\n")
    
    print("=" * 60)
