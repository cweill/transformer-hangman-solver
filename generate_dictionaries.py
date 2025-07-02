#!/usr/bin/env python3
"""
Script to generate training_dictionary.txt and test_dictionary.txt
from NLTK's word corpus with random split and no overlap.
"""

import random
from typing import List, Set

import nltk
from nltk.corpus import words


def download_nltk_data():
    """Download required NLTK data if not already present."""
    try:
        nltk.data.find("corpora/words")
    except LookupError:
        print("Downloading NLTK words corpus...")
        nltk.download("words")
        print("Download complete!")


def get_english_words() -> List[str]:
    """
    Get English words from NLTK corpus.

    Returns:
        List of lowercase English words, no duplicates
    """

    # Get all words and convert to lowercase
    all_words = [word.lower() for word in words.words()]

    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in all_words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)

    print(f"Found {len(unique_words)} unique English words")
    return unique_words


N = len(get_english_words())


def split_words_randomly(
    words: List[str], train_size: int = N - 10000, test_size: int = 10000
) -> tuple[List[str], List[str]]:
    """
    Randomly split words into training and test sets with no overlap.

    Args:
        words: List of words to split
        train_size: Number of words for training set
        test_size: Number of words for test set

    Returns:
        Tuple of (training_words, test_words)
    """
    if len(words) < train_size + test_size:
        raise ValueError(
            f"Not enough words available. Need {train_size + test_size}, but only have {len(words)}"
        )

    # Shuffle words randomly
    shuffled_words = words.copy()
    random.shuffle(shuffled_words)

    # Split into training and test sets
    training_words = shuffled_words[:train_size]
    test_words = shuffled_words[train_size : train_size + test_size]

    return training_words, test_words


def write_dictionary_file(words: List[str], filename: str):
    """
    Write words to a dictionary file, one word per line.

    Args:
        words: List of words to write
        filename: Output filename
    """
    with open(filename, "w", encoding="utf-8") as f:
        for word in words:
            f.write(word + "\n")

    print(f"Written {len(words)} words to {filename}")


def main():
    """Main function to generate the dictionary files."""
    print("Generating training and test dictionaries...")

    # Set random seed for reproducibility
    random.seed(42)

    # Download NLTK data if needed
    download_nltk_data()

    # Get English words
    words = get_english_words()

    # Split into training and test sets
    print(f"Splitting {len(words)} words into training and test sets...")
    training_words, test_words = split_words_randomly(words)

    # Write files
    write_dictionary_file(training_words, "training_dictionary.txt")
    write_dictionary_file(test_words, "test_dictionary.txt")

    # Verify no overlap
    training_set = set(training_words)
    test_set = set(test_words)
    overlap = training_set.intersection(test_set)

    if overlap:
        print(f"WARNING: Found {len(overlap)} overlapping words!")
    else:
        print("✓ No overlap between training and test sets")

    # Print statistics
    print(f"\nFinal Statistics:")
    print(f"Training dictionary: {len(training_words)} words")
    print(f"Test dictionary: {len(test_words)} words")
    print(f"Total unique words used: {len(training_words) + len(test_words)}")

    # Show some sample words
    print(f"\nSample training words: {training_words[:5]}")
    print(f"Sample test words: {test_words[:5]}")


if __name__ == "__main__":
    main()
