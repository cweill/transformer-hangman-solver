import timeit

import numpy as np
from numba import njit


# @njit(nopython=True, nogil=True)
def masked_words(word: str):
    result = []

    # Get unique characters in the word (excluding '.')
    unique_chars = []
    seen_chars = {}
    for char in word:
        if char != "." and char not in seen_chars:
            unique_chars.append(char)
            seen_chars[char] = True

    # Generate all possible combinations of which characters to mask
    # For n unique characters, we have 2^n possibilities
    for i in range(1, 2 ** len(unique_chars)):  # Start from 1 to exclude no masks
        chars_to_mask = []
        seen_chars = {}

        # Determine which characters to mask based on bit pattern
        for j in range(len(unique_chars)):
            if i & (1 << j):
                char = unique_chars[j]
                if char not in seen_chars:
                    seen_chars[char] = True
                    chars_to_mask.append(char)

        # Create the masked string
        masked = ""
        for char in word:
            if char in chars_to_mask:
                masked += "_"
            else:
                masked += char

        # Add the result
        result.append((masked, word))

    return result


print(masked_words("app"))
print(masked_words("app."))
print(masked_words("xyz"))
print(masked_words("xyz.."))

print(
    timeit.timeit(
        lambda: masked_words("xyzfoooabx.."),
        number=100,
    )
)
