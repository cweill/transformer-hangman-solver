import json
import random
import re
import string
from collections import Counter
from typing import List, Optional, Set

from tqdm import tqdm

# Remove requests import since we're not using an API


class LocalHangmanGame:
    """
    Local implementation of Hangman game for testing.
    Uses test_dictionary.txt for word selection.
    """

    def __init__(self, test_dictionary_path: str = "test_dictionary.txt"):
        self.test_words = self.load_dictionary(test_dictionary_path)
        self.current_word = ""
        self.guessed_letters = set()
        self.incorrect_guesses = 0
        self.max_incorrect_guesses = 6
        self.game_over = False
        self.won = False

    def load_dictionary(self, path: str) -> List[str]:
        """Load test dictionary words from file"""
        try:
            with open(path, "r") as f:
                return [word.strip().lower() for word in f.readlines()]
        except FileNotFoundError:
            print(f"Test dictionary file not found at {path}. Using sample words.")
            return ["apple", "banana", "orange", "grape", "melon"]

    def start_game(self) -> dict:
        """Start a new game with a random word from test dictionary"""
        self.current_word = random.choice(self.test_words)
        self.guessed_letters = set()
        self.incorrect_guesses = 0
        self.game_over = False
        self.won = False

        return {
            "current_word": "_" * len(self.current_word),
            "guessed_letters": [],
            "incorrect_guesses": 0,
            "game_over": False,
            "won": False,
        }

    def make_guess(self, letter: str) -> dict:
        """Make a guess and return updated game state"""
        if self.game_over:
            return self.get_game_state()

        letter = letter.lower()
        self.guessed_letters.add(letter)

        if letter in self.current_word:
            # Correct guess
            pass
        else:
            # Incorrect guess
            self.incorrect_guesses += 1

        # Check if game is over
        if self.incorrect_guesses >= self.max_incorrect_guesses:
            self.game_over = True
            self.won = False
        elif all(letter in self.guessed_letters for letter in self.current_word):
            self.game_over = True
            self.won = True

        return self.get_game_state()

    def get_game_state(self) -> dict:
        """Get current game state"""
        current_display = ""
        for letter in self.current_word:
            if letter in self.guessed_letters:
                current_display += letter
            else:
                current_display += "_"

        return {
            "current_word": current_display,
            "guessed_letters": list(self.guessed_letters),
            "incorrect_guesses": self.incorrect_guesses,
            "game_over": self.game_over,
            "won": self.won,
        }

    def get_actual_word(self) -> str:
        """Get the actual word being guessed (for debugging/testing)"""
        return self.current_word


class HangmanSolver:
    """Base class for Hangman solving algorithms"""

    def __init__(self, dictionary_path: str = "training_dictionary.txt"):
        """
        Initialize the solver with a training dictionary.

        Args:
            dictionary_path: Path to the training dictionary file
        """
        self.dictionary = self.load_dictionary(dictionary_path)
        self.full_dictionary_common_letter_sorted = self.calculate_letter_frequencies(
            self.dictionary
        )

    def load_dictionary(self, path: str) -> List[str]:
        """Load dictionary words from file"""
        with open(path, "r") as f:
            return [word.strip().lower() for word in f.readlines()]

    def calculate_letter_frequencies(self, words: List[str]) -> List[str]:
        """
        Calculate letter frequencies across all words and return sorted list.

        Args:
            words: List of words to analyze

        Returns:
            List of letters sorted by frequency (most common first)
        """
        letter_counter = Counter()
        for word in words:
            for letter in set(word):  # Count each letter once per word
                if letter in string.ascii_lowercase:
                    letter_counter[letter] += 1

        # Sort letters by frequency (descending)
        sorted_letters = sorted(
            letter_counter.keys(), key=lambda x: letter_counter[x], reverse=True
        )
        return sorted_letters

    def guess(self, current_word: str, guessed_letters: Set[str]) -> str:
        """
        Main guessing function - to be implemented by different strategies.

        Args:
            current_word: Current state of word with underscores (e.g., "a__le")
            guessed_letters: Set of already guessed letters

        Returns:
            Next letter to guess
        """
        raise NotImplementedError("Subclasses must implement the guess method")


class BaselineHangmanSolver(HangmanSolver):
    """
    Baseline algorithm that matches masked string to dictionary words
    and guesses based on letter frequency in matching words.
    """

    def guess(self, current_word: str, guessed_letters: Set[str]) -> str:
        """
        Baseline guessing strategy:
        1. Find all dictionary words matching the current pattern
        2. Calculate letter frequencies in matching words
        3. Guess the most frequent unguessed letter
        4. If no matches, use full dictionary frequencies
        """
        # Find all words matching the current pattern
        matching_words = self.find_matching_words(current_word)

        if matching_words:
            # Calculate letter frequencies in matching words
            letter_frequencies = self.calculate_letter_frequencies_in_words(
                matching_words, guessed_letters
            )

            # Return the most frequent unguessed letter
            for letter, _ in letter_frequencies:
                if letter not in guessed_letters:
                    return letter

        # No matching words or all letters guessed - use full dictionary
        for letter in self.full_dictionary_common_letter_sorted:
            if letter not in guessed_letters:
                return letter

        # This shouldn't happen if dictionary is proper
        return "a"

    def is_all_underscores(self, pattern: str) -> bool:
        return all(char == "_" for char in pattern)

    def find_matching_words(self, pattern: str) -> List[str]:
        """
        Find all dictionary words that match the given pattern.

        Args:
            pattern: Word pattern with underscores (e.g., "a__le")

        Returns:
            List of matching words
        """
        matching_words = []
        pattern_length = len(pattern)

        if self.is_all_underscores(pattern):
            return [w for w in self.dictionary if len(w) == pattern_length]

        regex = re.compile(f"{pattern.replace('_', '[a-z]')}")
        for word in self.dictionary:
            if len(word) != pattern_length:
                continue
            if regex.match(word):
                matching_words.append(word)

        if len(matching_words) > 0:
            return matching_words

        regex = re.compile(f"({pattern.replace('_', '[a-z]')})")
        for word in self.dictionary:
            if len(word) <= pattern_length:
                continue

            match = regex.search(word)
            if match:
                matching_words.append(match.group(1))

        return matching_words

    def calculate_letter_frequencies_in_words(
        self, words: List[str], guessed_letters: Set[str]
    ) -> List[tuple]:
        """
        Calculate letter frequencies in a list of words, excluding guessed letters.

        Args:
            words: List of words to analyze
            guessed_letters: Letters to exclude from frequency calculation

        Returns:
            List of (letter, frequency) tuples sorted by frequency
        """
        letter_counter = Counter()

        for word in words:
            for letter in set(word):
                if letter in string.ascii_lowercase and letter not in guessed_letters:
                    letter_counter[letter] += 1

        return letter_counter.most_common()


class BetterHangmanSolver(BaselineHangmanSolver):
    def guess(self, current_word: str, guessed_letters: Set[str]) -> str:
        """
        Better guessing strategy:
        1. Find all dictionary words matching the current pattern
        2. Calculate letter frequencies in matching words
        3. Guess the most frequent unguessed letter
        4. If no matches, use full dictionary frequencies
        """
        # Find all words matching the current pattern
        matching_words = self.find_matching_words(current_word)

        if matching_words:
            # Calculate letter frequencies in matching words
            letter_frequencies = self.calculate_letter_frequencies_in_words(
                matching_words, guessed_letters
            )

            # Return the most frequent unguessed letter
            for letter, _ in letter_frequencies:
                if letter not in guessed_letters:
                    return letter

        # No matching words or all letters guessed - use full dictionary
        for letter in self.full_dictionary_common_letter_sorted:
            if letter not in guessed_letters:
                return letter

        # This shouldn't happen if dictionary is proper
        raise ValueError("No matching words or all letters guessed")


class GameRunner:
    """Class to run multiple Hangman games and track statistics"""

    def __init__(self, solver: HangmanSolver, game: LocalHangmanGame):
        self.solver = solver
        self.game = game
        self.games_played = 0
        self.games_won = 0

    def play_game(self) -> dict:
        """Play a single game of Hangman"""
        # Start new game
        game_state = self.game.start_game()
        guessed_letters = set()

        while not game_state["game_over"]:
            # Get next guess from solver
            current_word = game_state["current_word"]
            next_guess = self.solver.guess(current_word, guessed_letters)

            # Make the guess
            guessed_letters.add(next_guess)

            # Update game state
            game_state = self.game.make_guess(next_guess)

        # Record game results
        result = {
            "won": game_state["won"],
            "word": self.game.get_actual_word(),
            "incorrect_guesses": game_state["incorrect_guesses"],
        }

        self.games_played += 1
        if result["won"]:
            self.games_won += 1

        return result

    def play_games(self, num_games: int = 1000) -> dict:
        """Play multiple games and return statistics"""
        print(f"Playing {num_games} games...")

        for i in (pbar := tqdm(range(num_games))):
            self.play_game()

            # Calculate win rate
            win_rate = (
                self.games_won / self.games_played if self.games_played > 0 else 0
            )
            pbar.set_postfix_str(f"Win rate: {win_rate:.2%}")

        stats = {
            "games_played": self.games_played,
            "games_won": self.games_won,
            "win_rate": win_rate,
        }

        print(f"Win rate: {stats['win_rate']:.2%}")

        return stats


# Example usage and testing
if __name__ == "__main__":
    # Initialize components
    game = LocalHangmanGame("test_dictionary.txt")
    # solver = BaselineHangmanSolver("training_dictionary.txt")
    solver = BetterHangmanSolver("training_dictionary.txt")
    runner = GameRunner(solver, game)

    # Run games to compute accuracy
    stats = runner.play_games(1000)
