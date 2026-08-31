"""
Plain-lexical text-matching helpers used by notion_mcp_server.py for ticket dedup.
"""
import re

from nltk.stem import PorterStemmer

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "cannot",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in",
    "into", "is", "it", "its", "just", "me", "my", "no", "not", "of", "on", "or",
    "our", "please", "so", "than", "that", "the", "their", "them", "then", "there",
    "this", "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
}

porter = PorterStemmer()


def stem(word: str) -> str:
    """So e.g. "returns"/"return" or "crashing"/"crash" overlap for dedup matching."""
    return porter.stem(word)


def words(text: str) -> set:
    return {stem(w) for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS and len(w) > 1}


def dice_overlap(a_words: set, b_words: set) -> float:
    """2*|intersection| / (|a|+|b|), normalized for length."""
    if not a_words or not b_words:
        return 0.0
    return 2 * len(a_words & b_words) / (len(a_words) + len(b_words))
