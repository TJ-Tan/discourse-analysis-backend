"""
Quick script to get proper nouns list for frontend
Run: python3 get_proper_nouns.py
"""

from proper_nouns_config import get_proper_noun_phrases
import json

if __name__ == "__main__":
    nouns = get_proper_noun_phrases()
    print(json.dumps(nouns, indent=2))

