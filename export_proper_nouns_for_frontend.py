"""
Export proper nouns to JavaScript array format for frontend
"""

from proper_nouns_config import get_proper_noun_phrases
import json

if __name__ == "__main__":
    nouns = get_proper_noun_phrases()
    
    # Format as JavaScript array
    print("// Proper nouns loaded from Excel file (proper_nouns.xlsx)")
    print("// Update the Excel file to modify this list")
    print("const properNounPhrases = [")
    
    # Print in chunks for readability
    for i, noun in enumerate(nouns):
        comma = "," if i < len(nouns) - 1 else ""
        print(f"  '{noun}'{comma}")
    
    print("];")
    print(f"\n// Total: {len(nouns)} proper nouns")

