"""
Export proper nouns to JavaScript array format for frontend
Usage: python3 export_proper_nouns_for_frontend.py
Output: Copy the array and paste into frontend/src/App.js (replace existing properNounPhrases array)
"""

from proper_nouns_config import get_proper_noun_phrases
import json

if __name__ == "__main__":
    nouns = get_proper_noun_phrases()
    
    # Format as JavaScript array with proper indentation for frontend
    print("                      // Proper nouns loaded from Excel file (proper_nouns.xlsx)")
    print("                      // Update the Excel file to modify this list")
    print(f"                      // Total: {len(nouns)} proper nouns across 18 categories")
    
    # Print with proper indentation (matching frontend code style)
    for i, noun in enumerate(nouns):
        comma = "," if i < len(nouns) - 1 else ""
        # Escape single quotes in the noun
        escaped_noun = noun.replace("'", "\\'")
        print(f"                      '{escaped_noun}'{comma}")
    
    print("                    ];")
    print(f"\n// ✅ Generated {len(nouns)} proper nouns")
    print("// 📋 Copy the array above (from '// Proper nouns' to '];')")
    print("// 📝 Paste into frontend/src/App.js, replacing the existing properNounPhrases array")

