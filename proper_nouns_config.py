"""
Configuration file for proper noun phrases that should not trigger sentence breaks in transcripts.
Reads from proper_nouns.xlsx Excel file for easy editing.
Users can update the Excel file without needing to modify code.
"""

from pathlib import Path
import openpyxl
import logging

logger = logging.getLogger(__name__)

# Cache for loaded proper nouns
_proper_nouns_cache = None

def get_proper_noun_phrases():
    """
    Returns the list of proper noun phrases from Excel file.
    Falls back to default list if Excel file is not found.
    """
    global _proper_nouns_cache
    
    # Return cached version if available
    if _proper_nouns_cache is not None:
        return _proper_nouns_cache
    
    # Try to load from Excel file
    excel_path = Path(__file__).parent / "proper_nouns.xlsx"
    
    if excel_path.exists():
        try:
            wb = openpyxl.load_workbook(excel_path, data_only=True)
            ws = wb.active
            
            proper_nouns = []
            
            # Read from column B (Proper Noun Phrase column)
            # Skip header row (row 1) and empty rows
            for row in ws.iter_rows(min_row=2, values_only=True):
                category = row[0]  # Column A
                noun_phrase = row[1]  # Column B
                
                if noun_phrase and isinstance(noun_phrase, str):
                    # Convert to lowercase for matching
                    proper_nouns.append(noun_phrase.lower().strip())
            
            wb.close()
            
            # Remove duplicates and sort
            proper_nouns = sorted(list(set(proper_nouns)))
            
            _proper_nouns_cache = proper_nouns
            logger.info(f"✅ Loaded {len(proper_nouns)} proper nouns from {excel_path.name}")
            
            return proper_nouns
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load proper nouns from Excel: {e}. Using default list.")
            return _get_default_proper_nouns()
    else:
        logger.warning(f"⚠️ Excel file not found at {excel_path}. Using default list.")
        logger.info(f"💡 To create the Excel file, run: python3 generate_proper_nouns_excel.py")
        return _get_default_proper_nouns()

def _get_default_proper_nouns():
    """
    Default fallback list if Excel file is not available.
    """
    return [
        'world war', 'world war ii', 'world war i', 'world war 2', 'world war 1',
        'cold war', 'civil war', 'vietnam war', 'korean war',
        'united states', 'united kingdom', 'united nations', 'united arab emirates',
        'south korea', 'north korea', 'south africa', 'north america', 'south america',
        'middle east', 'far east', 'east asia', 'southeast asia', 'west africa',
        'new zealand', 'new york', 'new jersey', 'new hampshire', 'new mexico',
        'los angeles', 'san francisco', 'san diego', 'san antonio',
        'prime minister', 'supreme court', 'house of', 'senate of',
        'world health', 'world bank', 'european union', 'nato',
        'oxford university', 'cambridge university', 'harvard university',
        'phd', 'mba', 'bachelor of', 'master of', 'doctor of',
        'chief executive', 'chief financial', 'chief technology',
        'great britain', 'soviet union', 'russian federation',
        'hong kong', 'sri lanka', 'costa rica'
    ]

def reload_proper_nouns():
    """
    Clear cache to force reload from Excel file on next call.
    Useful when Excel file has been updated.
    """
    global _proper_nouns_cache
    _proper_nouns_cache = None
    logger.info("🔄 Proper nouns cache cleared. Will reload from Excel on next access.")

