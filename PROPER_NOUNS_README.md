# Proper Nouns Configuration

This directory contains the proper nouns configuration for transcript segmentation.

## Files

- **`proper_nouns.xlsx`** - Excel file containing all proper noun phrases (530+ entries across 18 categories)
- **`proper_nouns_config.py`** - Python module that reads from the Excel file
- **`generate_proper_nouns_excel.py`** - Script to generate/regenerate the Excel file
- **`export_proper_nouns_for_frontend.py`** - Script to export proper nouns for frontend use

## How to Update Proper Nouns

### Option 1: Edit Excel File Directly (Recommended)

1. Open `proper_nouns.xlsx` in Microsoft Excel, Google Sheets, or any spreadsheet application
2. Navigate to the "Proper Nouns" sheet
3. Add new entries in column B (Proper Noun Phrase)
4. Optionally add category in column A and notes in column C
5. Save the file
6. The backend will automatically load the updated list on next restart

### Option 2: Regenerate Excel File

If you want to regenerate the Excel file with new categories or entries:

```bash
cd backend
python3 generate_proper_nouns_excel.py
```

## Categories

The Excel file is organized into 18 categories:

1. **Countries & Regions** - Countries, regions, territories
2. **Cities & Places** - Cities, landmarks, geographical features
3. **Historical Figures** - Famous people from history
4. **Companies & Brands** - Major corporations and brand names
5. **Educational Institutions** - Universities, colleges, schools
6. **Organizations & Institutions** - NGOs, government agencies, international organizations
7. **Historical Events** - Wars, revolutions, major historical events
8. **Academic Terms & Concepts** - Degrees, tests, educational terminology
9. **Scientific Concepts & Theories** - Scientific theories, laws, principles
10. **Government & Politics** - Political positions, government bodies
11. **Religious & Philosophical** - Religions, religious texts, philosophical concepts
12. **Technology & Innovation** - Tech terms, platforms, innovations
13. **Arts & Literature** - Art movements, famous works, awards
14. **Sports & Entertainment** - Sports leagues, events, entertainment
15. **Medical & Health** - Medical terms, diseases, health organizations
16. **Business & Economics** - Economic terms, financial institutions
17. **Time Periods & Eras** - Historical periods, ages, eras
18. **Educational Concepts** - Learning theories, pedagogical concepts

## Adding New Entries

When adding new proper nouns:

1. **Only include proper nouns that start with capital letters** (e.g., "United States", "Abraham Lincoln", "Toyota")
2. **Include multi-word phrases** that should be treated as single units
3. **Use lowercase in the Excel file** - the system converts to lowercase for matching
4. **Place in appropriate category** for easier management

## Examples

✅ **Good entries:**
- United States
- World War II
- Abraham Lincoln
- Toyota
- Harvard University
- Prime Minister

❌ **Bad entries:**
- the united states (don't include articles)
- world war (too generic, use "World War II")
- car (not a proper noun)

## Technical Details

- The backend reads from `proper_nouns.xlsx` using `openpyxl`
- Proper nouns are cached in memory for performance
- Frontend uses a static list generated from the Excel file
- To update frontend: Run `export_proper_nouns_for_frontend.py` and copy output to `frontend/src/App.js`

## Notes

- The Excel file is the source of truth
- Changes to the Excel file require backend restart to take effect
- Frontend list should be updated when Excel file is significantly modified

