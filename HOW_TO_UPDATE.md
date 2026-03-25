# 📝 How to Update Proper Nouns - Simple Guide

## The Process in 4 Steps

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Edit Excel File                                     │
│ ─────────────────────────────────────────────────────────── │
│ 1. Open: backend/proper_nouns.xlsx                          │
│ 2. Add new proper nouns in Column B (use lowercase)         │
│ 3. Save                                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Generate Frontend Array                             │
│ ─────────────────────────────────────────────────────────── │
│ Run: python3 export_proper_nouns_for_frontend.py            │
│ Copy the output array                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Update Frontend Code                                │
│ ─────────────────────────────────────────────────────────── │
│ 1. Open: frontend/src/App.js                                │
│ 2. Find: properNounPhrases array (line ~2388)               │
│ 3. Replace: Paste new array from Step 2                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Push to GitHub                                       │
│ ─────────────────────────────────────────────────────────── │
│ Backend:  git add proper_nouns.xlsx && git commit && push    │
│ Frontend: git add src/App.js && git commit && push          │
│                                                              │
│ ✅ Auto-deploys to Railway & Vercel                         │
└─────────────────────────────────────────────────────────────┘
```

## Quick Commands

```bash
# 1. Edit Excel file (in Excel/Sheets)
# 2. Generate frontend array
cd backend
python3 export_proper_nouns_for_frontend.py

# 3. Copy output and update frontend/src/App.js manually
# 4. Commit and push
cd backend && git add proper_nouns.xlsx && git commit -m "Update proper nouns" && git push
cd ../frontend && git add src/App.js && git commit -m "Update proper nouns list" && git push
```

## Important Notes

- ✅ **Backend**: Updates automatically when you push Excel file (no code change needed)
- ⚠️ **Frontend**: Must manually update `App.js` with new array
- 📋 **Excel Format**: Use lowercase in Excel (e.g., "toyota", not "Toyota")
- 🔄 **Cache**: Backend caches the list - cleared on server restart

## Example: Adding "Tesla Motors"

1. **Excel**: Add "tesla motors" in Column B
2. **Generate**: Run export script
3. **Frontend**: Replace array in `App.js` 
4. **Push**: Commit both files and push

**Result**: "Tesla Motors" won't break into separate paragraphs in transcripts!

---

For detailed instructions, see `UPDATE_PROPER_NOUNS_GUIDE.md`



