# Quick Guide: Updating Proper Nouns

## Simple 3-Step Process

### Step 1: Edit Excel File
1. Open `backend/proper_nouns.xlsx` in Excel/Google Sheets
2. Add new proper nouns in **Column B** (use lowercase, e.g., "toyota", "abraham lincoln")
3. Save the file

### Step 2: Update Frontend Code
```bash
cd backend
python3 export_proper_nouns_for_frontend.py
```
Copy the output array and replace the `properNounPhrases` array in `frontend/src/App.js` (around line 2388)

### Step 3: Push to GitHub
```bash
# Backend
cd backend
git add proper_nouns.xlsx
git commit -m "Update proper nouns: [describe changes]"
git push origin master

# Frontend  
cd frontend
git add src/App.js
git commit -m "Update proper nouns list"
git push origin master
```

**That's it!** Railway and Vercel will auto-deploy.

---

## Detailed Instructions

See `UPDATE_PROPER_NOUNS_GUIDE.md` for complete documentation.

