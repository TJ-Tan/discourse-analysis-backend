# How to Update Proper Nouns Excel File

## 📋 Quick Summary

1. **Edit** `backend/proper_nouns.xlsx` → Add new proper nouns in Column B
2. **Generate** frontend array → Run `python3 export_proper_nouns_for_frontend.py`
3. **Update** `frontend/src/App.js` → Replace the `properNounPhrases` array
4. **Push** to GitHub → Both repositories auto-deploy

---

## Step-by-Step Guide

### Step 1: Edit the Excel File

1. **Open** `backend/proper_nouns.xlsx` in:
   - Microsoft Excel
   - Google Sheets (upload to Google Drive)
   - LibreOffice Calc
   - Any spreadsheet application

2. **Navigate** to the "Proper Nouns" sheet

3. **Add new entries** in **Column B** (Proper Noun Phrase):
   - ✅ Only proper nouns that start with capital letters
   - ✅ Use **lowercase** in Excel (e.g., "toyota", "abraham lincoln", "united states")
   - ✅ Multi-word phrases are fine (e.g., "world war ii", "prime minister")
   - ✅ Place in appropriate category section (Column A is optional but helpful)

4. **Save the file**

**Example:**
```
Column A (Category)        | Column B (Proper Noun Phrase) | Column C (Notes)
---------------------------|------------------------------|------------------
Companies & Brands         | tesla motors                 |
Historical Figures         | steve jobs                   |
Educational Institutions   | stanford university          |
```

### Step 2: Generate Frontend Array

The frontend needs a JavaScript array. Generate it from the Excel file:

```bash
cd backend
python3 export_proper_nouns_for_frontend.py
```

**Output will look like:**
```javascript
                      // Proper nouns loaded from Excel file (proper_nouns.xlsx)
                      // Update the Excel file to modify this list
                      // Total: 530 proper nouns across 18 categories
                      '10 downing street',
                      '4g network',
                      '5g network',
                      ...
                      'zone of proximal development'
                    ];
```

**Copy the entire output** (from the comment line to the closing `];`)

### Step 3: Update Frontend Code

1. **Open** `frontend/src/App.js`

2. **Find** the `properNounPhrases` array (around line 2388):
   ```javascript
   const properNounPhrases = [
     // ... existing entries ...
   ];
   ```

3. **Replace** the entire array:
   - Select from `// Proper nouns loaded from Excel` comment
   - To the closing `];`
   - Paste the new array from Step 2

4. **Verify** the syntax is correct (no missing commas, quotes, brackets)

### Step 4: Commit and Push

**Backend (Excel file):**
```bash
cd backend
git add proper_nouns.xlsx
git commit -m "Update proper nouns: added [list new entries]"
git push origin master
```

**Frontend (JavaScript array):**
```bash
cd frontend
git add src/App.js
git commit -m "Update proper nouns list from Excel"
git push origin master
```

**Railway and Vercel will automatically deploy!** 🚀

---

## Backend Update (Automatic)

✅ **Backend updates automatically!**

The backend reads from the Excel file dynamically:
- Changes are loaded **on server restart**
- Railway auto-restarts when you push to GitHub
- **No code changes needed** for backend

**Note:** The backend caches the list for performance. The cache is cleared on server restart.

## Detailed Steps

### Editing the Excel File

**File Location:** `backend/proper_nouns.xlsx`

**Structure:**
- **Column A**: Category (optional, for organization)
- **Column B**: Proper Noun Phrase (required)
- **Column C**: Notes (optional, for future reference)

**Example:**
```
Category              | Proper Noun Phrase        | Notes
---------------------|---------------------------|------------------
Countries & Regions  | united states             |
Companies & Brands   | toyota                    |
Historical Figures   | abraham lincoln           |
```

**Important Rules:**
- ✅ Only proper nouns that start with capital letters
- ✅ Multi-word phrases (e.g., "United States", "World War II")
- ✅ Use lowercase in Excel (system converts automatically)
- ❌ Don't include articles ("the", "a", "an")
- ❌ Don't include generic terms

### Backend Update Process

**Automatic Method (Recommended):**
1. Edit `proper_nouns.xlsx`
2. Commit and push to GitHub:
   ```bash
   cd backend
   git add proper_nouns.xlsx
   git commit -m "Update proper nouns list"
   git push origin master
   ```
3. Railway will automatically restart and load the new list

**Manual Reload (If needed):**
The backend caches the list. To force reload without restart:
- Call `reload_proper_nouns()` function in `proper_nouns_config.py`
- Or restart the server

### Frontend Update Process

**Step-by-step:**

1. **Generate JavaScript array:**
   ```bash
   cd backend
   python3 export_proper_nouns_for_frontend.py > /tmp/new_proper_nouns.js
   ```

2. **View the output:**
   ```bash
   cat /tmp/new_proper_nouns.js
   ```

3. **Update frontend/src/App.js:**
   - Open `frontend/src/App.js`
   - Find line ~2388 where `const properNounPhrases = [`
   - Replace everything from `const properNounPhrases = [` to the closing `];` with the new array
   - Keep the comment lines above it

4. **Test locally (optional):**
   ```bash
   cd frontend
   npm start
   ```

5. **Commit and push:**
   ```bash
   git add src/App.js
   git commit -m "Update proper nouns list from Excel"
   git push origin master
   ```

## Automated Script (Future Enhancement)

You could create a script to automate the frontend update:

```bash
#!/bin/bash
# update_proper_nouns.sh

cd backend
python3 export_proper_nouns_for_frontend.py > /tmp/proper_nouns.js

# Extract just the array part
sed -n '/const properNounPhrases = \[/,/\];/p' /tmp/proper_nouns.js > /tmp/array_only.js

# Update frontend file (requires manual verification)
cd ../frontend
# ... replace in App.js ...
```

## Verification

After updating:

1. **Backend:** Check logs for "✅ Loaded X proper nouns from proper_nouns.xlsx"
2. **Frontend:** Check browser console - the array should have the new count
3. **Test:** Upload a video with a new proper noun and verify it doesn't break into separate paragraphs

## Troubleshooting

**Issue: Backend not loading new nouns**
- Solution: Restart the Railway server or wait for auto-deploy

**Issue: Frontend still using old list**
- Solution: Make sure you updated `src/App.js` and pushed to GitHub
- Clear browser cache if needed

**Issue: Excel file not found**
- Solution: Make sure `proper_nouns.xlsx` is in the `backend/` directory
- Run `python3 generate_proper_nouns_excel.py` to create it

**Issue: Syntax error in frontend**
- Solution: Make sure the JavaScript array is properly formatted
- Check for missing commas, quotes, or brackets

## Best Practices

1. **Test locally first** before pushing to production
2. **Keep categories organized** in the Excel file for easier management
3. **Add entries in lowercase** - the system handles case conversion
4. **Document additions** in the Notes column if needed
5. **Commit Excel changes** along with frontend updates for consistency
6. **Review the list periodically** to remove outdated entries

## Example Workflow

```bash
# 1. Edit Excel file (in Excel/Sheets)
# 2. Generate frontend array
cd backend
python3 export_proper_nouns_for_frontend.py

# 3. Copy output and update frontend/src/App.js manually
# 4. Commit both files
cd ../backend
git add proper_nouns.xlsx
git commit -m "Add new proper nouns: [list them]"
git push origin master

cd ../frontend
git add src/App.js
git commit -m "Update proper nouns list"
git push origin master
```

