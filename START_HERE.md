# Chatbot chalane ke exact steps

## Step 1: Terminal kholo
Cursor mein **Terminal** kholo: **Ctrl + `** (backtick) ya menu se **View → Terminal**.

---

## Step 2: Backend folder mein jao (zaroori)
Terminal mein ye type karo aur **Enter** dabao (sirf ye ek line):

```
cd "c:\Users\User\Desktop\chatbot only\Backend"
```

**Check:** Prompt ab `...\Backend>` dikhna chahiye. Step 4 ya 5 se pehle hamesha yahi pe hona zaroori hai.

---

## Step 3: Dependencies install karo (sirf pehli baar)
Phir ye type karo aur **Enter**:

```
pip install -r requirements.txt
```

Thoda time lagega (1–2 min). Jab "Successfully installed" dikhe, next step. **Agar install wala terminal khud band ho gaya ho** to koi baat nahi — Step 2 chala ke `Backend` mein aao, phir Step 3 dobara chala do; agar pehle install ho chuka hoga to turant "Requirement already satisfied" dikhega.

---

## Step 4: FAQ index karo (sirf pehli baar)
**Option A (easy):** `Backend` folder kholo → **`run_index_once.bat`** pe double-click karo. Jab "Done" dikhe, next step.

**Option B (terminal):** Zaroor pehle Step 2 chala ke `Backend` mein ho jao, phir ye ek line copy-paste karo:

```
cd "c:\Users\User\Desktop\chatbot only\Backend"; python index_faq.py
```

Jab "Done" dikhe, next step. Agar pehle chala chuke ho to skip karo.

---

## Step 5: Backend start karo
**Option A (easy):** `Backend` folder kholo → **`run_backend.bat`** pe double-click karo. Terminal khulega, server start ho jayega.

**Option B (terminal):** Pehle Step 2 se `Backend` mein ho, phir ye ek line:

```
cd "c:\Users\User\Desktop\chatbot only\Backend"; uvicorn main:app --reload --port 8000
```

Terminal mein **"Uvicorn running on http://127.0.0.1:8000"** dikhna chahiye. Is terminal ko **band mat karo**.

---

## Step 6: Browser mein kholo
Browser (Chrome/Edge) open karo aur address bar mein ye likho:

```
http://localhost:8000
```

**Enter** dabao. Chat wala page khul jayega — yahi se sawal pooch sakte ho.

---

## Short summary (order)
1. `cd "c:\Users\User\Desktop\chatbot only\Backend"`
2. `pip install -r requirements.txt`  (pehli baar)
3. `python index_faq.py`  (pehli baar)
4. `uvicorn main:app --reload --port 8000`
5. Browser: **http://localhost:8000**

---

## Agar error aaye
- **"can't open file 'index_faq.py'" / "No such file or directory"** → Terminal **project root** pe hai, Backend mein nahi. Pehle ye chalao: `cd "c:\Users\User\Desktop\chatbot only\Backend"` ya **run_index_once.bat** double-click karo.
- **"requirements.txt not found"** → Step 2 sahi se karo, `Backend` folder mein hona chahiye.
- **"Failed to fetch"** → Backend (Step 5) chal raha hona chahiye; **http://localhost:8000** use karo, file:// mat use karo.
- **PowerShell arrow error** → `→` mat type karo. Har command alag line pe type karo aur Enter dabao.
