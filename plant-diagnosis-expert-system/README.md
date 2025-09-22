# Plant Diagnosis Expert System

**Mini-project** for Knowledge Engineering & Intelligent Systems — rule-based expert system + ML.

## What's included
- `backend/` - FastAPI backend that serves a rule-based reasoner and ML prediction endpoint
- `data/diseases.csv` - small example dataset (symptoms -> disease)
- `knowledge/rules.json` - simple rule base for reasoning
- `train.py` - script to train a DecisionTree model and save `model.pkl`
- `static/index.html` - frontend (single-page) to query the system
- `requirements.txt` - Python deps

## Quick start (local)
1. Create a python venv:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Train the ML model (optional):
```bash
python train.py
```
This will create `backend/model.pkl`.

4. Start the backend:
```bash
uvicorn backend.app:app --reload
```
5. Open `static/index.html` in your browser (or serve static with an http server) and point it to the backend (default `http://localhost:8000`).

## Deployment
- Push repository to GitHub.
- You can deploy the backend to platforms supporting Python/FastAPI (Railway, Render, Heroku, etc.) and host `static/` on GitHub Pages or alongside backend.

## Project structure
```
plant-diagnosis-expert-system/
├─ backend/
│  ├─ app.py
│  └─ model.pkl  (created by train.py)
├─ knowledge/
│  └─ rules.json
├─ data/
│  └─ diseases.csv
├─ static/
│  └─ index.html
├─ train.py
├─ requirements.txt
└─ README.md
```

---
Feel free to ask me to modify the rules, expand the dataset, or convert the frontend into a React app for GitHub Pages deployment.
