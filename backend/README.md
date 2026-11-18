# VYREX AI - Render Backend Deployment Guide

This backend serves the **VYREX AI** OpenAI-compatible REST API. It is optimized to run smoothly on [Render](https://render.com) (both Free and Paid tiers) with instant startup, healthchecks, and CORS pre-configured for Vercel.

---

## 🚀 How to Deploy on Render (Step-by-Step)

### Option 1: One-Click Deploy (Using Blueprint)
1. Push this repository to GitHub.
2. Go to your [Render Dashboard](https://dashboard.render.com).
3. Click **New +** -> **Blueprint**.
4. Connect your `Vyrex-AI` repository.
5. Render will automatically detect `backend/render.yaml` and configure the Web Service with healthchecks and start commands!
6. Click **Apply**.

---

### Option 2: Manual Web Service Deployment
1. Go to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository (`Vyrex-AI`).
4. Configure the settings:
   - **Name:** `vyrex-ai-backend`
   - **Root Directory:** `backend`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free` (or Starter)
5. Under **Advanced Settings**:
   - **Health Check Path:** `/health`
6. Click **Create Web Service**.

Once deployed, Render gives you a public URL like:
👉 `https://vyrex-ai-backend.onrender.com`

---

## 🧪 Local Testing

To test the backend locally on your computer:

```bash
cd backend
python -m pip install -r requirements.txt
python main.py
```

* **Interactive Swagger UI:** `http://localhost:8000/docs`
* **Health Check:** `http://localhost:8000/health`
* **Chat API:** `http://localhost:8000/v1/chat/completions`

---

## 🔗 Connecting to Vercel Frontend
In your Vercel project settings, set:
```env
NEXT_PUBLIC_API_URL=https://your-vyrex-backend.onrender.com
```
