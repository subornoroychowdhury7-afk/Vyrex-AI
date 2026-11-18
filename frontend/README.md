# VYREX AI - Vercel Frontend Deployment Guide

This is the modern web interface for **VYREX AI**, built with **Next.js 14**, **Tailwind CSS**, and **Lucide Icons**. It includes full support for **Dark Mode & Bright Mode**, token-by-token real-time streaming, and interactive configuration.

---

## 🚀 How to Deploy on Vercel (Step-by-Step)

### Option 1: Deploy with Vercel Web Dashboard (Easiest)
1. Push your repository to GitHub (`https://github.com/subornoroychowdhury7-afk/Vyrex-AI`).
2. Go to [vercel.com](https://vercel.com) and log in.
3. Click **Add New...** -> **Project**.
4. Import your `Vyrex-AI` GitHub repository.
5. In the **Configure Project** screen:
   - **Framework Preset:** `Next.js` (detected automatically)
   - **Root Directory:** Click **Edit** and select `frontend` (⚠️ **Crucial step!**)
6. Under **Environment Variables**, add:
   - **Key:** `NEXT_PUBLIC_API_URL`
   - **Value:** Your Render backend URL (e.g. `https://vyrex-ai-backend.onrender.com` or your local tunnel URL).
7. Click **Deploy**.

Vercel will build and launch your website with a live URL like:
👉 `https://vyrex-ai.vercel.app`

---

### Option 2: Deploy Using Vercel CLI
If you have the Vercel CLI installed:

```bash
cd frontend
npx vercel
```
* Follow the prompts:
  * Set root directory to `.`
  * When prompted for environment variables, add `NEXT_PUBLIC_API_URL`.

---

## 🧪 Local Development

To run the frontend locally on your machine:

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🎨 Features
* 🌓 **Dark Mode & Bright Mode:** Switch seamlessly between deep midnight dark theme and clean pearl bright mode.
* ⚡ **Real-time SSE Streaming:** Generates responses token-by-token just like ChatGPT.
* 💻 **Markdown & Code Syntax:** Clean code block rendering with a 1-click **Copy Code** button.
* ⚙️ **Interactive Settings Modal:** Change the backend API endpoint, temperature, and token limits on the fly without editing code.
* 👤 **Creator Attribution Modal:** Highlights Suborno Roy Chowdhury, St. Thomas' College of Engineering & Technology, and IIT Kharagpur.
