"use client";

import { useState } from "react";
import { X, Server, Sliders, CheckCircle2, AlertCircle, RefreshCw } from "lucide-react";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  apiUrl: string;
  setApiUrl: (url: string) => void;
  temperature: number;
  setTemperature: (temp: number) => void;
  maxTokens: number;
  setMaxTokens: (tokens: number) => void;
}

export function SettingsModal({
  isOpen,
  onClose,
  apiUrl,
  setApiUrl,
  temperature,
  setTemperature,
  maxTokens,
  setMaxTokens,
}: SettingsModalProps) {
  const [testStatus, setTestStatus] = useState<"idle" | "testing" | "success" | "error">("idle");
  const [latency, setLatency] = useState<number | null>(null);

  if (!isOpen) return null;

  const testConnection = async () => {
    setTestStatus("testing");
    setLatency(null);
    const start = performance.now();
    try {
      const res = await fetch(`${apiUrl.replace(/\/$/, "")}/health`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });
      const end = performance.now();
      if (res.ok) {
        setTestStatus("success");
        setLatency(Math.round(end - start));
      } else {
        setTestStatus("error");
      }
    } catch (err) {
      setTestStatus("error");
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
      <div
        className="relative w-full max-w-md overflow-hidden rounded-2xl border border-slate-200 dark:border-slate-800 
          bg-white dark:bg-slate-900 shadow-2xl transition-all"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between p-5 border-b border-slate-100 dark:border-slate-800">
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            <h3 className="font-bold text-lg text-slate-900 dark:text-slate-100">
              Inference & Server Settings
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-5">
          {/* Backend API URL */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-2">
              Backend Endpoint (Render / Local)
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                placeholder="http://localhost:8000"
                className="flex-1 px-3.5 py-2.5 rounded-xl text-sm bg-slate-50 dark:bg-slate-800/80 
                  border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-slate-100
                  focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
              />
              <button
                onClick={testConnection}
                disabled={testStatus === "testing"}
                className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-slate-100 dark:bg-slate-800 
                  hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 
                  border border-slate-200 dark:border-slate-700 transition-colors flex items-center gap-1.5"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${testStatus === "testing" ? "animate-spin" : ""}`} />
                Test
              </button>
            </div>

            {/* Test result feedback */}
            {testStatus === "success" && (
              <p className="flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400 mt-2 font-medium">
                <CheckCircle2 className="w-4 h-4" /> Connected successfully ({latency} ms)
              </p>
            )}
            {testStatus === "error" && (
              <p className="flex items-center gap-1.5 text-xs text-rose-500 mt-2 font-medium">
                <AlertCircle className="w-4 h-4" /> Connection failed. Make sure backend is running.
              </p>
            )}

            {/* Quick preset buttons */}
            <div className="flex gap-2 mt-2.5">
              <button
                type="button"
                onClick={() => setApiUrl("http://localhost:8000")}
                className="text-[11px] font-medium px-2.5 py-1 rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-indigo-50 dark:hover:bg-indigo-950/40 text-slate-600 dark:text-slate-300 transition-colors"
              >
                Localhost (8000)
              </button>
              <button
                type="button"
                onClick={() => setApiUrl(process.env.NEXT_PUBLIC_API_URL || "https://vyrex-ai-backend.onrender.com")}
                className="text-[11px] font-medium px-2.5 py-1 rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-indigo-50 dark:hover:bg-indigo-950/40 text-slate-600 dark:text-slate-300 transition-colors"
              >
                Render Cloud Preset
              </button>
            </div>
          </div>

          {/* Temperature Slider */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Temperature
              </label>
              <span className="text-xs font-mono font-bold text-indigo-600 dark:text-indigo-400">
                {temperature.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.5"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-indigo-600 cursor-pointer"
            />
            <p className="text-[11px] text-slate-500 mt-1">
              Lower (0.0) is focused & deterministic; higher (0.8+) is creative.
            </p>
          </div>

          {/* Max Tokens Slider */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Max Output Tokens
              </label>
              <span className="text-xs font-mono font-bold text-indigo-600 dark:text-indigo-400">
                {maxTokens}
              </span>
            </div>
            <input
              type="range"
              min="64"
              max="2048"
              step="64"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              className="w-full accent-indigo-600 cursor-pointer"
            />
            <p className="text-[11px] text-slate-500 mt-1">
              Controls maximum length of generated response.
            </p>
          </div>
        </div>

        <div className="p-4 px-6 bg-slate-50 dark:bg-slate-950/50 border-t border-slate-100 dark:border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-indigo-600 text-white hover:bg-indigo-700 transition-colors shadow-sm"
          >
            Save & Close
          </button>
        </div>
      </div>
    </div>
  );
}
