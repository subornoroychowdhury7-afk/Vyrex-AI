"use client";

import { useState } from "react";
import { X, Sliders, CheckCircle2, AlertCircle, RefreshCw } from "lucide-react";

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
      const res = await fetch(`${apiUrl.replace(/\/$/, "")}/health`);
      const end = performance.now();
      if (res.ok) { setTestStatus("success"); setLatency(Math.round(end - start)); }
      else setTestStatus("error");
    } catch { setTestStatus("error"); }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
      <div
        className="relative w-full max-w-md overflow-hidden rounded-2xl shadow-2xl transition-all
          bg-white dark:bg-[#161616]
          border border-gray-200 dark:border-[#2a2a2a]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-gray-100 dark:border-[#2a2a2a]">
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-red-600 dark:text-red-400" />
            <h3 className="font-bold text-lg text-gray-900 dark:text-white">
              Inference & Server Settings
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-6">

          {/* API URL */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
              Backend Endpoint
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                placeholder="http://localhost:8000"
                className="flex-1 px-3.5 py-2.5 rounded-xl text-sm
                  bg-gray-50 dark:bg-[#1f1f1f]
                  border border-gray-200 dark:border-[#2a2a2a]
                  text-gray-900 dark:text-gray-100
                  focus:outline-none focus:ring-2 focus:ring-red-500/40 dark:focus:ring-red-600/40
                  placeholder-gray-400 dark:placeholder-gray-600"
              />
              <button
                onClick={testConnection}
                disabled={testStatus === "testing"}
                className="px-3.5 py-2 rounded-xl text-xs font-semibold
                  bg-gray-100 dark:bg-[#1f1f1f]
                  hover:bg-red-50 dark:hover:bg-red-950/30
                  text-gray-700 dark:text-gray-300
                  hover:text-red-600 dark:hover:text-red-400
                  border border-gray-200 dark:border-[#2a2a2a]
                  transition-colors flex items-center gap-1.5"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${testStatus === "testing" ? "animate-spin" : ""}`} />
                Test
              </button>
            </div>
            {testStatus === "success" && (
              <p className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400 mt-2 font-medium">
                <CheckCircle2 className="w-4 h-4" /> Connected ({latency} ms)
              </p>
            )}
            {testStatus === "error" && (
              <p className="flex items-center gap-1.5 text-xs text-red-500 mt-2 font-medium">
                <AlertCircle className="w-4 h-4" /> Connection failed. Make sure backend is running.
              </p>
            )}
            <div className="flex gap-2 mt-2.5">
              <button
                type="button"
                onClick={() => setApiUrl("http://localhost:8000")}
                className="text-[11px] font-medium px-2.5 py-1 rounded-lg
                  bg-gray-100 dark:bg-[#1f1f1f]
                  hover:bg-red-50 dark:hover:bg-red-950/30
                  text-gray-600 dark:text-gray-300
                  hover:text-red-600 dark:hover:text-red-400
                  border border-gray-200 dark:border-[#2a2a2a]
                  transition-colors"
              >
                Localhost (8000)
              </button>
              <button
                type="button"
                onClick={() => setApiUrl(process.env.NEXT_PUBLIC_API_URL || "https://vyrex-ai-backend.onrender.com")}
                className="text-[11px] font-medium px-2.5 py-1 rounded-lg
                  bg-gray-100 dark:bg-[#1f1f1f]
                  hover:bg-red-50 dark:hover:bg-red-950/30
                  text-gray-600 dark:text-gray-300
                  hover:text-red-600 dark:hover:text-red-400
                  border border-gray-200 dark:border-[#2a2a2a]
                  transition-colors"
              >
                Render Cloud
              </button>
            </div>
          </div>

          {/* Temperature */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                Temperature
              </label>
              <span className="text-xs font-mono font-bold text-red-600 dark:text-red-400">
                {temperature.toFixed(2)}
              </span>
            </div>
            <input
              type="range" min="0.0" max="1.5" step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full cursor-pointer"
              style={{ accentColor: "#dc2626" }}
            />
            <p className="text-[11px] text-gray-500 dark:text-gray-500 mt-1">
              Lower = more focused; higher = more creative.
            </p>
          </div>

          {/* Max Tokens */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                Max Output Tokens
              </label>
              <span className="text-xs font-mono font-bold text-red-600 dark:text-red-400">
                {maxTokens}
              </span>
            </div>
            <input
              type="range" min="64" max="2048" step="64"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              className="w-full cursor-pointer"
              style={{ accentColor: "#dc2626" }}
            />
            <p className="text-[11px] text-gray-500 dark:text-gray-500 mt-1">
              Controls maximum length of generated response.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 px-6 border-t border-gray-100 dark:border-[#2a2a2a] bg-gray-50 dark:bg-[#111111] flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl text-sm font-semibold text-white transition-all
              bg-red-600 hover:bg-red-700 shadow-sm hover:shadow-red-500/30 hover:shadow-md"
          >
            Save & Close
          </button>
        </div>
      </div>
    </div>
  );
}
