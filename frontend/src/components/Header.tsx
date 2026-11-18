"use client";

import { Sparkles, Sliders, Info, Github } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";

interface HeaderProps {
  onOpenSettings: () => void;
  onOpenCreator: () => void;
  status: "connected" | "connecting" | "offline";
  apiUrl: string;
}

export function Header({
  onOpenSettings,
  onOpenCreator,
  status,
  apiUrl,
}: HeaderProps) {
  return (
    <header className="sticky top-0 z-40 w-full backdrop-blur-md bg-white/70 dark:bg-[#070b14]/70 border-b border-slate-200/80 dark:border-slate-800/80 transition-colors">
      <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
        {/* Logo and Brand */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-700 to-purple-600 flex items-center justify-center text-white shadow-md shadow-indigo-500/25">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-extrabold text-base tracking-tight text-slate-900 dark:text-slate-100">
                VYREX AI
              </h1>
              <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 border border-indigo-100 dark:border-indigo-900/40">
                Qwen3-1.7B • QLoRA V3
              </span>
            </div>
            <p className="hidden sm:block text-[11px] text-slate-500 dark:text-slate-400">
              Architected by Suborno Roy Chowdhury
            </p>
          </div>
        </div>

        {/* Right Action Icons */}
        <div className="flex items-center gap-2">
          {/* Status Badge */}
          <div
            title={`Backend: ${apiUrl}`}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700/60 text-slate-600 dark:text-slate-300"
          >
            <span
              className={`w-2 h-2 rounded-full ${
                status === "connected"
                  ? "bg-emerald-500 shadow-sm shadow-emerald-500 animate-pulse"
                  : status === "connecting"
                  ? "bg-amber-400 animate-pulse"
                  : "bg-rose-500"
              }`}
            />
            <span className="text-[11px] capitalize hidden md:inline">
              {status}
            </span>
          </div>

          {/* Creator Info Button */}
          <button
            onClick={onOpenCreator}
            aria-label="About Creator"
            title="About Creator & Architecture"
            className="flex items-center justify-center w-10 h-10 rounded-xl transition-all duration-200
              bg-white/80 dark:bg-slate-800/80 hover:bg-slate-100 dark:hover:bg-slate-700
              border border-slate-200 dark:border-slate-700/60 text-slate-600 dark:text-slate-300"
          >
            <Info className="w-4 h-4" />
          </button>

          {/* Settings Button */}
          <button
            onClick={onOpenSettings}
            aria-label="Server Settings"
            title="Configure Backend URL & Parameters"
            className="flex items-center justify-center w-10 h-10 rounded-xl transition-all duration-200
              bg-white/80 dark:bg-slate-800/80 hover:bg-slate-100 dark:hover:bg-slate-700
              border border-slate-200 dark:border-slate-700/60 text-slate-600 dark:text-slate-300"
          >
            <Sliders className="w-4 h-4" />
          </button>

          {/* Dark / Bright Mode Toggle */}
          <ThemeToggle />

          {/* GitHub Repo Link */}
          <a
            href="https://github.com/subornoroychowdhury7-afk/Vyrex-AI"
            target="_blank"
            rel="noopener noreferrer"
            title="View Source on GitHub"
            className="flex items-center justify-center w-10 h-10 rounded-xl transition-all duration-200
              bg-white/80 dark:bg-slate-800/80 hover:bg-slate-100 dark:hover:bg-slate-700
              border border-slate-200 dark:border-slate-700/60 text-slate-600 dark:text-slate-300"
          >
            <Github className="w-4 h-4" />
          </a>
        </div>
      </div>
    </header>
  );
}
