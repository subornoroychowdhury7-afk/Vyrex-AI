"use client";

import Image from "next/image";
import { Sliders, Info, Github } from "lucide-react";
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
    <header className="sticky top-0 z-40 w-full backdrop-blur-md
      bg-white/90 dark:bg-[#0d0d0d]/90
      border-b border-gray-200 dark:border-[#2a2a2a]
      shadow-sm dark:shadow-none
      transition-colors duration-300">
      <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">

        {/* ── Logo + Brand ── */}
        <div className="flex items-center gap-3">
          <div className="relative w-10 h-10 flex-shrink-0">
            <Image
              src="/vyrex-symbol.png"
              alt="Vyrex AI"
              fill
              className="object-contain"
              priority
            />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-extrabold text-base tracking-tight text-gray-900 dark:text-white">
                Vyrex <span className="text-red-600 dark:text-red-500">AI</span>
              </h1>
              <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold
                bg-red-50 dark:bg-red-950/40
                text-red-700 dark:text-red-400
                border border-red-200 dark:border-red-900/50">
                Vyrex Hyperion 3
              </span>
            </div>
            <p className="hidden sm:block text-[11px] text-gray-500 dark:text-gray-400 mt-0.5">
              Independent AI Assistant
            </p>
          </div>
        </div>

        {/* ── Right Controls ── */}
        <div className="flex items-center gap-2">

          {/* Status dot */}
          <div
            title={`Backend: ${apiUrl}`}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium
              bg-gray-100 dark:bg-[#1f1f1f]
              border border-gray-200 dark:border-[#2a2a2a]
              text-gray-600 dark:text-gray-300"
          >
            <span className={`w-2 h-2 rounded-full ${
              status === "connected"
                ? "bg-green-500 shadow-sm shadow-green-500 animate-pulse"
                : status === "connecting"
                ? "bg-amber-400 animate-pulse"
                : "bg-red-500"
            }`} />
            <span className="text-[11px] capitalize hidden md:inline">{status}</span>
          </div>

          {/* Creator Info */}
          <button
            onClick={onOpenCreator}
            aria-label="About Creator"
            title="About Creator & Architecture"
            className="flex items-center justify-center w-10 h-10 rounded-xl transition-all duration-200
              bg-white dark:bg-[#1f1f1f]
              hover:bg-red-50 dark:hover:bg-red-950/30
              border border-gray-200 dark:border-[#2a2a2a]
              text-gray-500 dark:text-gray-400
              hover:text-red-600 dark:hover:text-red-400"
          >
            <Info className="w-4 h-4" />
          </button>

          {/* Settings */}
          <button
            onClick={onOpenSettings}
            aria-label="Settings"
            title="Configure Backend & Parameters"
            className="flex items-center justify-center w-10 h-10 rounded-xl transition-all duration-200
              bg-white dark:bg-[#1f1f1f]
              hover:bg-red-50 dark:hover:bg-red-950/30
              border border-gray-200 dark:border-[#2a2a2a]
              text-gray-500 dark:text-gray-400
              hover:text-red-600 dark:hover:text-red-400"
          >
            <Sliders className="w-4 h-4" />
          </button>

          {/* Theme Toggle */}
          <ThemeToggle />

          {/* GitHub */}
          <a
            href="https://github.com/subornoroychowdhury7-afk/Vyrex-AI"
            target="_blank"
            rel="noopener noreferrer"
            title="View on GitHub"
            className="flex items-center justify-center w-10 h-10 rounded-xl transition-all duration-200
              bg-white dark:bg-[#1f1f1f]
              hover:bg-red-50 dark:hover:bg-red-950/30
              border border-gray-200 dark:border-[#2a2a2a]
              text-gray-500 dark:text-gray-400
              hover:text-red-600 dark:hover:text-red-400"
          >
            <Github className="w-4 h-4" />
          </a>
        </div>
      </div>
    </header>
  );
}
