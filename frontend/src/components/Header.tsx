"use client";

import Image from "next/image";
import { History } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";

interface HeaderProps {
  onOpenHistory?: () => void;
  historyCount?: number;
}

export function Header({
  onOpenHistory,
  historyCount = 0,
}: HeaderProps) {
  return (
    <header className="sticky top-0 z-40 w-full backdrop-blur-md
      bg-white/90 dark:bg-[#0d0d0d]/90
      border-b border-gray-200 dark:border-[#2a2a2a]
      shadow-sm dark:shadow-none
      transition-colors duration-300">
      <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">

        {/* ── Left: History Button + Logo + Brand ── */}
        <div className="flex items-center gap-3">
          {/* Chat History Toggle Button */}
          {onOpenHistory && (
            <button
              onClick={onOpenHistory}
              aria-label="Open Chat History"
              title="Chat History"
              className="relative flex items-center justify-center w-10 h-10 rounded-xl transition-all duration-200
                bg-white dark:bg-[#1a1a1e]
                hover:bg-red-50 dark:hover:bg-red-950/30
                border border-gray-200 dark:border-[#2a2a2a]
                text-gray-600 dark:text-gray-300
                hover:text-red-600 dark:hover:text-red-400 shadow-sm"
            >
              <History className="w-4 h-4" />
              {historyCount > 0 && (
                <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-red-600 text-[10px] font-bold text-white flex items-center justify-center shadow">
                  {historyCount > 9 ? "9+" : historyCount}
                </span>
              )}
            </button>
          )}

          {/* Logo */}
          <div className="relative w-9 h-9 flex-shrink-0">
            <Image
              src="/vyrex-symbol.png"
              alt="Vyrex AI"
              fill
              className="object-contain"
              priority
            />
          </div>

          {/* Brand Titles */}
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

        {/* ── Right Controls: ONLY Dark and Light Mode ── */}
        <div className="flex items-center gap-2">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
