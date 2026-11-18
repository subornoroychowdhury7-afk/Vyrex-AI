"use client";

import { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";

export function ThemeToggle() {
  const [isDark, setIsDark] = useState<boolean>(false);
  const [mounted, setMounted] = useState<boolean>(false);

  useEffect(() => {
    setMounted(true);
    const isDarkMode = document.documentElement.classList.contains("dark");
    setIsDark(isDarkMode);
  }, []);

  const toggleTheme = () => {
    const nextDark = !isDark;
    setIsDark(nextDark);
    if (nextDark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  };

  if (!mounted) {
    return <div className="w-10 h-10 rounded-xl bg-gray-100 dark:bg-[#1f1f1f] animate-pulse border border-gray-200 dark:border-[#2a2a2a]" />;
  }

  return (
    <button
      onClick={toggleTheme}
      aria-label="Toggle Bright / Dark mode"
      title={isDark ? "Switch to Bright Mode" : "Switch to Dark Mode"}
      className="relative flex items-center justify-center w-10 h-10 rounded-xl transition-all duration-300
        bg-white dark:bg-[#1f1f1f]
        hover:bg-red-50 dark:hover:bg-red-950/30
        border border-gray-200 dark:border-[#2a2a2a]
        text-gray-600 dark:text-gray-300"
    >
      {isDark ? (
        <Sun className="w-5 h-5 text-amber-400 hover:rotate-45 transition-transform duration-300" />
      ) : (
        <Moon className="w-5 h-5 text-red-600 hover:-rotate-12 transition-transform duration-300" />
      )}
    </button>
  );
}
