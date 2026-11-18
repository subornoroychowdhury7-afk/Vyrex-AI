"use client";

import { useState } from "react";
import { User, Sparkles, Copy, Check } from "lucide-react";

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: string;
  isStreaming?: boolean;
}

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const copyToClipboard = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  // Helper to parse code blocks in markdown
  const renderContent = (content: string) => {
    const parts = content.split(/(```[\s\S]*?```)/g);

    return parts.map((part, index) => {
      if (part.startsWith("```") && part.endsWith("```")) {
        const lines = part.slice(3, -3).trim().split("\n");
        const firstLine = lines[0].trim();
        const hasLang = /^[a-zA-Z0-9_-]+$/.test(firstLine);
        const language = hasLang ? firstLine : "code";
        const codeText = hasLang ? lines.slice(1).join("\n") : lines.join("\n");

        return (
          <div key={index} className="my-3 rounded-xl overflow-hidden border border-slate-700/80 shadow-md">
            <div className="flex items-center justify-between px-4 py-1.5 bg-slate-900 border-b border-slate-800 text-xs text-slate-400">
              <span className="font-mono text-[11px] uppercase tracking-wider">{language}</span>
              <button
                onClick={() => copyToClipboard(codeText, index)}
                className="flex items-center gap-1.5 hover:text-slate-200 transition-colors py-0.5 px-2 rounded hover:bg-slate-800"
              >
                {copiedIndex === index ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-[11px] text-emerald-400">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span className="text-[11px]">Copy</span>
                  </>
                )}
              </button>
            </div>
            <pre className="m-0 p-4 text-xs font-mono bg-slate-950 text-slate-100 overflow-x-auto leading-relaxed">
              <code>{codeText}</code>
            </pre>
          </div>
        );
      }

      // Format bold text and linebreaks
      return (
        <div key={index} className="whitespace-pre-wrap leading-relaxed text-sm">
          {part}
        </div>
      );
    });
  };

  return (
    <div
      className={`flex gap-3.5 py-4 px-4 sm:px-6 rounded-2xl transition-colors duration-200 ${
        isUser
          ? "bg-slate-100/70 dark:bg-slate-800/40 border border-slate-200/50 dark:border-slate-800/40"
          : "bg-white dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800/80 shadow-sm"
      }`}
    >
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 shadow-sm ${
          isUser
            ? "bg-gradient-to-tr from-slate-600 to-slate-700 text-white"
            : "bg-gradient-to-tr from-indigo-600 via-indigo-700 to-purple-600 text-white shadow-indigo-500/20 shadow-md"
        }`}
      >
        {isUser ? <User className="w-4 h-4" /> : <Sparkles className="w-4 h-4" />}
      </div>

      {/* Message Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-semibold text-slate-800 dark:text-slate-200">
            {isUser ? "You" : "VYREX AI"}
          </span>
          <span className="text-[11px] text-slate-400 dark:text-slate-500 font-mono">
            {message.timestamp}
          </span>
        </div>

        <div className="text-slate-700 dark:text-slate-300">
          {renderContent(message.content)}
          {message.isStreaming && (
            <span className="inline-block w-2 h-4 ml-1 bg-indigo-500 animate-pulse rounded-sm" />
          )}
        </div>
      </div>
    </div>
  );
}
