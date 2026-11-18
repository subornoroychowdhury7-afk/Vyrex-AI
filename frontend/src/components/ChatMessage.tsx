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
          <div key={index} className="my-3 rounded-xl overflow-hidden border border-[#2a2a2a] shadow-md">
            <div className="flex items-center justify-between px-4 py-1.5 bg-[#1a1a1a] border-b border-[#2a2a2a] text-xs text-gray-400">
              <span className="font-mono text-[11px] uppercase tracking-wider text-red-400">{language}</span>
              <button
                onClick={() => copyToClipboard(codeText, index)}
                className="flex items-center gap-1.5 hover:text-white transition-colors py-0.5 px-2 rounded hover:bg-[#2a2a2a]"
              >
                {copiedIndex === index ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-green-400" />
                    <span className="text-[11px] text-green-400">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span className="text-[11px]">Copy</span>
                  </>
                )}
              </button>
            </div>
            <pre className="m-0 p-4 text-xs font-mono bg-[#111111] text-gray-100 overflow-x-auto leading-relaxed">
              <code>{codeText}</code>
            </pre>
          </div>
        );
      }

      return (
        <div key={index} className="whitespace-pre-wrap leading-relaxed text-sm">
          {part}
        </div>
      );
    });
  };

  return (
    <div
      className={`flex gap-3.5 py-4 px-4 sm:px-5 rounded-2xl transition-colors duration-200 ${
        isUser
          /* User bubble — light: very light red tint / dark: near-black with subtle red border */
          ? "bg-red-50/70 dark:bg-[#1a0a0a]/60 border border-red-100 dark:border-red-950/40"
          /* AI bubble — light: white card / dark: dark surface */
          : "bg-white dark:bg-[#161616] border border-gray-200 dark:border-[#2a2a2a] shadow-sm"
      }`}
    >
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 shadow-sm ${
          isUser
            ? "bg-gradient-to-br from-gray-700 to-gray-800 text-white"
            : "bg-gradient-to-br from-red-600 to-red-800 text-white shadow-red-500/20 shadow-md"
        }`}
      >
        {isUser ? <User className="w-4 h-4" /> : <Sparkles className="w-4 h-4" />}
      </div>

      {/* Message Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <span className={`text-xs font-semibold ${
            isUser ? "text-gray-700 dark:text-gray-300" : "text-red-600 dark:text-red-400"
          }`}>
            {isUser ? "You" : "Vyrex AI"}
          </span>
          <span className="text-[11px] text-gray-400 dark:text-gray-500 font-mono">
            {message.timestamp}
          </span>
        </div>

        <div className="text-gray-800 dark:text-gray-200">
          {renderContent(message.content)}
          {message.isStreaming && (
            <span className="inline-block w-2 h-4 ml-1 bg-red-500 animate-pulse rounded-sm" />
          )}
        </div>
      </div>
    </div>
  );
}
