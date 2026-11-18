"use client";

import { useState, useRef, useEffect } from "react";
import Image from "next/image";
import { User, Copy, Check } from "lucide-react";

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

// Split content into "settled" (old) and "new" (latest chunk) portions
// so we only fade the freshly arrived text.
function StreamingContent({ content }: { content: string }) {
  const prevRef = useRef<string>("");
  const settled = prevRef.current;
  const newChunk = content.slice(settled.length);

  // After render, advance settled to current content
  useEffect(() => {
    prevRef.current = content;
  });

  return (
    <>
      {settled && (
        <span className="whitespace-pre-wrap leading-relaxed text-sm">
          {settled}
        </span>
      )}
      {newChunk && (
        // key forces a fresh animation each time newChunk changes
        <span
          key={content.length}
          className="whitespace-pre-wrap leading-relaxed text-sm token-new"
        >
          {newChunk}
        </span>
      )}
    </>
  );
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
    <div className="flex gap-3.5 py-3 px-4 sm:px-6">
      {/* ─── Avatar ─── */}
      <div className={`w-8 h-8 flex items-center justify-center flex-shrink-0 ${
        isUser ? "rounded-xl bg-gradient-to-br from-gray-700 to-gray-800 text-white shadow-sm" : ""
      }`}>
        {isUser ? (
          <User className="w-4 h-4" />
        ) : message.isStreaming ? (
          /* Spinning while AI is answering (thinking + streaming) */
          <Image
            src="/vyrex-symbol.png"
            alt="Vyrex"
            width={28}
            height={28}
            className="animate-vyrex-spin"
          />
        ) : (
          /* Static when AI is done */
          <Image
            src="/vyrex-symbol.png"
            alt="Vyrex"
            width={28}
            height={28}
          />
        )}
      </div>

      {/* ─── Message Content ─── */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <span
            className={`text-xs font-semibold ${
              isUser ? "text-gray-700 dark:text-gray-300" : "text-red-600 dark:text-red-400"
            }`}
          >
            {isUser ? "You" : "Vyrex AI"}
          </span>
          <span className="text-[11px] text-gray-400 dark:text-gray-500 font-mono">
            {message.timestamp}
          </span>
        </div>

        <div className="text-gray-800 dark:text-gray-200">
          {!isUser && message.isStreaming && message.content === "" ? (
            /* ── Thinking state: animated dots ── */
            <div className="flex items-center gap-1.5 h-5 mt-1">
              <span className="text-[11px] text-red-400/70 dark:text-red-400/60 mr-1 font-medium tracking-wide">
                Thinking
              </span>
              {[1, 2, 3].map((i) => (
                <span
                  key={i}
                  className="thinking-dot inline-block w-1.5 h-1.5 rounded-full bg-red-500"
                />
              ))}
            </div>
          ) : message.isStreaming ? (
            /* ── Streaming state: fade-in new tokens ── */
            <>
              <StreamingContent content={message.content} />
              {/* blinking cursor while streaming */}
              <span className="inline-block w-[2px] h-4 ml-0.5 align-middle bg-red-500 animate-pulse rounded-sm" />
            </>
          ) : (
            /* ── Done: render full content normally ── */
            renderContent(message.content)
          )}
        </div>
      </div>
    </div>
  );
}
