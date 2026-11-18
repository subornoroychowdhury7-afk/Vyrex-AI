"use client";

import { useState } from "react";
import Image from "next/image";
import {
  Plus,
  MessageSquare,
  Trash2,
  Search,
  Clock,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import { Message } from "./ChatMessage";

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

interface ChatHistorySidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  sessions: ChatSession[];
  currentSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string) => void;
  onClearAll: () => void;
}

export function ChatHistorySidebar({
  isOpen,
  onToggle,
  sessions,
  currentSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onClearAll,
}: ChatHistorySidebarProps) {
  const [searchQuery, setSearchQuery] = useState("");

  const filteredSessions = sessions.filter((s) =>
    s.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const isToday =
      date.getDate() === now.getDate() &&
      date.getMonth() === now.getMonth() &&
      date.getFullYear() === now.getFullYear();

    if (isToday) {
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }
    return date.toLocaleDateString([], { month: "short", day: "numeric" });
  };

  if (!isOpen) {
    return (
      <button
        onClick={onToggle}
        aria-label="Open Sidebar"
        title="Open Chat History"
        className="fixed top-4 left-4 z-30 flex items-center justify-center w-10 h-10 rounded-xl transition-all duration-200
          bg-white dark:bg-[#161616]
          hover:bg-red-50 dark:hover:bg-red-950/30
          border border-gray-200 dark:border-[#2a2a2a]
          text-gray-600 dark:text-gray-300
          hover:text-red-600 dark:hover:text-red-400 shadow-sm"
      >
        <PanelLeftOpen className="w-5 h-5" />
      </button>
    );
  }

  return (
    <aside
      className="w-64 sm:w-72 lg:w-80 h-screen sticky top-0 flex flex-col flex-shrink-0
        bg-[#fafafa] dark:bg-[#121214]
        border-r border-gray-200 dark:border-[#242428]
        transition-all duration-300 z-30"
    >
      {/* ── Top Header: Brand + Collapse Toggle ── */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200/80 dark:border-[#242428]">
        <div className="flex items-center gap-2.5">
          <div className="relative w-7 h-7 flex-shrink-0">
            <Image
              src="/vyrex-symbol.png"
              alt="Vyrex AI"
              fill
              className="object-contain"
              priority
            />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-extrabold text-sm tracking-tight text-gray-900 dark:text-white">
                Vyrex <span className="text-red-600 dark:text-red-500">AI</span>
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded-full bg-red-100/70 dark:bg-red-950/40 text-red-700 dark:text-red-400 font-semibold">
                {sessions.length}
              </span>
            </div>
            <p className="text-[10px] text-gray-400 dark:text-gray-500 font-medium">
              Vyrex Hyperion 3
            </p>
          </div>
        </div>

        <button
          onClick={onToggle}
          className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-200/60 dark:hover:bg-[#1e1e24] transition-colors"
          title="Collapse sidebar"
          aria-label="Collapse sidebar"
        >
          <PanelLeftClose className="w-4 h-4" />
        </button>
      </div>

      {/* ── Action: New Chat Button ── */}
      <div className="p-3 border-b border-gray-200/60 dark:border-[#242428]">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 px-3.5 py-2.5 rounded-xl font-semibold text-xs text-white
            bg-red-600 hover:bg-red-700 shadow-sm hover:shadow-red-500/25 transition-all"
        >
          <Plus className="w-4 h-4" />
          <span>New Chat</span>
        </button>
      </div>

      {/* ── Search Bar (shows when multiple chats exist) ── */}
      {sessions.length > 2 && (
        <div className="px-3 pt-3">
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search chats..."
              className="w-full pl-8 pr-3 py-1.5 rounded-lg text-xs bg-white dark:bg-[#1a1a1e] border border-gray-200 dark:border-[#28282e] text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-red-500"
            />
          </div>
        </div>
      )}

      {/* ── Chat Sessions List ── */}
      <div className="flex-1 overflow-y-auto p-3 space-y-1">
        <div className="text-[10px] font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500 px-2 py-1">
          Recent Chats
        </div>

        {filteredSessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-44 text-center px-4">
            <Clock className="w-7 h-7 text-gray-300 dark:text-gray-700 mb-2" />
            <p className="text-xs font-medium text-gray-500 dark:text-gray-400">
              {searchQuery ? "No matching chats" : "No chats yet"}
            </p>
            <p className="text-[11px] text-gray-400 dark:text-gray-600 mt-1">
              Your conversations with Vyrex AI will appear here.
            </p>
          </div>
        ) : (
          filteredSessions.map((session) => {
            const isActive = session.id === currentSessionId;
            return (
              <div
                key={session.id}
                className={`group relative flex items-center justify-between p-2.5 rounded-xl text-left cursor-pointer transition-all ${
                  isActive
                    ? "bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900/50 text-red-700 dark:text-red-300 shadow-sm"
                    : "hover:bg-gray-200/60 dark:hover:bg-[#1a1a1e] text-gray-700 dark:text-gray-300 border border-transparent"
                }`}
                onClick={() => onSelectSession(session.id)}
              >
                <div className="flex items-center gap-2.5 min-w-0 pr-6">
                  <MessageSquare
                    className={`w-3.5 h-3.5 flex-shrink-0 ${
                      isActive
                        ? "text-red-600 dark:text-red-400"
                        : "text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300"
                    }`}
                  />
                  <div className="min-w-0">
                    <p className="text-xs font-medium truncate leading-tight">
                      {session.title || "New Conversation"}
                    </p>
                    <span className="text-[10px] text-gray-400 dark:text-gray-500">
                      {formatDate(session.updatedAt)}
                    </span>
                  </div>
                </div>

                {/* Delete button on hover */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteSession(session.id);
                  }}
                  title="Delete chat"
                  className="opacity-0 group-hover:opacity-100 p-1 rounded-md text-gray-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-white dark:hover:bg-[#25252b] transition-all absolute right-2"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            );
          })
        )}
      </div>

      {/* ── Bottom Actions ── */}
      {sessions.length > 0 && (
        <div className="p-3 border-t border-gray-200/80 dark:border-[#242428] bg-white/50 dark:bg-[#101012]">
          <button
            onClick={() => {
              if (confirm("Are you sure you want to clear all chat history?")) {
                onClearAll();
              }
            }}
            className="w-full flex items-center justify-center gap-1.5 py-1.5 px-3 rounded-lg text-xs font-medium text-gray-500 hover:text-red-600 dark:text-gray-400 dark:hover:text-red-400 hover:bg-gray-100 dark:hover:bg-[#1a1a1e] transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear All History</span>
          </button>
        </div>
      )}
    </aside>
  );
}
