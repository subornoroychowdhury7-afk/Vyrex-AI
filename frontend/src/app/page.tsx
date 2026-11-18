"use client";

import { useState, useEffect, useRef } from "react";
import Image from "next/image";
import {
  ArrowUp,
  Plus,
  Square,
  Trash2,
  Sparkles,
  Code2,
  Lightbulb,
  PenTool,
  BookOpen,
  Compass,
} from "lucide-react";
import { Header } from "@/components/Header";
import { ChatMessage, Message } from "@/components/ChatMessage";
import { ChatHistorySidebar, ChatSession } from "@/components/ChatHistorySidebar";

interface PromptItem {
  id: string;
  category: "brainstorm" | "code" | "write" | "learn" | "discover";
  icon: any;
  title: string;
  subtitle: string;
  prompt: string;
}

const PROMPT_SUGGESTIONS: PromptItem[] = [
  {
    id: "p1",
    category: "brainstorm",
    icon: Lightbulb,
    title: "Brainstorm Ideas",
    subtitle: "Creative concepts & innovation",
    prompt: "Give me 5 groundbreaking startup ideas using edge AI for 2026.",
  },
  {
    id: "p2",
    category: "code",
    icon: Code2,
    title: "Code & Architecture",
    subtitle: "Clean code & backend patterns",
    prompt: "Write a high-performance Python function with type hints and edge-case tests.",
  },
  {
    id: "p3",
    category: "write",
    icon: PenTool,
    title: "Write & Refine",
    subtitle: "Compelling copy & executive pitch",
    prompt: "Draft an engaging and persuasive product launch announcement for an AI platform.",
  },
  {
    id: "p4",
    category: "learn",
    icon: BookOpen,
    title: "Explain & Explore",
    subtitle: "Intuitive breakdowns of deep topics",
    prompt: "Explain how neural networks learn and optimize weights using simple, intuitive analogies.",
  },
  {
    id: "p5",
    category: "code",
    icon: Code2,
    title: "API & Backend Design",
    subtitle: "FastAPI, async & database best practices",
    prompt: "Show me an idiomatic async FastAPI route with request validation and clean error handling.",
  },
  {
    id: "p6",
    category: "discover",
    icon: Sparkles,
    title: "Who Created You?",
    subtitle: "Meet the creator & architecture",
    prompt: "Who created you and what is your architecture?",
  },
];

const CATEGORIES = [
  { id: "all", label: "✨ All" },
  { id: "brainstorm", label: "💡 Brainstorm" },
  { id: "code", label: "💻 Code" },
  { id: "write", label: "✍️ Write" },
  { id: "learn", label: "🔬 Learn" },
];

const SESSIONS_STORAGE_KEY = "vyrex_chat_sessions";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [apiUrl, setApiUrl] = useState<string>("http://localhost:8000");
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [greeting, setGreeting] = useState<string>("Hello");

  // Chat History state
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Time-based greeting like Claude / Gemini
  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting("Good morning");
    else if (hour < 17) setGreeting("Good afternoon");
    else setGreeting("Good evening");
  }, []);

  // Load API URL & Chat Sessions from localStorage on mount
  useEffect(() => {
    const savedUrl = localStorage.getItem("vyrex_api_url");
    const defaultUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    setApiUrl(savedUrl || defaultUrl);

    try {
      const savedSessions = localStorage.getItem(SESSIONS_STORAGE_KEY);
      if (savedSessions) {
        const parsed = JSON.parse(savedSessions);
        if (Array.isArray(parsed)) {
          setSessions(parsed);
        }
      }
    } catch (e) {
      console.error("Failed to load chat history from localStorage:", e);
    }
  }, []);

  // Save sessions to localStorage
  const saveSessions = (updated: ChatSession[]) => {
    setSessions(updated);
    try {
      localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(updated));
    } catch (e) {
      console.error("Failed to save chat sessions:", e);
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle switching to a past chat session
  const handleSelectSession = (sessionId: string) => {
    const target = sessions.find((s) => s.id === sessionId);
    if (target) {
      setCurrentSessionId(target.id);
      setMessages(target.messages || []);
    }
  };

  // Start a fresh new chat
  const handleNewChat = () => {
    if (isLoading && abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setCurrentSessionId(null);
    setMessages([]);
    setInput("");
  };

  // Delete an individual chat session
  const handleDeleteSession = (sessionId: string) => {
    const updated = sessions.filter((s) => s.id !== sessionId);
    saveSessions(updated);
    if (currentSessionId === sessionId) {
      handleNewChat();
    }
  };

  // Clear all sessions
  const handleClearAllHistory = () => {
    saveSessions([]);
    handleNewChat();
  };

  // Submit message & handle streaming
  const handleSubmit = async (e?: React.FormEvent, customPrompt?: string) => {
    if (e) e.preventDefault();
    const promptToSend = customPrompt || input.trim();
    if (!promptToSend || isLoading) return;

    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: "user",
      content: promptToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    const assistantMessageId = `asst-${Date.now()}`;
    const initialAssistantMessage: Message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      isStreaming: true,
    };

    const updatedMessages = [...messages, userMessage, initialAssistantMessage];
    setMessages(updatedMessages);
    setInput("");
    setIsLoading(true);

    // Determine session ID
    let activeSessionId = currentSessionId;
    let currentTitle = "";
    if (!activeSessionId) {
      activeSessionId = `session_${Date.now()}`;
      currentTitle = promptToSend.slice(0, 36) + (promptToSend.length > 36 ? "..." : "");
      setCurrentSessionId(activeSessionId);
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const conversationHistory = [...messages, userMessage].map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const response = await fetch(`${apiUrl.replace(/\/$/, "")}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          model: "Vyrex Hyperion 3",
          messages: conversationHistory,
          temperature: 0.0,
          max_tokens: 512,
          stream: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder("utf-8");
      let assistantText = "";

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          for (const line of chunk.split("\n")) {
            if (line.startsWith("data: ")) {
              const dataStr = line.slice(6).trim();
              if (dataStr === "[DONE]") break;
              try {
                const data = JSON.parse(dataStr);
                const deltaContent = data.choices?.[0]?.delta?.content || "";
                assistantText += deltaContent;
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessageId
                      ? { ...msg, content: assistantText, isStreaming: true }
                      : msg
                  )
                );
              } catch {
                // Skip incomplete chunks
              }
            }
          }
        }
      }

      // Finalize assistant message
      const finalMessages = updatedMessages.map((msg) =>
        msg.id === assistantMessageId
          ? { ...msg, content: assistantText, isStreaming: false }
          : msg
      );
      setMessages(finalMessages);

      // Save to sessions in localStorage
      setSessions((prevSessions) => {
        const existingIdx = prevSessions.findIndex((s) => s.id === activeSessionId);
        let nextSessions: ChatSession[];
        if (existingIdx >= 0) {
          nextSessions = [...prevSessions];
          nextSessions[existingIdx] = {
            ...nextSessions[existingIdx],
            messages: finalMessages,
            updatedAt: Date.now(),
          };
        } else {
          const newSession: ChatSession = {
            id: activeSessionId!,
            title: currentTitle || promptToSend.slice(0, 36),
            messages: finalMessages,
            createdAt: Date.now(),
            updatedAt: Date.now(),
          };
          nextSessions = [newSession, ...prevSessions];
        }
        try {
          localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(nextSessions));
        } catch (e) {
          console.error("Storage error:", e);
        }
        return nextSessions;
      });

    } catch (err: any) {
      if (err.name === "AbortError") {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? { ...msg, content: msg.content + " [Stopped]", isStreaming: false }
              : msg
          )
        );
      } else {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? {
                  ...msg,
                  content: "⚠️ Could not reach the Vyrex backend on localhost:8000. Please ensure the backend is running.",
                  isStreaming: false,
                }
              : msg
          )
        );
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleStop = () => abortControllerRef.current?.abort();

  const filteredPrompts =
    activeCategory === "all"
      ? PROMPT_SUGGESTIONS
      : PROMPT_SUGGESTIONS.filter((p) => p.category === activeCategory);

  return (
    <div className="flex flex-col min-h-screen bg-white dark:bg-[#0d0d0d] transition-colors duration-300">
      {/* Top Header — Clean with Chat History toggle and Theme Toggle ONLY */}
      <Header
        onOpenHistory={() => setIsHistoryOpen(true)}
        historyCount={sessions.length}
      />

      {/* Slide-out Chat History Drawer */}
      <ChatHistorySidebar
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
        onClearAll={handleClearAllHistory}
      />

      <main className="flex-1 max-w-4xl w-full mx-auto px-4 py-6 flex flex-col justify-between">
        {messages.length === 0 ? (
          /* ── Welcome Screen matching modern Claude / Gemini / ChatGPT design ── */
          <div className="my-auto py-8 flex flex-col items-center text-center animate-fade-in">
            {/* Logo */}
            <div className="relative w-24 h-24 mb-4 drop-shadow-xl hover:scale-105 transition-transform duration-300">
              <Image
                src="/vyrex-logo.png"
                alt="Vyrex AI"
                fill
                className="object-contain"
                priority
              />
            </div>

            {/* Line 1: Kept strictly as Vyrex AI */}
            <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight mb-2">
              <span className="text-gray-900 dark:text-white">Vyrex </span>
              <span className="text-red-600 dark:text-red-500">AI</span>
            </h2>

            {/* Line 2: Human greeting & question */}
            <p className="text-xl sm:text-2xl font-semibold text-gray-800 dark:text-gray-100 mb-2">
              {greeting}, where should we start?
            </p>

            {/* Line 3: Friendly subtitle */}
            <p className="text-xs sm:text-sm text-gray-500 dark:text-gray-400 mb-6 max-w-md leading-relaxed">
              Ask anything, brainstorm bold ideas, write clean code, or explore deep concepts.
            </p>

            {/* Line 4: Interactive category pills */}
            <div className="flex flex-wrap items-center justify-center gap-2 mb-6">
              {CATEGORIES.map((cat) => {
                const isActive = activeCategory === cat.id;
                return (
                  <button
                    key={cat.id}
                    onClick={() => setActiveCategory(cat.id)}
                    className={`px-3.5 py-1.5 rounded-full text-xs font-semibold transition-all duration-200 ${
                      isActive
                        ? "bg-red-600 text-white shadow-md shadow-red-500/25 scale-105"
                        : "bg-gray-100 dark:bg-[#1a1a1e] text-gray-600 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/30 border border-gray-200 dark:border-[#2a2a2a]"
                    }`}
                  >
                    {cat.label}
                  </button>
                );
              })}
            </div>

            {/* Line 5: Interactive & Cool Starter Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-2xl text-left">
              {filteredPrompts.slice(0, 4).map((card) => {
                const IconComponent = card.icon;
                return (
                  <button
                    key={card.id}
                    onClick={() => handleSubmit(undefined, card.prompt)}
                    className="p-4 rounded-2xl text-left group transition-all duration-200
                      bg-white dark:bg-[#161616]
                      border border-gray-200 dark:border-[#2a2a2a]
                      hover:border-red-500 dark:hover:border-red-600
                      shadow-sm hover:shadow-lg hover:shadow-red-500/10 hover:-translate-y-0.5"
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-2 text-xs font-bold text-gray-900 dark:text-gray-100 group-hover:text-red-600 dark:group-hover:text-red-400 transition-colors">
                        <div className="w-6 h-6 rounded-lg bg-red-50 dark:bg-red-950/40 text-red-600 dark:text-red-400 flex items-center justify-center">
                          <IconComponent className="w-3.5 h-3.5" />
                        </div>
                        <span>{card.title}</span>
                      </div>
                      <span className="text-[10px] text-gray-400 dark:text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity">
                        Ask &rarr;
                      </span>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400 line-clamp-2 leading-relaxed">
                      &ldquo;{card.prompt}&rdquo;
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          /* ── Message List ── */
          <div className="space-y-3 mb-6 animate-fade-in">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}

        {/* ── Chat Input Card ── */}
        <div className="sticky bottom-4 z-20 mt-auto">
          <form
            onSubmit={handleSubmit}
            className="rounded-2xl overflow-hidden transition-all duration-200
              bg-white dark:bg-[#161616]
              border border-gray-200 dark:border-[#2a2a2a]
              focus-within:border-red-400 dark:focus-within:border-red-700
              shadow-lg shadow-gray-200/60 dark:shadow-black/40"
          >
            {/* Textarea */}
            <div className="px-4 pt-3.5 pb-1">
              <textarea
                rows={2}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit();
                  }
                }}
                placeholder="How can I help you today?"
                className="w-full max-h-40 min-h-[52px] bg-transparent
                  text-gray-900 dark:text-gray-100
                  placeholder-gray-400 dark:placeholder-gray-600
                  focus:outline-none resize-none leading-relaxed text-sm sm:text-[15px]"
              />
            </div>

            {/* Bottom Toolbar */}
            <div className="flex items-center justify-between px-3.5 pb-3 pt-1">
              {/* Left Controls */}
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={handleNewChat}
                  title="New Chat"
                  className="flex items-center gap-1 px-2.5 py-1 text-xs rounded-lg text-gray-500 dark:text-gray-300 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/30 transition-colors border border-gray-200 dark:border-[#2a2a2a]"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>New Chat</span>
                </button>
                {messages.length > 0 && (
                  <button
                    type="button"
                    onClick={handleNewChat}
                    title="Clear current view"
                    className="flex items-center gap-1 px-2 py-1 text-xs rounded-lg
                      text-gray-400
                      hover:text-red-600 dark:hover:text-red-400
                      hover:bg-red-50 dark:hover:bg-red-950/30
                      transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Clear</span>
                  </button>
                )}
              </div>

              {/* Right: Model Name & Send / Stop Button */}
              <div className="flex items-center gap-3">
                <span className="text-xs sm:text-sm font-medium
                  text-gray-500 dark:text-gray-400
                  select-none cursor-default tracking-tight">
                  Vyrex Hyperion 3
                </span>

                {isLoading ? (
                  <button
                    type="button"
                    onClick={handleStop}
                    title="Stop generation"
                    className="flex items-center justify-center w-8 h-8 rounded-xl
                      bg-red-600 hover:bg-red-700
                      text-white shadow-sm transition-colors"
                  >
                    <Square className="w-3.5 h-3.5 fill-white" />
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!input.trim()}
                    title="Send message"
                    className="flex items-center justify-center w-8 h-8 rounded-xl text-white
                      bg-red-600 hover:bg-red-700
                      disabled:opacity-30 disabled:hover:bg-red-600
                      shadow-sm hover:shadow-red-500/40 hover:shadow-md
                      transition-all duration-150"
                  >
                    <ArrowUp className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          </form>

          <p className="text-[11px] text-center text-gray-400 dark:text-gray-600 mt-2">
            Vyrex AI · Vyrex Hyperion 3 · Independent Conversational Intelligence
          </p>
        </div>
      </main>
    </div>
  );
}
