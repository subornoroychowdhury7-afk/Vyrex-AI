"use client";

import { useState, useEffect, useRef } from "react";
import { ArrowUp, Plus, Square, Trash2, Sparkles, Terminal, Code2, Cpu } from "lucide-react";
import { Header } from "@/components/Header";
import { ChatMessage, Message } from "@/components/ChatMessage";
import { SettingsModal } from "@/components/SettingsModal";
import { CreatorModal } from "@/components/CreatorModal";

const STARTER_PROMPTS = [
  {
    icon: Sparkles,
    title: "Identity & Creator",
    prompt: "Who created you and what is your architecture?",
  },
  {
    icon: Cpu,
    title: "4-bit NF4 Quantization",
    prompt: "Explain how 4-bit NF4 quantization works in exactly two sentences. Do not include an introduction or conclusion.",
  },
  {
    icon: Code2,
    title: "Python Algorithm",
    prompt: "Write a Python function called find_duplicates(items) that returns duplicate elements from a list preserving order, without using set(). Include type hints.",
  },
  {
    icon: Terminal,
    title: "Machine Learning Concept",
    prompt: "What is the difference between supervised and unsupervised learning?",
  },
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<"connected" | "connecting" | "offline">("connecting");
  const [apiUrl, setApiUrl] = useState<string>("http://localhost:8000");
  const [temperature, setTemperature] = useState<number>(0.7);
  const [maxTokens, setMaxTokens] = useState<number>(512);

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isCreatorOpen, setIsCreatorOpen] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize API URL from env or localStorage
  useEffect(() => {
    const savedUrl = localStorage.getItem("vyrex_api_url");
    const defaultUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const targetUrl = savedUrl || defaultUrl;
    setApiUrl(targetUrl);
    checkHealth(targetUrl);
  }, []);

  const handleUpdateApiUrl = (newUrl: string) => {
    setApiUrl(newUrl);
    localStorage.setItem("vyrex_api_url", newUrl);
    checkHealth(newUrl);
  };

  const checkHealth = async (url: string) => {
    setStatus("connecting");
    try {
      const res = await fetch(`${url.replace(/\/$/, "")}/health`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });
      if (res.ok) {
        setStatus("connected");
      } else {
        setStatus("offline");
      }
    } catch {
      setStatus("offline");
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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

    setMessages((prev) => [...prev, userMessage, initialAssistantMessage]);
    setInput("");
    setIsLoading(true);

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
          temperature,
          max_tokens: maxTokens,
          stream: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder("utf-8");
      let assistantText = "";

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split("\n");

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const dataStr = line.slice(6).trim();
              if (dataStr === "[DONE]") {
                break;
              }
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
                // Skip incomplete JSON chunks
              }
            }
          }
        }
      }

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId ? { ...msg, isStreaming: false } : msg
        )
      );
      setStatus("connected");
    } catch (err: any) {
      if (err.name === "AbortError") {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? { ...msg, content: msg.content + " [Generation Stopped]", isStreaming: false }
              : msg
          )
        );
      } else {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? {
                  ...msg,
                  content:
                    "⚠️ Could not reach the VYREX backend. Please ensure the backend is running or check your settings.",
                  isStreaming: false,
                }
              : msg
          )
        );
        setStatus("offline");
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const handleClearHistory = () => {
    setMessages([]);
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Top Navigation */}
      <Header
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenCreator={() => setIsCreatorOpen(true)}
        status={status}
        apiUrl={apiUrl}
      />

      {/* Main Chat Canvas */}
      <main className="flex-1 max-w-4xl w-full mx-auto px-4 py-6 flex flex-col justify-between">
        {messages.length === 0 ? (
          /* Empty State / Welcome Screen */
          <div className="my-auto py-12 flex flex-col items-center text-center animate-fade-in">
            <div className="relative mb-6">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-indigo-600 via-indigo-700 to-purple-600 flex items-center justify-center text-white shadow-xl shadow-indigo-500/25">
                <Sparkles className="w-8 h-8" />
              </div>
              <span className="absolute -bottom-1 -right-1 px-1.5 py-0.5 rounded-md text-[9px] font-bold bg-emerald-500 text-white uppercase tracking-wider shadow">
                Active
              </span>
            </div>

            <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-slate-100 mb-3">
              Meet <span className="bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">VYREX AI</span>
            </h2>

            <p className="max-w-xl text-sm sm:text-base text-slate-600 dark:text-slate-400 mb-8 leading-relaxed">
              Powered by <span className="font-semibold text-slate-800 dark:text-slate-200">Vyrex Hyperion 3</span> — an independent conversational AI fine-tuned with 4-bit QLoRA and calibrated natural termination.
            </p>

            {/* Quick Starter Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 w-full max-w-2xl text-left">
              {STARTER_PROMPTS.map((card, idx) => {
                const IconComponent = card.icon;
                return (
                  <button
                    key={idx}
                    onClick={() => handleSubmit(undefined, card.prompt)}
                    className="p-4 rounded-2xl bg-white dark:bg-slate-900/90 border border-slate-200 dark:border-slate-800/80 
                      hover:border-indigo-500/50 dark:hover:border-indigo-500/50 shadow-sm hover:shadow-md transition-all duration-200 group text-left"
                  >
                    <div className="flex items-center gap-2 mb-1.5 text-xs font-semibold text-indigo-600 dark:text-indigo-400">
                      <IconComponent className="w-4 h-4" />
                      <span>{card.title}</span>
                    </div>
                    <p className="text-xs text-slate-600 dark:text-slate-300 line-clamp-2 leading-relaxed">
                      "{card.prompt}"
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          /* Message List */
          <div className="space-y-4 mb-6">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}

        {/* Input Bar Section */}
        <div className="sticky bottom-4 z-20 mt-auto">
          {/* Form Input Card matching Claude layout */}
          <form
            onSubmit={handleSubmit}
            className="relative rounded-2xl bg-white dark:bg-[#18191e] border border-slate-200/90 dark:border-slate-800 
              shadow-lg shadow-slate-200/40 dark:shadow-black/20 focus-within:border-indigo-500/70 dark:focus-within:border-indigo-500/60 
              transition-all overflow-hidden"
          >
            {/* Top Textarea */}
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
                className="w-full max-h-40 min-h-[50px] bg-transparent text-slate-900 dark:text-slate-100 
                  placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none resize-none leading-relaxed text-sm sm:text-base"
              />
            </div>

            {/* Bottom Toolbar Row */}
            <div className="flex items-center justify-between px-3.5 pb-2.5 pt-1">
              {/* Left Action Controls */}
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  title="Add prompt / Attachment"
                  className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800/80 transition-colors"
                >
                  <Plus className="w-4 h-4" />
                </button>
                <div className="px-2.5 py-1 text-xs font-medium rounded-lg bg-slate-100 dark:bg-slate-800/90 text-slate-700 dark:text-slate-200 border border-slate-200/60 dark:border-slate-700/50">
                  Chat
                </div>
                {messages.length > 0 && (
                  <button
                    type="button"
                    onClick={handleClearHistory}
                    title="Clear conversation"
                    className="flex items-center gap-1 px-2 py-1 text-xs text-slate-400 hover:text-rose-500 dark:hover:text-rose-400 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800/80 transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Clear</span>
                  </button>
                )}
              </div>

              {/* Right: Model Name & Send / Stop Button */}
              <div className="flex items-center gap-3">
                <span className="text-xs sm:text-sm font-medium text-slate-500 dark:text-slate-400 select-none cursor-default tracking-tight">
                  Vyrex Hyperion 3
                </span>

                {isLoading ? (
                  <button
                    type="button"
                    onClick={handleStop}
                    className="flex items-center justify-center w-8 h-8 rounded-xl bg-rose-600 hover:bg-rose-700 text-white shadow-sm transition-colors"
                    title="Stop generation"
                  >
                    <Square className="w-3.5 h-3.5 fill-white" />
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!input.trim()}
                    className="flex items-center justify-center w-8 h-8 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-30 disabled:hover:bg-indigo-600 text-white shadow-sm transition-all"
                    title="Send message"
                  >
                    <ArrowUp className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          </form>

          <p className="text-[11px] text-center text-slate-400 dark:text-slate-500 mt-2">
            Vyrex AI • Vyrex Hyperion 3. Fine-tuned with calibrated natural boundary termination. Architected by Suborno Roy Chowdhury.
          </p>
        </div>
      </main>

      {/* Modals */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        apiUrl={apiUrl}
        setApiUrl={handleUpdateApiUrl}
        temperature={temperature}
        setTemperature={setTemperature}
        maxTokens={maxTokens}
        setMaxTokens={setMaxTokens}
      />

      <CreatorModal
        isOpen={isCreatorOpen}
        onClose={() => setIsCreatorOpen(false)}
      />
    </div>
  );
}
