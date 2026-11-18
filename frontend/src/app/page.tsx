"use client";

import { useState, useEffect, useRef } from "react";
import Image from "next/image";
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
    title: "ML Concept",
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
      const res = await fetch(`${url.replace(/\/$/, "")}/health`);
      setStatus(res.ok ? "connected" : "offline");
    } catch {
      setStatus("offline");
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
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
              } catch { /* skip */ }
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
                  content: "⚠️ Could not reach the Vyrex backend. Please check your settings.",
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

  const handleStop = () => abortControllerRef.current?.abort();
  const handleClearHistory = () => setMessages([]);

  return (
    <div className="flex flex-col min-h-screen bg-white dark:bg-[#0d0d0d] transition-colors duration-300">

      <Header
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenCreator={() => setIsCreatorOpen(true)}
        status={status}
        apiUrl={apiUrl}
      />

      <main className="flex-1 max-w-4xl w-full mx-auto px-4 py-6 flex flex-col justify-between">

        {messages.length === 0 ? (
          /* ── Welcome / Empty state ── */
          <div className="my-auto py-12 flex flex-col items-center text-center animate-fade-in">

            {/* Logo */}
            <div className="relative w-28 h-28 mb-6 drop-shadow-xl">
              <Image
                src="/vyrex-logo.png"
                alt="Vyrex AI"
                fill
                className="object-contain"
                priority
              />
            </div>

            <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight mb-2">
              <span className="text-gray-900 dark:text-white">Meet </span>
              <span className="text-red-600 dark:text-red-500">Vyrex AI</span>
            </h2>
            <p className="max-w-lg text-sm sm:text-base text-gray-500 dark:text-gray-400 mb-2 leading-relaxed">
              Powered by{" "}
              <span className="font-semibold text-gray-800 dark:text-gray-200">Vyrex Hyperion 3</span>
              {" "}— an independent conversational AI fine-tuned with 4-bit QLoRA and
              calibrated natural termination.
            </p>
            <p className="text-xs text-gray-400 dark:text-gray-500 mb-10">
              Architected by Suborno Roy Chowdhury
            </p>

            {/* Starter Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-2xl text-left">
              {STARTER_PROMPTS.map((card, idx) => {
                const IconComponent = card.icon;
                return (
                  <button
                    key={idx}
                    onClick={() => handleSubmit(undefined, card.prompt)}
                    className="p-4 rounded-2xl text-left group transition-all duration-200
                      bg-white dark:bg-[#161616]
                      border border-gray-200 dark:border-[#2a2a2a]
                      hover:border-red-400 dark:hover:border-red-700
                      shadow-sm hover:shadow-md hover:shadow-red-500/10"
                  >
                    <div className="flex items-center gap-2 mb-1.5 text-xs font-semibold text-red-600 dark:text-red-400">
                      <IconComponent className="w-4 h-4" />
                      <span>{card.title}</span>
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

        {/* ── Input Card ── */}
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

            {/* Bottom toolbar */}
            <div className="flex items-center justify-between px-3.5 pb-3 pt-1">

              {/* Left controls */}
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  title="Attach / New"
                  className="p-1.5 rounded-lg text-gray-400
                    hover:text-red-600 dark:hover:text-red-400
                    hover:bg-red-50 dark:hover:bg-red-950/30
                    transition-colors"
                >
                  <Plus className="w-4 h-4" />
                </button>
                <div className="px-2.5 py-1 text-xs font-medium rounded-lg
                  bg-gray-100 dark:bg-[#1f1f1f]
                  text-gray-700 dark:text-gray-300
                  border border-gray-200 dark:border-[#2a2a2a]">
                  Chat
                </div>
                {messages.length > 0 && (
                  <button
                    type="button"
                    onClick={handleClearHistory}
                    title="Clear conversation"
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

              {/* Right: Model label + Send/Stop */}
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
            Vyrex AI · Vyrex Hyperion 3 · Fine-tuned by Suborno Roy Chowdhury
          </p>
        </div>
      </main>

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
