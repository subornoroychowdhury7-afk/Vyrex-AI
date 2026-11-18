"use client";

import Image from "next/image";
import { X, Cpu, GraduationCap, Github, ExternalLink, Code2 } from "lucide-react";

interface CreatorModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function CreatorModal({ isOpen, onClose }: CreatorModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
      <div
        className="relative w-full max-w-lg overflow-hidden rounded-2xl shadow-2xl transition-all
          bg-white dark:bg-[#161616]
          border border-gray-200 dark:border-[#2a2a2a]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header Banner */}
        <div className="relative p-6 bg-gradient-to-r from-red-700 via-red-600 to-red-800 text-white overflow-hidden">
          {/* Decorative orb */}
          <div className="absolute -top-6 -right-6 w-32 h-32 rounded-full bg-white/5" />
          <div className="absolute bottom-0 left-1/2 w-64 h-1 bg-white/10 rounded-full" />

          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-1.5 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>

          <div className="flex items-center gap-4">
            <div className="relative w-16 h-16 flex-shrink-0 rounded-xl overflow-hidden bg-white/10">
              <Image
                src="/vyrex-logo.png"
                alt="Vyrex AI Logo"
                fill
                className="object-contain p-1"
              />
            </div>
            <div>
              <span className="inline-flex px-2.5 py-0.5 rounded-full text-xs font-semibold bg-white/15 uppercase tracking-wider mb-1.5">
                Project Architect
              </span>
              <h2 className="text-2xl font-bold tracking-tight">Suborno Roy Chowdhury</h2>
              <p className="text-sm text-red-100 mt-0.5">AI/ML Engineer & Generative AI Researcher</p>
            </div>
          </div>
        </div>

        {/* Body */}
        <div className="p-6 space-y-4 max-h-[65vh] overflow-y-auto">

          {/* Education */}
          <div className="flex items-start gap-3.5 p-3.5 rounded-xl
            bg-gray-50 dark:bg-[#1f1f1f]
            border border-gray-100 dark:border-[#2a2a2a]">
            <GraduationCap className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
            <div className="text-sm">
              <p className="font-semibold text-gray-900 dark:text-white">
                St. Thomas&apos; College of Engineering & Technology, Kolkata
              </p>
              <p className="text-gray-500 dark:text-gray-400 mt-0.5">
                B.Tech in Artificial Intelligence & Machine Learning (YGPA: 8.71)
              </p>
              <p className="text-xs text-red-600 dark:text-red-400 font-medium mt-1">
                + IIT Kharagpur Certificate in AI for Real-World Applications
              </p>
            </div>
          </div>

          {/* Model Architecture */}
          <div className="flex items-start gap-3.5 p-3.5 rounded-xl
            bg-gray-50 dark:bg-[#1f1f1f]
            border border-gray-100 dark:border-[#2a2a2a]">
            <Cpu className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
            <div className="text-sm">
              <p className="font-semibold text-gray-900 dark:text-white">
                Vyrex AI • Vyrex Hyperion 3 Architecture
              </p>
              <p className="text-gray-500 dark:text-gray-400 mt-0.5 leading-relaxed">
                Fine-tuned on Qwen3-1.7B using 4-bit NF4 Quantization and QLoRA (Rank 16, Alpha 32).
                Engineered with a Calibrated Boundary Stop-Logits Processor (+4.8 bias) and 5× weighted end-token loss.
              </p>
            </div>
          </div>

          {/* Tech Stack */}
          <div className="flex items-start gap-3.5 p-3.5 rounded-xl
            bg-gray-50 dark:bg-[#1f1f1f]
            border border-gray-100 dark:border-[#2a2a2a]">
            <Code2 className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
            <div className="text-sm">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">Core Engineering Stack</p>
              <div className="flex flex-wrap gap-1.5">
                {["Python", "PyTorch", "QLoRA", "FastAPI", "Next.js", "Docker", "ChromaDB", "vLLM"].map((tech) => (
                  <span
                    key={tech}
                    className="px-2 py-0.5 rounded-md text-xs font-medium
                      bg-red-50 dark:bg-red-950/30
                      text-red-700 dark:text-red-300
                      border border-red-100 dark:border-red-900/30"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 px-6 border-t border-gray-100 dark:border-[#2a2a2a] bg-gray-50 dark:bg-[#111111] flex items-center justify-between">
          <a
            href="https://github.com/subornoroychowdhury7-afk/Vyrex-AI"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-xs font-semibold
              text-gray-600 dark:text-gray-300
              hover:text-red-600 dark:hover:text-red-400 transition-colors"
          >
            <Github className="w-4 h-4" />
            GitHub Repository
            <ExternalLink className="w-3 h-3" />
          </a>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-semibold text-white
              bg-red-600 hover:bg-red-700 transition-colors shadow-sm"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
