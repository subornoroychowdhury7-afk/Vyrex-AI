"use client";

import { X, Award, Cpu, GraduationCap, Github, ExternalLink, Code2 } from "lucide-react";

interface CreatorModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function CreatorModal({ isOpen, onClose }: CreatorModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
      <div 
        className="relative w-full max-w-lg overflow-hidden rounded-2xl border border-slate-200 dark:border-slate-800 
          bg-white dark:bg-slate-900 shadow-2xl transition-all"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header Banner */}
        <div className="relative p-6 bg-gradient-to-r from-indigo-600 via-indigo-700 to-purple-700 text-white">
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-1.5 rounded-full bg-white/10 hover:bg-white/20 transition-colors text-white"
            aria-label="Close modal"
          >
            <X className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-3 mb-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-white/20 uppercase tracking-wider">
              Project Architect
            </span>
          </div>
          <h2 className="text-2xl font-bold tracking-tight">Suborno Roy Chowdhury</h2>
          <p className="text-sm text-indigo-100 font-medium mt-0.5">
            AI/ML Engineer & Generative AI Researcher
          </p>
        </div>

        {/* Content Body */}
        <div className="p-6 space-y-4 max-h-[75vh] overflow-y-auto">
          {/* Education */}
          <div className="flex items-start gap-3.5 p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800">
            <GraduationCap className="w-5 h-5 text-indigo-600 dark:text-indigo-400 mt-0.5 flex-shrink-0" />
            <div className="text-sm">
              <p className="font-semibold text-slate-900 dark:text-slate-100">
                St. Thomas' College of Engineering & Technology, Kolkata
              </p>
              <p className="text-slate-600 dark:text-slate-400 mt-0.5">
                B.Tech in Artificial Intelligence & Machine Learning (YGPA: 8.71)
              </p>
              <p className="text-xs text-indigo-600 dark:text-indigo-400 font-medium mt-1">
                + IIT Kharagpur Certificate in AI for Real-World Applications
              </p>
            </div>
          </div>

          {/* Model Architecture */}
          <div className="flex items-start gap-3.5 p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800">
            <Cpu className="w-5 h-5 text-purple-600 dark:text-purple-400 mt-0.5 flex-shrink-0" />
            <div className="text-sm">
              <p className="font-semibold text-slate-900 dark:text-slate-100">
                VYREX AI Architecture
              </p>
              <p className="text-slate-600 dark:text-slate-400 mt-0.5 leading-relaxed">
                Fine-tuned on Qwen3-1.7B utilizing 4-bit NF4 Quantization and QLoRA (Rank 16, Alpha 32).
                Engineered with a Calibrated Boundary Stop-Logits Processor (+4.8 bias) and 5× weighted end-token loss.
              </p>
            </div>
          </div>

          {/* Core Tech Stack */}
          <div className="flex items-start gap-3.5 p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800">
            <Code2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400 mt-0.5 flex-shrink-0" />
            <div className="text-sm">
              <p className="font-semibold text-slate-900 dark:text-slate-100">
                Core Engineering Stack
              </p>
              <div className="flex flex-wrap gap-1.5 mt-2">
                {["Python", "PyTorch", "QLoRA", "FastAPI", "Next.js", "Docker", "ChromaDB", "vLLM"].map((tech) => (
                  <span
                    key={tech}
                    className="px-2 py-0.5 rounded-md text-xs font-medium bg-slate-200/70 dark:bg-slate-700/60 text-slate-700 dark:text-slate-300"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 px-6 bg-slate-50 dark:bg-slate-950/50 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
          <a
            href="https://github.com/subornoroychowdhury7-afk/Vyrex-AI"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-xs font-semibold text-slate-600 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
          >
            <Github className="w-4 h-4" />
            GitHub Repository
            <ExternalLink className="w-3 h-3" />
          </a>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-indigo-600 text-white hover:bg-indigo-700 transition-colors shadow-sm"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
