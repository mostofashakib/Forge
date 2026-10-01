"use client";

import { useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  Boxes,
  CheckCircle2,
  Compass,
  Cpu,
  FlaskConical,
  Layers,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Workflow,
  Zap,
} from "lucide-react";

interface RouteGuide {
  id: string;
  title: string;
  route: string;
  badge: string;
  eyebrow: string;
  icon: typeof Sparkles;
  accent: string;
  headline: string;
  description: string;
  whenToUse: string;
  keyFeatures: string[];
  ctaLabel: string;
}

const ROUTE_GUIDES: RouteGuide[] = [
  {
    id: "generator",
    title: "Generator",
    route: "/generator",
    badge: "01 // SYNTHESIS",
    eyebrow: "Synthetic Data & Verification",
    icon: Sparkles,
    accent: "text-amber-500",
    headline: "Build Taxonomies & Synthesize Verifiable Datasets",
    description:
      "Generate domain-specific task taxonomies and thousands of executable problems with verified golden solutions. Noise is eliminated using pass^k multi-pass execution and cross-model family judging.",
    whenToUse:
      "Go here when you need high-quality RL task environments, DPO preference pairs, or clean SFT demonstrations.",
    keyFeatures: [
      "Multi-step task taxonomy generation",
      "Pass^k execution proof (k=1 to 5)",
      "Cross-model family validator judge",
      "Versioned JSONL dataset export",
    ],
    ctaLabel: "Open Generator",
  },
  {
    id: "training",
    title: "Training",
    route: "/training",
    badge: "02 // RLVR & SFT",
    eyebrow: "Policy Optimization Engine",
    icon: Cpu,
    accent: "text-primary",
    headline: "Train Models with Verifiable Execution Rewards",
    description:
      "Run state-of-the-art policy optimization with real-world execution rewards instead of subjective LLM vibes. Supports GRPO, PPO, DPO, and SFT with live telemetry feeds and checkpoint management.",
    whenToUse:
      "Go here when you have generated datasets or task directories and want to update model weights using RLVR or fine-tuning.",
    keyFeatures: [
      "GRPO, PPO, DPO, and SFT algorithms",
      "Online self-play vs. Offline cross-family alignment",
      "Real-time loss & mean reward telemetry",
      "One-click checkpointing and export",
    ],
    ctaLabel: "Launch Training",
  },
  {
    id: "benchmark",
    title: "Benchmark",
    route: "/benchmark",
    badge: "03 // EVALUATION",
    eyebrow: "Agent Performance & Auditing",
    icon: FlaskConical,
    accent: "text-emerald-500",
    headline: "Measure Pass Rates Across Verifiable Test Suites",
    description:
      "Evaluate frontier agents and fine-tuned checkpoints across parallel sandbox environments. Measure Pass@1, Pass@k, tool-use fidelity, and identify exact failure clusters.",
    whenToUse:
      "Go here when you want to measure model accuracy, compare checkpoints, inspect step-by-step rollout traces, or find edge-case failures.",
    keyFeatures: [
      "Parallel rollout workers in isolated sandboxes",
      "Pass@1 and Pass@k evaluation metrics",
      "Step-by-step trajectory replay & inspection",
      "Automated failure clustering & root-cause audits",
    ],
    ctaLabel: "Run Benchmarks",
  },
  {
    id: "environments",
    title: "Environments",
    route: "/environments",
    badge: "04 // RUNTIMES",
    eyebrow: "Gymnasium Sandboxes",
    icon: Boxes,
    accent: "text-blue-500",
    headline: "Isolated Execution Sandboxes & Tools",
    description:
      "Spin up and inspect Gymnasium-compatible execution environments powered by Docker containers. Provides Python runtimes, shell terminals, and headless browser automation with deterministic state resets.",
    whenToUse:
      "Go here when inspecting sandbox containers, testing custom tools, reviewing state schemas, or configuring policy engine guards.",
    keyFeatures: [
      "Docker isolation (Python, CLI, Headless Browser)",
      "Gymnasium standard reset() & step() APIs",
      "PolicyEngine security & observation filtering",
      "Deterministic state rollback & TTL management",
    ],
    ctaLabel: "Manage Environments",
  },
];

const PIPELINE_STEPS = [
  {
    step: "01",
    label: "Define Environment",
    target: "/environments",
    desc: "Wrap enterprise tasks or coding tools into Gymnasium environments with deterministic reward checks.",
    badge: "SANDBOX",
  },
  {
    step: "02",
    label: "Synthesize & Verify",
    target: "/generator",
    desc: "Generate task taxonomies. Only tasks that pass execution k times and cross-judge review are admitted.",
    badge: "PASS^K FILTER",
  },
  {
    step: "03",
    label: "Train & Align",
    target: "/training",
    desc: "Optimize agent policies using GRPO, PPO, or DPO backed by verifiable rewards in online or offline modes.",
    badge: "RLVR & SFT",
  },
  {
    step: "04",
    label: "Benchmark & Refine",
    target: "/benchmark",
    desc: "Evaluate rollouts in parallel, isolate failure modes, and feed hard edge cases back into the generator.",
    badge: "EVALUATION",
  },
];

export default function LandingPage() {
  const [activeTab, setActiveTab] = useState<string>("all");

  const filteredGuides =
    activeTab === "all"
      ? ROUTE_GUIDES
      : ROUTE_GUIDES.filter((guide) => guide.id === activeTab);

  return (
    <div className="space-y-12 pb-16 app-width">
      {/* 1. HERO SECTION */}
      <section className="landing-hero mt-4">
        <div className="relative z-10 max-w-4xl space-y-6">
          <div className="flex flex-wrap items-center gap-3">
            <span className="font-mono text-[0.62rem] font-semibold uppercase tracking-[0.24em] text-primary">
              FORGE // RLVR & POST-TRAINING PLATFORM
            </span>
            <span className="inline-flex items-center gap-1.5 border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 font-mono text-[0.56rem] font-medium uppercase tracking-[0.16em] text-emerald-600 dark:text-emerald-400">
              <span className="size-1.5 rounded-full bg-emerald-500 animate-pulse" />
              SYSTEM ACTIVE
            </span>
          </div>

          <h1 className="text-[clamp(2.4rem,6vw,5.2rem)] font-semibold leading-[0.88] tracking-[-0.07em]">
            WHERE WORK BECOMES<br />
            <em className="not-italic text-primary">VERIFIABLE INTELLIGENCE.</em>
          </h1>

          <p className="max-w-2xl text-xs leading-5 text-muted-foreground sm:text-sm sm:leading-6">
            Forge is the complete platform for post-training reasoning models and autonomous agents.
            Synthesize noise-free datasets with automated <span className="font-mono font-medium text-foreground">pass^k</span> code execution,
            train policies using <span className="font-mono font-medium text-foreground">RLVR</span> and preference alignment,
            and benchmark rollouts inside hardened <span className="font-mono font-medium text-foreground">Gymnasium</span> sandboxes.
          </p>
        </div>
      </section>

      {/* 2. NAVIGATION DIRECTORY */}
      <section className="space-y-6">
        <div className="flex flex-col justify-between gap-4 border-b border-foreground/25 pb-4 md:flex-row md:items-end">
          <div>
            <div className="flex items-center gap-2 font-mono text-[0.62rem] font-semibold uppercase tracking-[0.2em] text-primary">
              <Compass size={14} />
              <span>NAVIGATION DIRECTORY</span>
            </div>
          </div>

          {/* Filter Pills */}
          <div className="flex flex-wrap items-center gap-1.5 font-mono text-[0.6rem] uppercase tracking-wider">
            <button
              onClick={() => setActiveTab("all")}
              className={`border px-3 py-1.5 transition-all ${
                activeTab === "all"
                  ? "border-primary bg-primary text-primary-foreground font-semibold"
                  : "border-foreground/20 hover:border-foreground/40 bg-card"
              }`}
            >
              All Routes (4)
            </button>
            {ROUTE_GUIDES.map((guide) => (
              <button
                key={guide.id}
                onClick={() => setActiveTab(guide.id)}
                className={`border px-3 py-1.5 transition-all ${
                  activeTab === guide.id
                    ? "border-primary bg-primary text-primary-foreground font-semibold"
                    : "border-foreground/20 hover:border-foreground/40 bg-card"
                }`}
              >
                {guide.title}
              </button>
            ))}
          </div>
        </div>

        {/* Directory Grid */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {filteredGuides.map((guide) => {
            const Icon = guide.icon;
            return (
              <div
                key={guide.id}
                className="landing-card group"
              >
                <div className="space-y-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-center gap-3">
                      <div className="grid size-10 place-items-center border border-foreground/20 bg-background/80 text-foreground group-hover:border-primary group-hover:text-primary transition-colors">
                        <Icon size={18} />
                      </div>
                      <div>
                        <span className="font-mono text-[0.55rem] font-semibold uppercase tracking-[0.18em] text-primary">
                          {guide.badge}
                        </span>
                        <h3 className="text-lg font-semibold tracking-tight text-foreground">
                          {guide.title}
                        </h3>
                      </div>
                    </div>
                    <span className="font-mono text-[0.58rem] uppercase tracking-wider text-muted-foreground border border-border/80 px-2 py-0.5">
                      {guide.route}
                    </span>
                  </div>

                  <div>
                    <h4 className="font-medium text-sm text-foreground mb-1">
                      {guide.headline}
                    </h4>
                    <p className="text-xs leading-relaxed text-muted-foreground">
                      {guide.description}
                    </p>
                  </div>

                  {/* When to use callout */}
                  <div className="border-l-2 border-primary/60 bg-muted/30 p-2.5 text-[0.72rem] leading-normal text-foreground/90">
                    <strong className="font-mono uppercase text-[0.6rem] text-primary block mb-0.5">
                      When to use:
                    </strong>
                    {guide.whenToUse}
                  </div>

                  {/* Key capabilities list */}
                  <div className="space-y-1.5 pt-1">
                    <span className="block font-mono text-[0.55rem] uppercase tracking-[0.16em] text-muted-foreground">
                      Core Capabilities:
                    </span>
                    <ul className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 text-xs text-foreground/80">
                      {guide.keyFeatures.map((feat, i) => (
                        <li key={i} className="flex items-center gap-1.5 text-[0.7rem]">
                          <CheckCircle2 size={12} className="text-emerald-500 shrink-0" />
                          <span className="truncate">{feat}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className="mt-6 flex items-center justify-end border-t border-border/60 pt-4">
                  <Link
                    href={guide.route}
                    className="inline-flex items-center gap-2 border border-foreground/30 bg-foreground px-4 py-2 font-mono text-[0.65rem] font-semibold uppercase tracking-[0.12em] text-background hover:bg-primary hover:border-primary transition-all"
                  >
                    <span>{guide.ctaLabel}</span>
                    <ArrowRight size={12} />
                  </Link>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* 3. THE FORGE FLYWHEEL (WORKFLOW OVERVIEW) */}
      <section className="space-y-6 border-t border-foreground/25 pt-10">
        <div>
          <div className="flex items-center gap-2 font-mono text-[0.62rem] font-semibold uppercase tracking-[0.2em] text-primary">
            <Workflow size={14} />
            <span>THE FORGE PIPELINE</span>
          </div>
          <h2 className="mt-1 text-2xl font-semibold tracking-[-0.03em] uppercase">
            How Forge Closes the Learning Loop
          </h2>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
          {PIPELINE_STEPS.map((s) => (
            <div key={s.step} className="landing-pipeline-step flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xl font-bold text-primary">
                    {s.step}
                  </span>
                  <span className="border border-foreground/20 px-2 py-0.5 font-mono text-[0.52rem] uppercase tracking-wider text-muted-foreground">
                    {s.badge}
                  </span>
                </div>
                <h3 className="mt-3 font-semibold text-sm text-foreground">
                  {s.label}
                </h3>
                <p className="mt-2 text-xs leading-5 text-muted-foreground">
                  {s.desc}
                </p>
              </div>

              <div className="mt-5 border-t border-border/60 pt-3">
                <Link
                  href={s.target}
                  className="inline-flex items-center gap-1.5 font-mono text-[0.62rem] uppercase tracking-wider text-primary hover:underline"
                >
                  <span>Go to stage</span>
                  <ArrowRight size={11} />
                </Link>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 4. KEY ARCHITECTURAL PILLARS */}
      <section className="space-y-6 border-t border-foreground/25 pt-10">
        <div>
          <div className="flex items-center gap-2 font-mono text-[0.62rem] font-semibold uppercase tracking-[0.2em] text-primary">
            <Layers size={14} />
            <span>CORE ARCHITECTURE</span>
          </div>
          <h2 className="mt-1 text-2xl font-semibold tracking-[-0.03em] uppercase">
            Built for Verifiable Reasoning
          </h2>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          <div className="border border-foreground/20 bg-card p-6 space-y-3">
            <div className="grid size-9 place-items-center border border-foreground/20 bg-background text-primary">
              <Zap size={16} />
            </div>
            <h3 className="font-semibold text-base">Execution Over Vibes (RLVR)</h3>
            <p className="text-xs leading-relaxed text-muted-foreground">
              Traditional LLM evaluation relies on subjective LLM-as-a-judge scorers that hallucinate and suffer from sycophancy.
              Forge validates tasks using executable unit tests, AST parsers, and sandbox outputs that deliver deterministic 0 or 1 ground truth.
            </p>
          </div>

          <div className="border border-foreground/20 bg-card p-6 space-y-3">
            <div className="grid size-9 place-items-center border border-foreground/20 bg-background text-primary">
              <RotateCcw size={16} />
            </div>
            <h3 className="font-semibold text-base">Online & Offline Training</h3>
            <p className="text-xs leading-relaxed text-muted-foreground">
              Forge unifies both training philosophies. Run <strong>Online Mode</strong> for self-play policy rollouts where the generator model matches the actor, or <strong>Offline Mode</strong> for aligning models on verified cross-family datasets.
            </p>
          </div>

          <div className="border border-foreground/20 bg-card p-6 space-y-3">
            <div className="grid size-9 place-items-center border border-foreground/20 bg-background text-primary">
              <ShieldCheck size={16} />
            </div>
            <h3 className="font-semibold text-base">Hardened Sandbox Isolation</h3>
            <p className="text-xs leading-relaxed text-muted-foreground">
              Run arbitrary agent code without risking your host machine. Forge provides ephemeral Docker containers with observation filtering, PII redaction, and strict PolicyEngine boundaries.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
