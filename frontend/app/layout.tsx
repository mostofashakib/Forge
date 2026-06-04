import type { Metadata } from "next";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono",
});

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";

const DESCRIPTION =
  "Forge converts enterprise workflow specifications into Gymnasium-compatible reinforcement learning environments. Extract structure with LLMs, compile to runnable Python envs, run parallel rollouts, and export training data.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "Forge",
    template: "%s — Forge",
  },
  description: DESCRIPTION,
  applicationName: "Forge",
  keywords: [
    "reinforcement learning",
    "RL environment",
    "Gymnasium",
    "enterprise workflow",
    "LLM extraction",
    "policy engine",
    "training data export",
    "SFT",
    "GRPO",
    "preference pairs",
  ],
  authors: [{ name: "Forge" }],
  robots: {
    index: true,
    follow: true,
    googleBot: { index: true, follow: true },
  },
  openGraph: {
    type: "website",
    siteName: "Forge",
    title: "Forge — Enterprise Workflows to RL Environments",
    description: DESCRIPTION,
    url: SITE_URL,
  },
  twitter: {
    card: "summary",
    title: "Forge — Enterprise Workflows to RL Environments",
    description: DESCRIPTION,
  },
  alternates: {
    canonical: SITE_URL,
  },
};

const JSON_LD = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "Forge",
  applicationCategory: "DeveloperApplication",
  operatingSystem: "Web",
  description: DESCRIPTION,
  offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
  featureList: [
    "LLM-powered entity and action extraction",
    "Jinja2 compiler to Gymnasium-compatible environments",
    "Six built-in verifier types",
    "Decomposed reward engine",
    "Parallel Celery rollout workers",
    "SFT, preference pairs, and GRPO training export",
    "PolicyEngine DSL with sandboxed evaluation",
    "RBAC observation filtering",
    "Network isolation and PII redaction",
    "Audit log and policy violation viewer",
  ],
};

const NAV = [
  { label: "Environments", href: "/environments" },
  { label: "Benchmark", href: "/benchmark" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(JSON_LD) }}
        />
      </head>
      <body className={`${plexSans.className} ${plexMono.variable}`}>
        <div className="min-h-screen bg-background flex flex-col">
          {/* Header */}
          <header className="sticky top-0 z-50 bg-background/80 backdrop-blur-md border-b border-border/50 px-6 h-14 flex items-center justify-between">
            <Link href="/environments/new" className="flex items-center gap-2.5 group">
              <div className="w-7 h-7 rounded-lg bg-primary flex items-center justify-center shadow-sm group-hover:shadow-md group-hover:scale-105 transition-all">
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <path d="M7 1L13 4V10L7 13L1 10V4L7 1Z" stroke="white" strokeWidth="1.5" strokeLinejoin="round" />
                  <path d="M7 5L9 6.5V9L7 10.5L5 9V6.5L7 5Z" fill="white" />
                </svg>
              </div>
              <span className="font-semibold text-sm tracking-widest text-foreground">FORGE</span>
            </Link>

            <nav className="flex items-center gap-1">
              {NAV.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="px-3 py-1.5 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-muted/80 transition-colors"
                >
                  {item.label}
                </Link>
              ))}
            </nav>
          </header>

          <main className="container mx-auto px-6 py-10 max-w-5xl flex-1">{children}</main>

          <footer className="border-t border-border/40 mt-auto py-5 px-6 flex items-center justify-center">
            <p className="text-xs text-muted-foreground/70">
              Developed by{" "}
              <a
                href="https://www.mostofashakib.com/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-muted-foreground font-medium hover:text-foreground transition-colors"
              >
                Mostofa Shakib
              </a>
            </p>
          </footer>
        </div>
      </body>
    </html>
  );
}
