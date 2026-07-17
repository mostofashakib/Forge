import type { Metadata } from "next";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import AppHeader from "@/components/AppHeader";
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
        <div className="min-h-screen flex flex-col">
          <AppHeader />

          <main className="app-main forge-enter">{children}</main>

          <footer className="app-footer">
            <p>
              <span className="app-footer__stamp">FORGE / 2026</span>
              <span className="app-footer__divider" />
              Developed by{" "}
              <a
                href="https://www.mostofashakib.com/"
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium hover:text-foreground transition-colors"
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
