import type { Metadata } from "next";
import localFont from "next/font/local";
import AppHeader from "@/components/AppHeader";
import "./globals.css";

// Font files ship with the app, so the build never fetches fonts over the network.
const plexSans = localFont({
  src: [
    { path: "./fonts/ibm-plex-sans-latin-300-normal.woff2", weight: "300", style: "normal" },
    { path: "./fonts/ibm-plex-sans-latin-400-normal.woff2", weight: "400", style: "normal" },
    { path: "./fonts/ibm-plex-sans-latin-500-normal.woff2", weight: "500", style: "normal" },
    { path: "./fonts/ibm-plex-sans-latin-600-normal.woff2", weight: "600", style: "normal" },
  ],
});

const plexMono = localFont({
  src: [
    { path: "./fonts/ibm-plex-mono-latin-400-normal.woff2", weight: "400", style: "normal" },
    { path: "./fonts/ibm-plex-mono-latin-500-normal.woff2", weight: "500", style: "normal" },
  ],
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

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
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

          <main className="app-main">{children}</main>

          <footer className="app-footer">
            <p>
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
