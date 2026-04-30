import type { Metadata } from "next";
import { Outfit } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const outfit = Outfit({ subsets: ["latin"], weight: ["400", "500", "600", "700"] });

export const metadata: Metadata = {
  title: "Forge",
  description: "Forge",
};

const NAV = [
  { label: "Environments", href: "/environments/new" },
  { label: "Dashboard", href: "/dashboard" },
  { label: "Rollouts", href: "/rollouts" },
  { label: "Violations", href: "/violations" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={outfit.className}>
        <div className="min-h-screen bg-background">
          <header className="border-b border-border/60 px-6 h-14 flex items-center justify-between">
            <Link href="/environments/new" className="flex items-center gap-2.5 group">
              <div className="w-6 h-6 rounded bg-primary flex items-center justify-center">
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <path d="M7 1L13 4V10L7 13L1 10V4L7 1Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" className="text-primary-foreground" />
                  <path d="M7 5L9 6.5V9L7 10.5L5 9V6.5L7 5Z" fill="currentColor" className="text-primary-foreground" />
                </svg>
              </div>
              <span className="font-semibold text-sm tracking-wide text-foreground">FORGE</span>
            </Link>

            <nav className="flex items-center gap-1">
              {NAV.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="px-3 py-1.5 rounded text-sm text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                >
                  {item.label}
                </Link>
              ))}
            </nav>
          </header>

          <main className="container mx-auto px-6 py-8 max-w-5xl">{children}</main>
        </div>
      </body>
    </html>
  );
}
