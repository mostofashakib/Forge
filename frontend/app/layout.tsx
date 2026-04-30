import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Forge",
  description: "Forge",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-background">
          <header className="border-b px-6 py-3 flex items-center gap-3">
            <span className="font-semibold text-lg">Forge</span>

          </header>
          <main className="container mx-auto px-6 py-8 max-w-4xl">{children}</main>
        </div>
      </body>
    </html>
  );
}
