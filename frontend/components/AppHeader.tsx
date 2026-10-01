"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import ForgeMark from "@/components/ForgeMark";
import { Boxes, Cpu, FlaskConical, Settings, Sparkles } from "lucide-react";

const NAV_ITEMS = [
  { label: "Generator", href: "/generator", icon: Sparkles },
  { label: "Training", href: "/training", icon: Cpu },
  { label: "Benchmark", href: "/benchmark", icon: FlaskConical },
  { label: "Settings", href: "/settings", icon: Settings },
  { label: "Environments", href: "/environments", icon: Boxes },
];

function isActivePath(pathname: string, href: string): boolean {
  return pathname === href || pathname.startsWith(`${href}/`);
}

export default function AppHeader() {
  const pathname = usePathname();

  return (
    <header className="app-header">
      <div className="app-header__rail">
        <Link href="/generator" className="forge-mark group" aria-label="Forge home">
          <span className="forge-mark__icon" aria-hidden="true">
            <ForgeMark />
          </span>
          <span>
            <span className="forge-mark__word">FORGE</span>
          </span>
        </Link>

        <nav className="app-nav" aria-label="Primary navigation">
          {NAV_ITEMS.map((item, index) => {
            const active = isActivePath(pathname, item.href);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={`app-nav__link ${active ? "app-nav__link--active" : ""}`}
              >
                <span className="app-nav__index">0{index + 1}</span>
                <Icon size={15} strokeWidth={1.8} aria-hidden="true" />
                <span className="app-nav__label">{item.label}</span>
              </Link>
            );
          })}
        </nav>

      </div>
    </header>
  );
}
