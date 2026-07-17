"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Boxes, FlaskConical } from "lucide-react";

const NAV_ITEMS = [
  { label: "Environments", href: "/environments", icon: Boxes },
  { label: "Benchmark", href: "/benchmark", icon: FlaskConical },
];

function isActivePath(pathname: string, href: string): boolean {
  return pathname === href || pathname.startsWith(`${href}/`);
}

export default function AppHeader() {
  const pathname = usePathname();

  return (
    <header className="app-header">
      <div className="app-header__rail">
        <Link href="/environments" className="forge-mark group" aria-label="Forge home">
          <span className="forge-mark__icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none">
              <path d="M12 2.5 21 7v10l-9 4.5L3 17V7l9-4.5Z" stroke="currentColor" strokeWidth="1.6" />
              <path d="m8 9.25 4-2 4 2v5.5l-4 2-4-2v-5.5Z" fill="currentColor" />
            </svg>
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
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

      </div>
    </header>
  );
}
