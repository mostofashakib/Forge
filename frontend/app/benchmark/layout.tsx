"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const SIDEBAR_ITEMS = [
  { href: "/benchmark/run",      label: "Run",      available: true,  badge: undefined },
  { href: "/benchmark/report",   label: "Report",   available: true,  badge: undefined },
  { href: "/benchmark/transfer", label: "Transfer", available: false, badge: "GPU"     },
  { href: "/benchmark/eval",     label: "Eval",     available: false, badge: "soon"    },
];

export default function BenchmarkLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="flex gap-8">
      <aside className="w-40 shrink-0 pt-1">
        <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3 px-3">
          Benchmark
        </p>
        <nav className="flex flex-col gap-0.5">
          {SIDEBAR_ITEMS.map((item) => {
            const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
            return (
              <Link
                key={item.href}
                href={item.available ? item.href : "#"}
                onClick={(e) => !item.available && e.preventDefault()}
                className={`px-3 py-2 rounded-lg text-sm transition-colors flex items-center justify-between ${
                  isActive
                    ? "bg-primary/10 text-primary font-medium"
                    : item.available
                    ? "text-muted-foreground hover:text-foreground hover:bg-muted/60"
                    : "text-muted-foreground/40 cursor-not-allowed"
                }`}
              >
                {item.label}
                {item.badge && (
                  <span className="text-[10px] opacity-60 font-normal">{item.badge}</span>
                )}
              </Link>
            );
          })}
        </nav>
      </aside>
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  );
}
