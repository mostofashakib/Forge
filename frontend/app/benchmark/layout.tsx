"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const SIDEBAR_ITEMS = [
  { href: "/benchmark/run", label: "Run", available: true, badge: undefined },
  {
    href: "/benchmark/report",
    label: "Report",
    available: true,
    badge: undefined,
  },
  {
    href: "/benchmark/transfer",
    label: "Transfer",
    available: false,
    badge: "GPU",
  },
  { href: "/benchmark/eval", label: "Eval", available: false, badge: "soon" },
];

export default function BenchmarkLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <div className="benchmark-shell">
      <aside className="benchmark-rail">
        <div className="benchmark-rail__header">
          <span>Suite</span>
          <strong>
            BENCH
            <br />
            MARK
          </strong>
          <p>Quality Control</p>
        </div>
        <nav className="benchmark-rail__nav" aria-label="Benchmark navigation">
          {SIDEBAR_ITEMS.map((item) => {
            const isActive =
              pathname === item.href || pathname.startsWith(item.href + "/");
            return (
              <Link
                key={item.href}
                href={item.available ? item.href : "#"}
                onClick={(e) => !item.available && e.preventDefault()}
                aria-current={isActive ? "page" : undefined}
                className={`benchmark-rail__link ${isActive ? "benchmark-rail__link--active" : ""} ${
                  !item.available ? "benchmark-rail__link--disabled" : ""
                }`}
              >
                <span className="benchmark-rail__index">
                  0{SIDEBAR_ITEMS.indexOf(item) + 1}
                </span>
                <span>{item.label}</span>
                {item.badge && (
                  <span className="benchmark-rail__badge">{item.badge}</span>
                )}
              </Link>
            );
          })}
        </nav>
      </aside>
      <div className="benchmark-stage">{children}</div>
    </div>
  );
}
