"use client";
import { useEffect, useRef, useState } from "react";
import { wsBase } from "@/lib/api";

interface ActivityEvent {
  type: "command" | "log";
  ts?: string;
  content: string;
}

interface Props {
  envName: string;
}

const TYPE_STYLE: Record<string, string> = {
  command: "text-yellow-400",
  log: "text-blue-400",
};

export function ActivityLog({ envName }: Props) {
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [filter, setFilter] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const autoScroll = useRef(true);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ws = new WebSocket(`${wsBase()}/api/sandbox/ws/activity/${envName}`);
    ws.onmessage = (e) => {
      const ev: ActivityEvent = JSON.parse(e.data);
      setEvents((prev) => [...prev.slice(-1999), ev]);
    };
    return () => ws.close();
  }, [envName]);

  useEffect(() => {
    if (autoScroll.current) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [events]);

  const filtered = filter
    ? events.filter((e) => e.content.toLowerCase().includes(filter.toLowerCase()))
    : events;

  return (
    <div className="flex flex-col h-full font-mono text-xs bg-[#0d1117] text-gray-300">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-800 bg-[#161b22] shrink-0">
        <input
          placeholder="filter logs…"
          className="border border-gray-700 rounded px-2 py-1 flex-1 text-xs bg-[#0d1117] text-gray-300 placeholder-gray-600 focus:outline-none focus:border-gray-500"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <span className="text-gray-600 shrink-0">{events.length} events</span>
        <button
          className="text-gray-600 hover:text-gray-400 shrink-0 text-xs px-1"
          onClick={() => setEvents([])}
          title="Clear"
        >
          ✕
        </button>
      </div>
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto"
        onScroll={() => {
          const el = containerRef.current;
          if (!el) return;
          autoScroll.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
        }}
      >
        {filtered.length === 0 ? (
          <p className="text-gray-600 italic p-3">Waiting for activity…</p>
        ) : (
          <table className="w-full border-collapse">
            <tbody>
              {filtered.map((ev, i) => (
                <tr key={i} className="hover:bg-white/5 border-b border-gray-900">
                  <td className="px-2 py-0.5 text-gray-600 whitespace-nowrap w-20 align-top">
                    {ev.ts?.slice(11, 19) ?? ""}
                  </td>
                  <td className={`px-2 py-0.5 w-16 font-semibold whitespace-nowrap align-top ${TYPE_STYLE[ev.type] ?? "text-gray-400"}`}>
                    {ev.type}
                  </td>
                  <td className="px-2 py-0.5 break-all align-top">
                    {ev.content}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
