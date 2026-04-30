"use client";
import { useEffect, useRef, useState } from "react";
import { wsBase } from "@/lib/api";

interface FeedEvent {
  id?: string;
  timestamp?: string;
  actor?: string;
  action_type?: string;
  payload?: string;
  state_before?: string;
  state_after?: string;
}

interface Props {
  envName: string;
}

export function SandboxEventFeed({ envName }: Props) {
  const [events, setEvents] = useState<FeedEvent[]>([]);
  const [filterActor, setFilterActor] = useState("");
  const [filterAction, setFilterAction] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ws = new WebSocket(`${wsBase()}/api/sandbox/ws/feed/${envName}`);
    ws.onmessage = (e) => {
      const event: FeedEvent = JSON.parse(e.data);
      setEvents((prev) => [...prev.slice(-999), event]);
    };
    return () => ws.close();
  }, [envName]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  const filtered = events.filter((e) => {
    if (filterActor && e.actor !== filterActor) return false;
    if (filterAction && !e.action_type?.includes(filterAction)) return false;
    return true;
  });

  return (
    <div className="flex flex-col h-full text-xs font-mono">
      <div className="flex gap-1 p-2 border-b bg-gray-50 flex-shrink-0">
        <input
          placeholder="actor..."
          className="border rounded px-2 py-1 flex-1 text-xs"
          value={filterActor}
          onChange={(e) => setFilterActor(e.target.value)}
        />
        <input
          placeholder="action..."
          className="border rounded px-2 py-1 flex-1 text-xs"
          value={filterAction}
          onChange={(e) => setFilterAction(e.target.value)}
        />
      </div>
      <div className="flex-1 overflow-auto">
        {filtered.map((e, i) => (
          <EventRow key={i} event={e} envName={envName} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function EventRow({ event, envName }: { event: FeedEvent; envName: string }) {
  const [expanded, setExpanded] = useState(false);

  async function replayFromHere() {
    if (!event.state_before) return;
    await fetch(`/api/proxy/${envName}/forge/restore-state`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: event.state_before,
    });
  }

  return (
    <div className="border-b px-2 py-1 hover:bg-gray-50">
      <div
        className="flex gap-2 cursor-pointer items-center"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="text-gray-400 w-20 flex-shrink-0">
          {event.timestamp?.slice(11, 19) ?? "—"}
        </span>
        <span
          className={`w-12 flex-shrink-0 font-semibold ${
            event.actor === "user" ? "text-blue-600" : "text-green-600"
          }`}
        >
          {event.actor ?? "—"}
        </span>
        <span className="flex-1 truncate">{event.action_type ?? "—"}</span>
        <button
          onClick={(ev) => { ev.stopPropagation(); replayFromHere(); }}
          className="text-gray-400 hover:text-blue-500 flex-shrink-0 text-xs"
          title="Replay from this state"
        >
          ⏮
        </button>
      </div>
      {expanded && (
        <pre className="mt-1 p-2 bg-gray-100 rounded overflow-auto max-h-48 text-xs">
          {JSON.stringify(
            {
              before: JSON.parse(event.state_before ?? "{}"),
              after: JSON.parse(event.state_after ?? "{}"),
            },
            null,
            2
          )}
        </pre>
      )}
    </div>
  );
}
