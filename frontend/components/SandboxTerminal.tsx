"use client";
import { useEffect, useRef } from "react";
import { wsBase } from "@/lib/api";

interface Props {
  envName: string;
}

export function SandboxTerminal({ envName }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    let term: import("@xterm/xterm").Terminal;
    let ws: WebSocket;

    (async () => {
      const { Terminal } = await import("@xterm/xterm");
      await import("@xterm/xterm/css/xterm.css");

      term = new Terminal({
        cursorBlink: true,
        fontFamily: "monospace",
        fontSize: 13,
        theme: { background: "#1a1a1a" },
      });

      if (containerRef.current) {
        term.open(containerRef.current);
      }

      ws = new WebSocket(`${wsBase()}/api/sandbox/ws/exec/${envName}`);
      ws.onopen = () => term.writeln("\x1b[32mConnected to container shell.\x1b[0m");
      ws.onmessage = (e) => term.write(typeof e.data === "string" ? e.data : "");
      ws.onclose = () => term.writeln("\r\n\x1b[31mConnection closed.\x1b[0m");
      ws.onerror = () => term.writeln("\r\n\x1b[31mConnection error.\x1b[0m");

      term.onData((data) => {
        if (ws.readyState === WebSocket.OPEN) ws.send(data);
      });
    })();

    return () => {
      ws?.close();
      term?.dispose();
    };
  }, [envName]);

  return <div ref={containerRef} className="h-full w-full bg-[#1a1a1a]" />;
}
