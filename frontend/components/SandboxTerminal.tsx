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

    let cancelled = false;
    let term: import("@xterm/xterm").Terminal | undefined;
    let ws: WebSocket | undefined;
    let observer: ResizeObserver | undefined;

    (async () => {
      const [{ Terminal }, { FitAddon }] = await Promise.all([
        import("@xterm/xterm"),
        import("@xterm/addon-fit"),
      ]);
      await import("@xterm/xterm/css/xterm.css");

      // Guard: if cleanup already ran while imports were loading, abort.
      if (cancelled || !containerRef.current) return;

      term = new Terminal({
        cursorBlink: true,
        fontFamily: "monospace",
        fontSize: 13,
        theme: { background: "#1a1a1a" },
        scrollback: 1000,
      });

      const fitAddon = new FitAddon();
      term.loadAddon(fitAddon);
      term.open(containerRef.current);
      fitAddon.fit();

      ws = new WebSocket(`${wsBase()}/api/sandbox/ws/exec/${envName}`);

      ws.onopen = () => {
        term!.writeln("\x1b[32mConnected to container shell.\x1b[0m");
        if (ws!.readyState === WebSocket.OPEN) {
          ws!.send(JSON.stringify({ type: "resize", cols: term!.cols, rows: term!.rows }));
        }
      };
      ws.onmessage = (e) => term!.write(typeof e.data === "string" ? e.data : "");
      ws.onclose = () => term!.writeln("\r\n\x1b[31mConnection closed.\x1b[0m");
      ws.onerror = () => term!.writeln("\r\n\x1b[31mConnection error.\x1b[0m");

      term.onData((data) => {
        if (ws!.readyState === WebSocket.OPEN) ws!.send(data);
      });

      term.onResize(({ cols, rows }) => {
        if (ws!.readyState === WebSocket.OPEN) {
          ws!.send(JSON.stringify({ type: "resize", cols, rows }));
        }
      });

      observer = new ResizeObserver(() => fitAddon.fit());
      observer.observe(containerRef.current!);
    })();

    return () => {
      cancelled = true;
      observer?.disconnect();
      ws?.close();
      term?.dispose();
    };
  }, [envName]);

  return <div ref={containerRef} className="h-full w-full bg-[#1a1a1a]" />;
}
