"use client";
import { useEffect } from "react";

interface Props {
  message: string;
  onDismiss: () => void;
  duration?: number;
}

export function Toast({ message, onDismiss, duration = 3000 }: Props) {
  useEffect(() => {
    const t = setTimeout(onDismiss, duration);
    return () => clearTimeout(t);
  }, [onDismiss, duration]);

  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2.5 px-4 py-3 bg-foreground text-background rounded-lg shadow-xl text-sm font-medium whitespace-nowrap animate-in slide-in-from-bottom-4 fade-in duration-200">
      <span className="text-green-400 text-base leading-none">✓</span>
      {message}
    </div>
  );
}
