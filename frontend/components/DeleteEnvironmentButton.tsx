"use client";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { API_BASE } from "@/lib/api";

interface Props {
  envName: string;
  hasSandbox: boolean;
  className?: string;
  icon?: boolean;
}

export function DeleteEnvironmentButton({ envName, hasSandbox, className, icon }: Props) {
  const router = useRouter();
  const [deleting, setDeleting] = useState(false);

  async function handleDelete(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm(`Delete environment "${envName}"? This cannot be undone.`)) return;
    setDeleting(true);
    try {
      const url = hasSandbox
        ? `${API_BASE}/api/sandbox/${envName}`
        : `${API_BASE}/api/envs/${envName}`;
      await fetch(url, { method: "DELETE" });
      router.push("/environments");
      router.refresh();
    } finally {
      setDeleting(false);
    }
  }

  const defaultClass = icon
    ? "p-1 text-muted-foreground hover:text-red-600 hover:bg-red-50 rounded transition-colors disabled:opacity-50"
    : "px-3 py-1.5 border border-red-200 text-red-600 rounded-md text-sm hover:bg-red-50 transition-colors disabled:opacity-50";

  return (
    <button
      onClick={handleDelete}
      disabled={deleting}
      className={className ?? defaultClass}
      title={`Delete ${envName}`}
    >
      {icon ? (
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M2 3.5h10M5 3.5V2.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v1M11.5 3.5l-.7 8a1 1 0 0 1-1 .9H4.2a1 1 0 0 1-1-.9l-.7-8" />
          <path d="M5.5 6.5v3M8.5 6.5v3" />
        </svg>
      ) : deleting ? "Deleting…" : "Delete"}
    </button>
  );
}
