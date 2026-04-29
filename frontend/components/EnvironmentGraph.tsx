"use client";

import { useEffect, useRef, useState } from "react";
import { ReactFlow, Background, Controls, useNodesState, useEdgesState } from "@xyflow/react";
import type { Node, Edge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";

interface CompilerInput {
  project_name: string;
  domain: string;
  entities: Array<{ name: string; fields: Array<{ name: string }> }>;
  actions: Array<{ name: string }>;
  tasks: Array<{ name: string; success_conditions: unknown[] }>;
  policies?: Array<{ name: string }>;
}

interface StepEvent {
  type: "step" | "complete" | "error";
  step_index?: number;
  action?: { type: string };
  diff?: { changed?: Record<string, { before: unknown; after: unknown }> };
  verifier_results?: Array<{ verifier_id: string; passed: boolean; checks: Array<{ name: string; passed: boolean }> }>;
  events?: Array<{ type: string; [key: string]: unknown }>;
  total_reward?: number;
  passed?: boolean;
  total_steps?: number;
}

function buildInitialGraph(ci: CompilerInput): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  ci.entities.forEach((entity, i) => {
    nodes.push({
      id: `entity_${entity.name}`,
      position: { x: 50, y: i * 120 + 50 },
      data: { label: entity.name, fieldValues: {} as Record<string, unknown> },
      style: {
        border: "1px solid #38bdf8",
        background: "#0f2744",
        color: "#f8fafc",
        borderRadius: 8,
        padding: "8px 12px",
        minWidth: 120,
      },
    });
  });

  ci.actions.forEach((action, i) => {
    nodes.push({
      id: `action_${action.name}`,
      position: { x: 280, y: i * 80 + 50 },
      data: { label: action.name, active: false },
      style: {
        border: "1px solid #334155",
        background: "#1e293b",
        color: "#94a3b8",
        borderRadius: 6,
        padding: "6px 10px",
        minWidth: 120,
      },
    });
  });

  ci.tasks.forEach((task, i) => {
    nodes.push({
      id: `task_${task.name}`,
      position: { x: 500, y: i * 100 + 50 },
      data: {
        label: task.name,
        checksPassed: 0,
        checksTotal: (task.success_conditions as unknown[]).length,
      },
      style: {
        border: "1px solid #34d399",
        background: "#0d2318",
        color: "#34d399",
        borderRadius: 8,
        padding: "8px 12px",
        minWidth: 140,
      },
    });
  });

  (ci.policies ?? []).forEach((policy, i) => {
    nodes.push({
      id: `policy_${policy.name}`,
      position: { x: 500, y: ci.tasks.length * 100 + i * 80 + 50 },
      data: { label: policy.name, violated: false },
      style: {
        border: "1px solid #334155",
        background: "#1e293b",
        color: "#94a3b8",
        borderRadius: 8,
        padding: "8px 12px",
        minWidth: 140,
      },
    });
  });

  ci.entities.forEach((entity) => {
    ci.actions.forEach((action) => {
      edges.push({
        id: `e_${entity.name}_${action.name}`,
        source: `entity_${entity.name}`,
        target: `action_${action.name}`,
        style: { stroke: "#334155" },
      });
    });
  });

  ci.actions.forEach((action) => {
    ci.tasks.forEach((task) => {
      edges.push({
        id: `e_${action.name}_${task.name}`,
        source: `action_${action.name}`,
        target: `task_${task.name}`,
        style: { stroke: "#334155" },
      });
    });
  });

  return { nodes, edges };
}

interface EnvironmentGraphProps {
  envName: string;
  episodeId?: string;
  compilerInput: CompilerInput | null;
}

export default function EnvironmentGraph({
  envName,
  episodeId,
  compilerInput,
}: EnvironmentGraphProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [stepLabel, setStepLabel] = useState<string>("");
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!compilerInput) return;
    const { nodes: n, edges: e } = buildInitialGraph(compilerInput);
    setNodes(n);
    setEdges(e);
  }, [compilerInput, setNodes, setEdges]);

  useEffect(() => {
    if (!episodeId) return;
    const ws = new WebSocket(`ws://localhost:8000/api/episodes/${episodeId}/stream`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data: StepEvent = JSON.parse(event.data as string);

      if (data.type === "step") {
        const stepIndex = data.step_index ?? 0;
        const actionType = data.action?.type ?? "";
        setStepLabel(`step ${stepIndex}`);

        setNodes((nds) =>
          nds.map((n) => {
            if (n.id === `action_${actionType}`) {
              return {
                ...n,
                data: { ...n.data, active: true },
                style: {
                  ...n.style,
                  border: "2px solid #f59e0b",
                  background: "#1c1400",
                  color: "#fbbf24",
                },
              };
            }
            if (n.id.startsWith("action_") && n.id !== `action_${actionType}`) {
              return {
                ...n,
                data: { ...n.data, active: false },
                style: {
                  border: "1px solid #334155",
                  background: "#1e293b",
                  color: "#94a3b8",
                  borderRadius: 6,
                  padding: "6px 10px",
                  minWidth: 120,
                },
              };
            }
            return n;
          })
        );

        const changed = data.diff?.changed ?? {};
        setNodes((nds) =>
          nds.map((n) => {
            if (!n.id.startsWith("entity_")) return n;
            const entityName = n.id.replace("entity_", "");
            const updates: Record<string, unknown> = { ...(n.data.fieldValues as Record<string, unknown>) };
            Object.entries(changed).forEach(([key, val]) => {
              if (key.startsWith(`${entityName}.`)) {
                const field = key.split(".").slice(2).join(".");
                updates[field] = (val as { after: unknown }).after;
              }
            });
            return { ...n, data: { ...n.data, fieldValues: updates } };
          })
        );

        const events = data.events ?? [];
        const violatedPolicies = events
          .filter((e) => e.type === "policy_violation")
          .map((e) => e.policy_id as string);
        if (violatedPolicies.length > 0) {
          setNodes((nds) =>
            nds.map((n) => {
              if (!n.id.startsWith("policy_")) return n;
              const policyId = n.id.replace("policy_", "");
              if (violatedPolicies.includes(policyId)) {
                return {
                  ...n,
                  data: { ...n.data, violated: true },
                  style: {
                    border: "1px solid #f87171",
                    background: "#1f0d0d",
                    color: "#f87171",
                    borderRadius: 8,
                    padding: "8px 12px",
                    minWidth: 140,
                  },
                };
              }
              return n;
            })
          );
        }

        setTimeout(() => {
          setNodes((nds) =>
            nds.map((n) => {
              if (n.id === `action_${actionType}`) {
                return {
                  ...n,
                  data: { ...n.data, active: false },
                  style: {
                    border: "1px solid #334155",
                    background: "#1e293b",
                    color: "#94a3b8",
                    borderRadius: 6,
                    padding: "6px 10px",
                    minWidth: 120,
                  },
                };
              }
              return n;
            })
          );
        }, 1500);
      }

      if (data.type === "complete") {
        setStepLabel(`done — ${data.total_steps} steps, r=${data.total_reward?.toFixed(2)}`);
        ws.close();
      }
    };

    return () => {
      ws.close();
    };
  }, [episodeId, setNodes]);

  if (!compilerInput) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
        No compiler input available for this environment.
      </div>
    );
  }

  return (
    <div className="relative w-full h-[600px] rounded-lg border bg-card overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
      >
        <Background />
        <Controls />
      </ReactFlow>
      {stepLabel && (
        <div className="absolute bottom-2 right-2 text-xs text-muted-foreground bg-background/80 px-2 py-1 rounded">
          {episodeId ? `live · ${stepLabel}` : "static"}
        </div>
      )}
    </div>
  );
}
