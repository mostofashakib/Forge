"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

type CompilerInput = {
  project_name: string;
  domain: string;
  entities: Array<{ name: string; fields: Array<{ name: string; type: string }> }>;
  actions: Array<{ name: string; params: Array<{ name: string; type: string }> }>;
  policies: Array<{ id: string; description: string }>;
  tasks: Array<{ name: string; description: string; success_conditions: Array<{ type: string; expression: string }> }>;
};

export default function CompilerReview({
  jobId,
  initialCompilerInput,
}: {
  jobId: string;
  initialCompilerInput: CompilerInput;
}) {
  const router = useRouter();
  const [ci] = useState(initialCompilerInput);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleApprove() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`http://localhost:8000/api/compile/generate/${jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(ci),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      alert(`Generation ${data.status}! Package at: ${data.output_path ?? "see backend logs"}`);
      router.push("/environments/new");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Compiler Review</h1>
          <p className="text-muted-foreground mt-1">
            Review the extracted structure before generating your environment.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {error && <p className="text-sm text-destructive">{error}</p>}
          <Button onClick={handleApprove} disabled={loading}>
            {loading ? "Generating..." : "Approve & Generate →"}
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            {ci.project_name}
            <Badge variant="outline">{ci.domain}</Badge>
          </CardTitle>
        </CardHeader>
      </Card>

      <Accordion multiple defaultValue={["entities", "actions", "tasks"]}>
        <AccordionItem value="entities">
          <AccordionTrigger>
            Entities <Badge className="ml-2">{ci.entities.length}</Badge>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-3 pt-2">
              {ci.entities.map((entity) => (
                <Card key={entity.name}>
                  <CardContent className="pt-4">
                    <p className="font-medium capitalize">{entity.name}</p>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {entity.fields.map((f) => (
                        <Badge key={f.name} variant="secondary">
                          {f.name}: {f.type}
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="actions">
          <AccordionTrigger>
            Actions <Badge className="ml-2">{ci.actions.length}</Badge>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-3 pt-2">
              {ci.actions.map((action) => (
                <Card key={action.name}>
                  <CardContent className="pt-4">
                    <p className="font-mono text-sm font-medium">{action.name}</p>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {action.params.map((p) => (
                        <Badge key={p.name} variant="secondary">
                          {p.name}: {p.type}
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="tasks">
          <AccordionTrigger>
            Tasks <Badge className="ml-2">{ci.tasks.length}</Badge>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-3 pt-2">
              {ci.tasks.map((task) => (
                <Card key={task.name}>
                  <CardContent className="pt-4">
                    <p className="font-medium">{task.name}</p>
                    <p className="text-sm text-muted-foreground mt-1">{task.description}</p>
                    <div className="mt-2 space-y-1">
                      {task.success_conditions.map((c, i) => (
                        <p key={i} className="text-xs font-mono text-green-700 dark:text-green-400">
                          ✓ {c.expression}
                        </p>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}
