"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";

export default function ConfigEditor({
  envName,
  initialYaml,
}: {
  envName: string;
  initialYaml: string;
}) {
  const [yaml, setYaml] = useState(initialYaml);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSave() {
    setSaving(true);
    setError(null);
    setSaved(false);
    try {
      const res = await fetch(`http://localhost:8000/api/envs/${envName}/config`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ yaml }),
      });
      if (!res.ok) throw new Error(await res.text());
      setSaved(true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Config Editor</h1>
          <p className="text-muted-foreground mt-1">
            Edit <code className="text-xs bg-muted px-1 rounded">custom/config.yaml</code> for{" "}
            <span className="font-medium">{envName}</span>
          </p>
        </div>
        <div className="flex items-center gap-3">
          {saved && <Badge variant="secondary">Saved</Badge>}
          {error && <p className="text-sm text-destructive">{error}</p>}
          <Button onClick={handleSave} disabled={saving}>
            {saving ? "Saving..." : "Save Config"}
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-mono text-muted-foreground">
            custom/config.yaml
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Textarea
            value={yaml}
            onChange={(e) => { setYaml(e.target.value); setSaved(false); }}
            rows={24}
            className="font-mono text-sm"
          />
        </CardContent>
      </Card>
    </div>
  );
}
