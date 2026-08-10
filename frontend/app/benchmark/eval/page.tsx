export default function BenchmarkEvalPage() {
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Held-out Eval</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Measure policy generalization on internal environments excluded from training.
        </p>
      </div>

      <div className="space-y-4 rounded-lg border border-emerald-200 bg-emerald-50 p-6">
        <div>
          <p className="text-sm font-semibold text-emerald-900">Internal harness ready</p>
          <p className="mt-1 text-sm text-emerald-800">
            The experiment YAML defines disjoint train and held-out environments. Eval rejects
            checkpoints trained with a different split and records the headline metrics under runs/.
          </p>
        </div>

        <div>
          <p className="mb-1.5 text-xs font-medium text-emerald-900">Run via CLI:</p>
          <pre className="overflow-x-auto rounded-md bg-emerald-100 px-3 py-2 font-mono text-xs text-emerald-950">{`forge benchmark eval \\\n+  --checkpoint ./policy_checkpoint \\\n+  --experiment experiments/internal_heldout.yaml`}</pre>
        </div>
      </div>
    </div>
  );
}
