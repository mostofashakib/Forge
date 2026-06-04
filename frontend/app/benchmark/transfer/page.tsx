export default function BenchmarkTransferPage() {
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Transfer Pipeline</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Fine-tune a base model on Forge-collected data and evaluate zero-shot on WebArena / WorkArena.
        </p>
      </div>

      <div className="border border-amber-200 bg-amber-50 rounded-lg p-6 space-y-4">
        <div className="flex items-start gap-3">
          <span className="text-amber-500 text-lg leading-none">⚠</span>
          <div>
            <p className="text-sm font-semibold text-amber-800">Not yet available — GPU node required</p>
            <p className="text-sm text-amber-700 mt-1">
              The transfer pipeline requires a CUDA-capable GPU and the{" "}
              <code className="font-mono text-xs bg-amber-100 px-1 py-0.5 rounded">trl</code>,{" "}
              <code className="font-mono text-xs bg-amber-100 px-1 py-0.5 rounded">transformers</code>, and{" "}
              <code className="font-mono text-xs bg-amber-100 px-1 py-0.5 rounded">datasets</code> packages.
            </p>
          </div>
        </div>

        <div>
          <p className="text-xs font-medium text-amber-800 mb-1.5">Install dependencies on your GPU node:</p>
          <pre className="bg-amber-100 rounded-md px-3 py-2 text-xs font-mono text-amber-900 overflow-x-auto">{`pip install trl transformers datasets`}</pre>
        </div>

        <div>
          <p className="text-xs font-medium text-amber-800 mb-1.5">Implement the fine-tuning stub:</p>
          <pre className="bg-amber-100 rounded-md px-3 py-2 text-xs font-mono text-amber-900 overflow-x-auto">{`forge/benchmark/_fine_tune.py → fine_tune_model()`}</pre>
        </div>

        <div>
          <p className="text-xs font-medium text-amber-800 mb-1.5">Once implemented, run via CLI:</p>
          <pre className="bg-amber-100 rounded-md px-3 py-2 text-xs font-mono text-amber-900 overflow-x-auto">{`forge benchmark transfer --data benchmark_results/data --base-model meta-llama/Llama-3.1-8B`}</pre>
        </div>
      </div>
    </div>
  );
}
