export default function BenchmarkEvalPage() {
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Zero-shot Eval</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Evaluate a fine-tuned checkpoint against WebArena or WorkArena without any in-distribution training.
        </p>
      </div>

      <div className="border border-amber-200 bg-amber-50 rounded-lg p-6 space-y-4">
        <div className="flex items-start gap-3">
          <span className="text-amber-500 text-lg leading-none">⚠</span>
          <div>
            <p className="text-sm font-semibold text-amber-800">Not yet available — eval harness required</p>
            <p className="text-sm text-amber-700 mt-1">
              Zero-shot evaluation requires integrating an external harness such as WebArena or WorkArena.
            </p>
          </div>
        </div>

        <div>
          <p className="text-xs font-medium text-amber-800 mb-1.5">Implement the evaluation stub:</p>
          <pre className="bg-amber-100 rounded-md px-3 py-2 text-xs font-mono text-amber-900 overflow-x-auto">{`forge/benchmark/_eval.py → evaluate_on_suite()`}</pre>
        </div>

        <div>
          <p className="text-xs font-medium text-amber-800 mb-1.5">Once implemented, run via CLI:</p>
          <pre className="bg-amber-100 rounded-md px-3 py-2 text-xs font-mono text-amber-900 overflow-x-auto">{`forge benchmark eval --checkpoint ./benchmark_results/forge_ft --suite webArena`}</pre>
        </div>
      </div>
    </div>
  );
}
