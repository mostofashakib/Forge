/** Schedule each request after the previous one settles, and abort on cleanup. */
export function startPolling(
  task: (signal: AbortSignal) => Promise<void>,
  intervalMs: number,
  onError: (cause: unknown) => void,
): () => void {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout>;

  async function run() {
    try {
      await task(controller.signal);
    } catch (cause) {
      if (!controller.signal.aborted) onError(cause);
    } finally {
      if (!controller.signal.aborted) timer = setTimeout(run, intervalMs);
    }
  }

  timer = setTimeout(run, 0);
  return () => {
    controller.abort();
    clearTimeout(timer);
  };
}
