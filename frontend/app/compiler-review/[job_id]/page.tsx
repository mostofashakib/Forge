import CompilerReview from "@/components/CompilerReview";

async function getJob(jobId: string) {
  const res = await fetch(`http://localhost:8000/api/compile/${jobId}`, {
    cache: "no-store",
  });
  if (!res.ok) return null;
  return res.json();
}

export default async function CompilerReviewPage({
  params,
}: {
  params: Promise<{ job_id: string }>;
}) {
  const { job_id } = await params;
  const job = await getJob(job_id);

  if (!job || !job.compiler_input) {
    return (
      <div className="text-center py-20">
        <p className="text-muted-foreground">Job not found or still processing.</p>
      </div>
    );
  }

  return (
    <CompilerReview jobId={job_id} initialCompilerInput={job.compiler_input} />
  );
}
