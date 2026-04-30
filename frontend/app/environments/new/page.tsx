import PromptForm from "@/components/PromptForm";

export default function NewEnvironmentPage() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">New Environment</h1>
        <p className="text-muted-foreground mt-1.5 text-sm">
          Describe your workflow and Forge will extract the structure for you to review.
        </p>
      </div>
      <PromptForm />
    </div>
  );
}
