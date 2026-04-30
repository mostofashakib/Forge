import RolloutLauncher from "@/components/RolloutLauncher";

export default function RolloutsPage() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Rollouts</h1>
        <p className="text-muted-foreground text-sm mt-1.5">
          Launch parallel episode rollouts against any compiled environment.
        </p>
      </div>
      <RolloutLauncher />
    </div>
  );
}
