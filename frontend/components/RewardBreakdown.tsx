"use client";

interface RewardComponent {
  name: string;
  value: number;
}

interface RewardBreakdownProps {
  components: RewardComponent[];
  total: number;
}

export default function RewardBreakdown({ components, total }: RewardBreakdownProps) {
  return (
    <div className="space-y-1">
      {components.map((c) => (
        <div key={c.name} className="flex justify-between text-sm font-mono">
          <span className="text-muted-foreground">{c.name}:</span>
          <span className={c.value >= 0 ? "text-green-400" : "text-red-400"}>
            {c.value >= 0 ? "+" : ""}
            {c.value.toFixed(3)}
          </span>
        </div>
      ))}
      <div className="border-t pt-1 flex justify-between text-sm font-mono font-semibold">
        <span className="text-muted-foreground">total:</span>
        <span className={total >= 0 ? "text-green-400" : "text-red-400"}>
          {total >= 0 ? "+" : ""}
          {total.toFixed(3)}
        </span>
      </div>
    </div>
  );
}
