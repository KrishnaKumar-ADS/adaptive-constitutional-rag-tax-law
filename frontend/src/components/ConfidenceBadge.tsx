export function ConfidenceBadge({ uncertainty, tier }: { uncertainty: number; tier: string }) {
  let color = "bg-green-100 text-green-800 border-green-200";
  let label = "High Confidence";
  
  if (tier === "high" || uncertainty >= 0.7) {
    color = "bg-red-100 text-red-800 border-red-200";
    label = "High Uncertainty / Abstain";
  } else if (tier === "medium" || uncertainty >= 0.2) {
    color = "bg-yellow-100 text-yellow-800 border-yellow-200";
    label = "Medium Uncertainty";
  }

  return (
    <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${color}`}>
      <span className="mr-1">{label}</span>
      <span className="font-mono text-[10px] opacity-75">
        (U={uncertainty.toFixed(2)})
      </span>
    </div>
  );
}
