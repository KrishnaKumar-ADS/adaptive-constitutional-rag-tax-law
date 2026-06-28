export function CitationCard({ citations }: { citations: any[] }) {
  if (!citations || citations.length === 0) return null;

  // We take the first one or aggregate them. Usually there's one verdict object.
  const verdict = citations[0]?.verdict || "Valid";
  
  let color = "bg-gray-50 border-gray-200";
  let icon = "📋";
  let badge = "bg-gray-100 text-gray-800";
  
  if (verdict === "Valid") {
    color = "bg-green-50 border-green-200";
    icon = "✅";
    badge = "bg-green-100 text-green-800";
  } else if (verdict === "Invalid") {
    color = "bg-red-50 border-red-200";
    icon = "❌";
    badge = "bg-red-100 text-red-800";
  } else if (verdict === "Partially Supported") {
    color = "bg-yellow-50 border-yellow-200";
    icon = "⚠️";
    badge = "bg-yellow-100 text-yellow-800";
  }

  return (
    <div className={`mt-4 rounded-lg border p-4 ${color}`}>
      <div className="flex items-start justify-between">
        <div className="flex items-center">
          <span className="text-xl mr-2">{icon}</span>
          <h4 className="text-sm font-semibold text-gray-900">Zero-LLM Verification</h4>
        </div>
        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${badge}`}>
          {verdict}
        </span>
      </div>
      
      {citations[0]?.reasoning && (
        <p className="mt-2 text-xs text-gray-600">
          <span className="font-semibold text-gray-700">Reasoning:</span> {citations[0].reasoning}
        </p>
      )}
      
      {citations[0]?.invalid_citations?.length > 0 && (
        <div className="mt-2 text-xs text-red-600">
          <span className="font-semibold">Hallucinated Sections:</span> 
          {citations[0].invalid_citations.join(", ")}
        </div>
      )}
    </div>
  );
}
