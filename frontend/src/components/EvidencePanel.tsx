export function EvidencePanel({ evidence }: { evidence: any[] }) {
  if (!evidence || evidence.length === 0) return null;

  return (
    <div className="mt-6 border-t pt-4">
      <h3 className="text-sm font-semibold text-gray-900 mb-3">Retrieved Evidence ({evidence.length})</h3>
      <div className="space-y-3">
        {evidence.map((ev, i) => (
          <div key={i} className="bg-white border rounded-md p-3 shadow-sm text-xs">
            <div className="flex justify-between items-center mb-1">
              <span className="font-mono font-bold text-indigo-700">{ev.citation_id}</span>
              <span className="px-2 py-0.5 bg-gray-100 rounded text-[10px] font-medium text-gray-600 uppercase tracking-wider">
                {ev.source_type}
              </span>
            </div>
            {ev.score !== undefined && (
              <div className="text-[10px] text-gray-500 mb-2">Relevance: {ev.score.toFixed(3)}</div>
            )}
            <p className="text-gray-700 leading-relaxed truncate">{ev.text || "Context not included in summary..."}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
