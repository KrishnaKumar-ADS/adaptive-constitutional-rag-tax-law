"use client";

import { useState } from "react";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { CitationCard } from "@/components/CitationCard";
import { EvidencePanel } from "@/components/EvidencePanel";

export default function Home() {
  const [query, setQuery] = useState("");
  const [useLocal, setUseLocal] = useState(true);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<any>(null);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError("");
    setResponse(null);

    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: query,
          use_local_model: useLocal
        }),
      });

      let data;
      if (!res.ok) {
        let errorMsg = `API error: ${res.status} ${res.statusText}`;
        try {
          const errData = await res.json();
          errorMsg = `API error (${res.status}): ${JSON.stringify(errData, null, 2)}`;
        } catch (e) {
          try {
            const errText = await res.text();
            if (errText) errorMsg = `API error (${res.status}): ${errText}`;
          } catch (e2) {}
        }
        throw new Error(errorMsg);
      }

      data = await res.json();
      setResponse(data);
    } catch (err: any) {
      console.error("Fetch error details:", err);
      setError(err.message || "Failed to fetch response. Make sure the backend API is running.");
    } finally {
      setLoading(false);
    }
  };

  // Helper to get quality score color
  const getQualityColor = (score: number) => {
    if (score >= 0.8) return "text-emerald-600 bg-emerald-50 border-emerald-200";
    if (score >= 0.5) return "text-amber-600 bg-amber-50 border-amber-200";
    return "text-red-600 bg-red-50 border-red-200";
  };

  const getQualityLabel = (score: number) => {
    if (score >= 0.9) return "Excellent";
    if (score >= 0.7) return "Good";
    if (score >= 0.5) return "Fair";
    if (score >= 0.3) return "Poor";
    return "Very Poor";
  };

  return (
    <main className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8 font-sans">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-10">
          <h1 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
            Adaptive Constitutional RAG
          </h1>
          <p className="mt-3 text-xl text-gray-500">
            Indian Tax Law Legal Assistant
          </p>
        </div>

        {/* Query Box */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8 border border-gray-100">
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
                Legal Query
              </label>
              <textarea
                id="query"
                rows={3}
                className="shadow-sm block w-full focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm border border-gray-300 rounded-md p-3 text-black"
                placeholder="e.g. What is the penalty under Section 271F of the Income Tax Act?"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <input
                  id="useLocal"
                  type="checkbox"
                  checked={useLocal}
                  onChange={(e) => setUseLocal(e.target.checked)}
                  className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                />
                <label htmlFor="useLocal" className="ml-2 block text-sm text-gray-900">
                  Use fine-tuned local model (Qwen3-8B-Tax)
                </label>
              </div>
              <button
                type="submit"
                disabled={loading}
                className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${
                  loading ? "bg-indigo-400" : "bg-indigo-600 hover:bg-indigo-700"
                } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500`}
              >
                {loading ? "Processing..." : "Submit Query"}
              </button>
            </div>
          </form>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-8">
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        {/* Results */}
        {response && (
          <div className="bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden">
            {/* Header / Meta */}
            <div className="bg-gray-50 px-6 py-4 border-b flex justify-between items-center">
              <div>
                <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider block mb-1">
                  Model
                </span>
                <span className="text-sm font-medium text-gray-900">
                  {response.model_name}
                </span>
              </div>
              <div className="flex items-center gap-3">
                {/* Reformatted Badge */}
                {response.reformatted && (
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800 border border-purple-200">
                    ✨ Reformatted by Groq
                  </span>
                )}
                <ConfidenceBadge 
                  uncertainty={response.uncertainty_score} 
                  tier={response.strictness_tier} 
                />
              </div>
            </div>

            {/* Quality Score Bar */}
            {response.quality_score !== null && response.quality_score !== undefined && (
              <div className="px-6 py-3 border-b bg-gray-50/50">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                    Quality Score
                  </span>
                  <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold border ${getQualityColor(response.quality_score)}`}>
                    {(response.quality_score * 100).toFixed(0)}% — {getQualityLabel(response.quality_score)}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-500 ${
                      response.quality_score >= 0.8
                        ? "bg-emerald-500"
                        : response.quality_score >= 0.5
                        ? "bg-amber-500"
                        : "bg-red-500"
                    }`}
                    style={{ width: `${(response.quality_score * 100).toFixed(0)}%` }}
                  />
                </div>
              </div>
            )}

            {/* Issues Found */}
            {response.issues_found && response.issues_found.length > 0 && (
              <div className="px-6 py-3 border-b">
                <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <svg className="w-4 h-4 text-amber-600" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
                    </svg>
                    <span className="text-sm font-semibold text-amber-800">Issues Detected</span>
                  </div>
                  <ul className="list-disc list-inside space-y-1">
                    {response.issues_found.map((issue: string, idx: number) => (
                      <li key={idx} className="text-sm text-amber-700">{issue}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            {/* Answer */}
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <h2 className="text-lg font-bold text-gray-900">Response</h2>
                <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                  response.decision === "Abstained" 
                    ? "bg-red-100 text-red-800" 
                    : response.decision?.includes("Hallucination")
                    ? "bg-orange-100 text-orange-800"
                    : "bg-blue-100 text-blue-800"
                }`}>
                  Decision: {response.decision || "Answered"}
                </span>
              </div>
              
              <div className="prose prose-sm max-w-none text-gray-800 whitespace-pre-wrap">
                {response.answer}
              </div>

              {/* Citation Verification */}
              <CitationCard citations={response.citations_verified} />
              
              {/* Evidence */}
              <EvidencePanel evidence={response.evidence} />
            </div>
            
            {/* Footer metrics */}
            <div className="bg-gray-50 px-6 py-3 border-t flex justify-between text-xs text-gray-500">
              <span>Time: {response.processing_time_ms?.toFixed(0)}ms</span>
              <span>Retrieval: {response.retrieval_method}</span>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
