import { useState, useCallback, useRef } from "react";
import { fetchSearchStream } from "../api/streamClient";
import type { SearchResult, ThoughtStep } from "../api/types";

function emptyResult(query: string, mode: string): SearchResult {
  return {
    query,
    mode,
    steps: [],
    answer: null,
    citations: [],
    queryType: null,
    isStreaming: false,
    error: null,
  };
}

export function useSearchStream(mode: string = "auto") {
  const [result, setResult] = useState<SearchResult | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const stepIndexRef = useRef(0);

  const search = useCallback(
    async (query: string) => {
      abortRef.current?.abort();
      abortRef.current = new AbortController();
      stepIndexRef.current = 0;

      setResult({ ...emptyResult(query, mode), isStreaming: true });

      try {
        for await (const event of fetchSearchStream(query, mode)) {
          if (event.type === "thought_step") {
            const step: ThoughtStep = {
              ...event,
              timestamp: new Date(),
              index: stepIndexRef.current++,
            };
            setResult((prev) =>
              prev ? { ...prev, steps: [...prev.steps, step] } : prev
            );
          } else if (event.type === "answer") {
            setResult((prev) =>
              prev
                ? {
                    ...prev,
                    answer: event.content,
                    citations: event.citations,
                    queryType: event.query_type,
                    isStreaming: false,
                  }
                : prev
            );
          } else if (event.type === "error") {
            setResult((prev) =>
              prev ? { ...prev, error: event.message, isStreaming: false } : prev
            );
          }
        }
        // Mark done if stream ended without an answer event
        setResult((prev) =>
          prev?.isStreaming ? { ...prev, isStreaming: false } : prev
        );
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") return;
        setResult((prev) =>
          prev ? { ...prev, error: String(err), isStreaming: false } : null
        );
      }
    },
    [mode]
  );

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setResult(null);
  }, []);

  return { result, search, reset };
}
