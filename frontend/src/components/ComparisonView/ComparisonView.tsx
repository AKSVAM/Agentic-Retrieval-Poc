import ThinkingPanel from "../ThinkingPanel/ThinkingPanel";
import AnswerPanel from "../AnswerPanel/AnswerPanel";
import type { SearchResult, ThoughtStep } from "../../api/types";
import styles from "./ComparisonView.module.css";

interface ComparisonViewProps {
  vectorResult: SearchResult | null;
  graphResult: SearchResult | null;
}

// Educational annotations for each vector_search step (0-indexed occurrence within the result)
const VECTOR_STEP_NOTES: Record<number, string> = {
  0: "The query is encoded into a high-dimensional vector. Similarity is measured by cosine distance — effective for lexical overlap but blind to cross-document relationships.",
  1: "Chunks are ranked by similarity score alone. All chunks compete in a flat index: a PO and its matching Invoice are not linked — they only appear together if both happen to be similar to the query.",
  2: "The context window is a flat list of passages. The LLM must infer relationships from overlapping text, which fails when the connecting information lives in a different document.",
};

function vectorNotes(step: ThoughtStep, vsIndex: number): string | undefined {
  if (step.step_type === "vector_search") {
    return VECTOR_STEP_NOTES[vsIndex];
  }
  if (step.step_type === "answer_generation") {
    return "LLM is given only the top-K similar chunks. Multi-hop context (e.g. the GRN that closes a PO) is absent unless it happened to score highly on its own.";
  }
  return undefined;
}

function SummaryBar({ vector, graph }: { vector: SearchResult; graph: SearchResult }) {
  const graphHops = graph.steps.filter((s) => s.step_type === "hop_discovery").length;
  const graphEntities = (() => {
    const last = [...graph.steps].reverse().find((s) => s.step_type === "chunk_retrieval");
    if (!last) return "?";
    const m = last.description.match(/from (\d+) entities/);
    return m ? m[1] : "?";
  })();
  const graphChunks = (() => {
    const last = [...graph.steps].reverse().find((s) => s.step_type === "chunk_retrieval");
    if (!last) return "?";
    const m = last.description.match(/Retrieved (\d+) chunks/);
    return m ? m[1] : "?";
  })();
  const vectorChunks = (() => {
    const last = [...vector.steps].reverse().find((s) => s.step_type === "vector_search" && s.description.includes("chunks"));
    if (!last) return "?";
    const m = last.description.match(/top (\d+) chunks/);
    return m ? m[1] : "?";
  })();

  return (
    <div className={styles.summary}>
      <span className={styles.summaryItem}>
        <strong>GraphRAG</strong> traversed <strong>{graphEntities}</strong> entities across <strong>{graphHops}</strong> hop{graphHops !== 1 ? "s" : ""}, retrieved <strong>{graphChunks}</strong> chunks
      </span>
      <span className={styles.summaryDivider}>·</span>
      <span className={styles.summaryItem}>
        <strong>Standard RAG</strong> retrieved <strong>{vectorChunks}</strong> chunks by similarity alone
      </span>
    </div>
  );
}

export default function ComparisonView({ vectorResult, graphResult }: ComparisonViewProps) {
  const bothDone =
    vectorResult && !vectorResult.isStreaming &&
    graphResult && !graphResult.isStreaming;

  return (
    <div className={styles.root}>
      {bothDone && (
        <SummaryBar vector={vectorResult!} graph={graphResult!} />
      )}

      <div className={styles.columns}>
        {/* Left: Standard Vector RAG */}
        <div className={styles.col}>
          <div className={styles.colHeader} data-side="vector">
            <span className={styles.colTitle}>Standard Vector RAG</span>
            <span className={styles.colBadge} data-side="vector">No graph</span>
          </div>
          <div className={styles.colNote}>
            Flat similarity search — retrieves the most lexically similar chunks but cannot follow cross-document entity relationships.
          </div>
          <ThinkingPanel
            steps={vectorResult?.steps ?? []}
            isStreaming={vectorResult?.isStreaming ?? false}
            label="Vector Search Steps"
            educationalNotes={vectorNotes}
          />
          <AnswerPanel
            answer={vectorResult?.answer ?? null}
            citations={vectorResult?.citations ?? []}
            queryType={vectorResult?.queryType ?? null}
            isStreaming={vectorResult?.isStreaming ?? false}
            error={vectorResult?.error ?? null}
          />
        </div>

        <div className={styles.divider} />

        {/* Right: Agentic GraphRAG */}
        <div className={styles.col}>
          <div className={styles.colHeader} data-side="graphrag">
            <span className={styles.colTitle}>Agentic GraphRAG</span>
            <span className={styles.colBadge} data-side="graphrag">Multi-hop</span>
          </div>
          <div className={styles.colNote}>
            LLM-driven entity extraction → semantic graph traversal → relationship-aware answer synthesis across documents.
          </div>
          <ThinkingPanel
            steps={graphResult?.steps ?? []}
            isStreaming={graphResult?.isStreaming ?? false}
            label="Graph Traversal Steps"
          />
          <AnswerPanel
            answer={graphResult?.answer ?? null}
            citations={graphResult?.citations ?? []}
            queryType={graphResult?.queryType ?? null}
            isStreaming={graphResult?.isStreaming ?? false}
            error={graphResult?.error ?? null}
          />
        </div>
      </div>
    </div>
  );
}
