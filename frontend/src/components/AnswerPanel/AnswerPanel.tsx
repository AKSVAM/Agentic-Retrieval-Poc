import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import styles from "./AnswerPanel.module.css";

interface AnswerPanelProps {
  answer: string | null;
  citations: string[];
  queryType: string | null;
  isStreaming: boolean;
  error: string | null;
}

const BADGE_LABELS: Record<string, string> = {
  graphrag: "GraphRAG",
  vector: "Standard RAG",
  keyword_fallback: "Vector Fallback",
};

const BADGE_CLASSES: Record<string, string> = {
  graphrag: "blue",
  vector: "amber",
  keyword_fallback: "gray",
};

export default function AnswerPanel({
  answer,
  citations,
  queryType,
  isStreaming,
  error,
}: AnswerPanelProps) {
  if (error) {
    return (
      <div className={styles.panel}>
        <div className={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      </div>
    );
  }

  if (!answer && isStreaming) {
    return (
      <div className={styles.panel}>
        <div className={styles.waitingHint}>Waiting for answer…</div>
      </div>
    );
  }

  if (!answer) return null;

  const badgeLabel = queryType ? (BADGE_LABELS[queryType] ?? queryType) : null;
  const badgeClass = queryType ? (BADGE_CLASSES[queryType] ?? "gray") : "gray";

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.answerLabel}>Answer</span>
        {badgeLabel && (
          <span className={`${styles.badge} ${styles[badgeClass]}`}>
            {badgeLabel}
          </span>
        )}
      </div>

      <div className={styles.markdown}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
      </div>

      {citations.length > 0 && (
        <div className={styles.citations}>
          <span className={styles.citLabel}>Sources:</span>
          {citations.map((c) => (
            <span key={c} className={styles.chip}>
              {c}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
