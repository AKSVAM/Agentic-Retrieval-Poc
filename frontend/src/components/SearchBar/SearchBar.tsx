import { useState, useRef, type KeyboardEvent } from "react";
import { Send, GitBranch, Columns2 } from "lucide-react";
import styles from "./SearchBar.module.css";

export type SearchMode = "graphrag" | "compare";

interface SearchBarProps {
  onSearch: (query: string, mode: SearchMode) => void;
  isLoading: boolean;
}

export default function SearchBar({ onSearch, isLoading }: SearchBarProps) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("graphrag");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function handleSubmit() {
    const q = query.trim();
    if (!q || isLoading) return;
    onSearch(q, mode);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  function handleInput() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  }

  return (
    <div className={styles.wrapper}>
      <h1 className={styles.title}>GraphRAG Procurement Search</h1>
      <p className={styles.subtitle}>
        Multi-hop graph traversal over procurement documents
      </p>

      <div className={styles.modeToggle}>
        <button
          className={`${styles.modeBtn} ${mode === "graphrag" ? styles.active : ""}`}
          onClick={() => setMode("graphrag")}
        >
          <GitBranch size={14} />
          AI Search
        </button>
        <button
          className={`${styles.modeBtn} ${mode === "compare" ? styles.active : ""}`}
          onClick={() => setMode("compare")}
        >
          <Columns2 size={14} />
          Compare Modes
        </button>
      </div>

      <div className={styles.inputRow}>
        <textarea
          ref={textareaRef}
          className={styles.textarea}
          placeholder={
            mode === "compare"
              ? "Ask a question to compare Standard RAG vs. GraphRAG…"
              : "Ask a question about your procurement documents…"
          }
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          rows={1}
          disabled={isLoading}
        />
        <button
          className={styles.sendBtn}
          onClick={handleSubmit}
          disabled={!query.trim() || isLoading}
          title="Send (Enter)"
        >
          <Send size={16} />
        </button>
      </div>

      <p className={styles.hint}>
        {mode === "compare"
          ? "Runs both retrieval methods simultaneously so you can compare depth of results."
          : "Press Enter to search · Shift+Enter for newline"}
      </p>
    </div>
  );
}
