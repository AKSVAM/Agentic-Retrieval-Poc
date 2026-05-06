import { useState, useCallback } from "react";
import SearchBar, { type SearchMode } from "../../components/SearchBar/SearchBar";
import ThinkingPanel from "../../components/ThinkingPanel/ThinkingPanel";
import AnswerPanel from "../../components/AnswerPanel/AnswerPanel";
import ComparisonView from "../../components/ComparisonView/ComparisonView";
import { useSearchStream } from "../../hooks/useSearchStream";
import styles from "./SearchPage.module.css";

export default function SearchPage() {
  const { result: agentResult, search: searchAgent } = useSearchStream("auto");
  const { result: graphResult, search: searchGraph } = useSearchStream("graphrag");
  const { result: vectorResult, search: searchVector } = useSearchStream("vector");
  const [activeMode, setActiveMode] = useState<SearchMode>("auto");

  const isLoading =
    (agentResult?.isStreaming ?? false) ||
    (graphResult?.isStreaming ?? false) ||
    (vectorResult?.isStreaming ?? false);

  const handleSearch = useCallback(
    (query: string, mode: SearchMode) => {
      setActiveMode(mode);
      if (mode === "compare") {
        searchGraph(query);
        searchVector(query);
      } else {
        searchAgent(query);
      }
    },
    [searchAgent, searchGraph, searchVector]
  );

  const hasResult = activeMode === "compare" ? !!graphResult : !!agentResult;

  return (
    <div className={styles.page}>
      <div className={`${styles.searchArea} ${hasResult ? styles.top : styles.center}`}>
        <SearchBar onSearch={handleSearch} isLoading={isLoading} />
      </div>

      {hasResult && (
        <div className={styles.results}>
          {activeMode === "auto" && agentResult && (
            <div className={styles.singleLayout}>
              <ThinkingPanel
                steps={agentResult.steps}
                isStreaming={agentResult.isStreaming}
                label="Agent Search"
              />
              <AnswerPanel
                answer={agentResult.answer}
                citations={agentResult.citations}
                queryType={agentResult.queryType}
                isStreaming={agentResult.isStreaming}
                error={agentResult.error}
              />
            </div>
          )}

          {activeMode === "compare" && (
            <ComparisonView
              vectorResult={vectorResult}
              graphResult={graphResult}
            />
          )}
        </div>
      )}
    </div>
  );
}
