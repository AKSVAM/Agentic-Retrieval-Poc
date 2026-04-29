import { useState, useCallback } from "react";
import SearchBar, { type SearchMode } from "../../components/SearchBar/SearchBar";
import ThinkingPanel from "../../components/ThinkingPanel/ThinkingPanel";
import AnswerPanel from "../../components/AnswerPanel/AnswerPanel";
import ComparisonView from "../../components/ComparisonView/ComparisonView";
import { useSearchStream } from "../../hooks/useSearchStream";
import styles from "./SearchPage.module.css";

export default function SearchPage() {
  const { result: graphResult, search: searchGraph } = useSearchStream("graphrag");
  const { result: vectorResult, search: searchVector } = useSearchStream("vector");
  const [activeMode, setActiveMode] = useState<SearchMode>("graphrag");

  const isLoading =
    (graphResult?.isStreaming ?? false) || (vectorResult?.isStreaming ?? false);

  const handleSearch = useCallback(
    (query: string, mode: SearchMode) => {
      setActiveMode(mode);
      searchGraph(query);
      if (mode === "compare") {
        searchVector(query);
      }
    },
    [searchGraph, searchVector]
  );

  const hasResult = !!graphResult;

  return (
    <div className={styles.page}>
      <div className={`${styles.searchArea} ${hasResult ? styles.top : styles.center}`}>
        <SearchBar onSearch={handleSearch} isLoading={isLoading} />
      </div>

      {hasResult && (
        <div className={styles.results}>
          {activeMode === "graphrag" && (
            <div className={styles.singleLayout}>
              <ThinkingPanel
                steps={graphResult.steps}
                isStreaming={graphResult.isStreaming}
                label="GraphRAG Traversal"
              />
              <AnswerPanel
                answer={graphResult.answer}
                citations={graphResult.citations}
                queryType={graphResult.queryType}
                isStreaming={graphResult.isStreaming}
                error={graphResult.error}
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
