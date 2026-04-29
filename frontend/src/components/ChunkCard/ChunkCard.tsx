import { useState } from "react";
import type { ChunkData } from "../../api/types";
import styles from "./ChunkCard.module.css";

interface ChunkCardProps {
  chunk: ChunkData;
}

const DOC_TYPE_LABELS: Record<string, string> = {
  purchase_order: "PO",
  invoice: "INV",
  grn: "GRN",
  goods_receipt_note: "GRN",
};

export default function ChunkCard({ chunk }: ChunkCardProps) {
  const [expanded, setExpanded] = useState(false);

  const label = DOC_TYPE_LABELS[chunk.document_type] ?? chunk.document_type ?? "DOC";
  const filename = chunk.sourcefile.split("/").pop() ?? chunk.sourcefile;
  const preview = chunk.content.length > 200
    ? chunk.content.slice(0, 200).trimEnd() + "…"
    : chunk.content;

  return (
    <div className={`${styles.card} ${styles[chunk.document_type] ?? ""}`}>
      <div className={styles.meta}>
        <span className={`${styles.typeTag} ${styles[`tag_${chunk.document_type}`] ?? styles.tag_default}`}>
          {label}
        </span>
        <span className={styles.filename}>{filename}</span>
        {chunk.vendor && <span className={styles.metaItem}>{chunk.vendor}</span>}
        {chunk.project && <span className={styles.metaItem}>{chunk.project}</span>}
      </div>
      <p className={styles.content}>
        {expanded ? chunk.content : preview}
      </p>
      {chunk.content.length > 200 && (
        <button
          className={styles.toggle}
          onClick={() => setExpanded(v => !v)}
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}
