import { useState } from "react";
import {
  Search,
  Network,
  GitBranch,
  CheckCircle2,
  FileText,
  BarChart3,
  AlertCircle,
  Sparkles,
  ChevronRight,
  ChevronDown,
} from "lucide-react";
import type { ThoughtStep, StepType, EntityData } from "../../api/types";
import ChunkCard from "../ChunkCard/ChunkCard";
import styles from "./StepCard.module.css";

const ICONS: Record<StepType, React.ReactNode> = {
  entity_extraction: <Search size={16} />,
  hop_discovery: <Network size={16} />,
  hop_planning: <GitBranch size={16} />,
  traversal_complete: <CheckCircle2 size={16} />,
  chunk_retrieval: <FileText size={16} />,
  vector_search: <BarChart3 size={16} />,
  fallback: <AlertCircle size={16} />,
  answer_generation: <Sparkles size={16} />,
};

const ENTITY_TYPE_COLORS: Record<string, string> = {
  vendor: "#7c3aed",
  customer: "#2563eb",
  po: "#d97706",
  invoice: "#ea580c",
  grn: "#16a34a",
  goods_receipt_note: "#16a34a",
  project: "#0891b2",
  item: "#6b7280",
  contact: "#db2777",
};

interface StepCardProps {
  step: ThoughtStep;
  firstStepTime?: Date;
  educationalNote?: string;
}

function EntityList({ entities }: { entities: EntityData[] }) {
  return (
    <div className={styles.entityList}>
      {entities.map((entity) => {
        const color = ENTITY_TYPE_COLORS[entity.entity_type] ?? "#6b7280";
        return (
          <div key={entity.entity_id} className={styles.entityRow}>
            <div className={styles.entityHeader}>
              <span
                className={styles.entityTypeBadge}
                style={{ background: `${color}18`, color }}
              >
                {entity.entity_type}
              </span>
              <span className={styles.entityName}>{entity.entity_name || entity.entity_id.slice(0, 8) + "…"}</span>
            </div>
            {entity.relationships.length > 0 ? (
              <div className={styles.relationshipList}>
                {entity.relationships.map((rel, i) => (
                  <div key={i} className={styles.relationshipRow}>
                    <span className={styles.relArrow}>└─</span>
                    <span className={styles.relType}>{rel.relationship_type}</span>
                    <span className={styles.relArrowIcon}>→</span>
                    <span className={styles.relTarget}>
                      {rel.entity_name || (rel.entity_id ? rel.entity_id.slice(0, 8) + "…" : "unknown")}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className={styles.relationshipList}>
                <span className={styles.noRels}>no connections mapped yet</span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function StepCard({ step, firstStepTime, educationalNote }: StepCardProps) {
  const [entitiesOpen, setEntitiesOpen] = useState(false);
  const [chunksOpen, setChunksOpen] = useState(false);

  const elapsed = firstStepTime
    ? ((step.timestamp.getTime() - firstStepTime.getTime()) / 1000).toFixed(1)
    : null;

  const delay = Math.min(step.index * 0.06, 0.3);

  const hasEntities = step.entities && step.entities.length > 0;
  const hasChunks = step.chunks && step.chunks.length > 0;

  return (
    <div
      className={`${styles.card} ${styles[step.step_type]}`}
      style={{ animationDelay: `${delay}s` }}
    >
      <div className={styles.header}>
        <span className={styles.icon}>{ICONS[step.step_type]}</span>
        <span className={styles.title}>{step.title}</span>
        {elapsed !== null && (
          <span className={styles.elapsed}>+{elapsed}s</span>
        )}
      </div>
      <p className={styles.description}>{step.description}</p>
      {step.reasoning && (
        <p className={styles.reasoning}>{step.reasoning}</p>
      )}

      {educationalNote && (
        <div className={styles.note}>{educationalNote}</div>
      )}

      {hasEntities && (
        <div className={styles.expandSection}>
          <button
            className={styles.expandToggle}
            onClick={() => setEntitiesOpen(v => !v)}
          >
            {entitiesOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
            {step.entities!.length} {step.entities!.length === 1 ? "entity" : "entities"} discovered
          </button>
          {entitiesOpen && <EntityList entities={step.entities!} />}
        </div>
      )}

      {hasChunks && (
        <div className={styles.expandSection}>
          <button
            className={styles.expandToggle}
            onClick={() => setChunksOpen(v => !v)}
          >
            {chunksOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
            {step.chunks!.length} {step.chunks!.length === 1 ? "chunk" : "chunks"} retrieved
          </button>
          {chunksOpen && (
            <div className={styles.chunkList}>
              {step.chunks!.map((chunk) => (
                <ChunkCard key={chunk.id} chunk={chunk} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
