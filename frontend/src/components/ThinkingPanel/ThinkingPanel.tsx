import { useEffect, useRef } from "react";
import StepCard from "../StepCard/StepCard";
import type { ThoughtStep, StepType } from "../../api/types";
import styles from "./ThinkingPanel.module.css";

interface ThinkingPanelProps {
  steps: ThoughtStep[];
  isStreaming: boolean;
  label?: string;
  educationalNotes?: (step: ThoughtStep, vsIndex: number) => string | undefined;
}

export default function ThinkingPanel({
  steps,
  isStreaming,
  label = "Thought Process",
  educationalNotes,
}: ThinkingPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const firstTime = steps[0]?.timestamp;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [steps.length]);

  // Track per-step_type index for educational note selection
  const typeCounters: Partial<Record<StepType, number>> = {};

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.label}>{label}</span>
        {isStreaming && <span className={styles.dot} />}
        {!isStreaming && steps.length > 0 && (
          <span className={styles.count}>{steps.length} steps</span>
        )}
      </div>

      <div className={styles.steps}>
        {isStreaming && steps.length === 0 && (
          <>
            <div className={`${styles.skeleton} ${styles.sk1}`} />
            <div className={`${styles.skeleton} ${styles.sk2}`} />
            <div className={`${styles.skeleton} ${styles.sk3}`} />
          </>
        )}

        {steps.map((step) => {
          const prevCount = typeCounters[step.step_type] ?? 0;
          typeCounters[step.step_type] = prevCount + 1;
          const note = educationalNotes?.(step, prevCount);
          return (
            <StepCard
              key={step.index}
              step={step}
              firstStepTime={firstTime}
              educationalNote={note}
            />
          );
        })}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
