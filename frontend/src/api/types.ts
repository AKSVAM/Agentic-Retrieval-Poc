export type StepType =
  | "entity_extraction"
  | "hop_discovery"
  | "hop_planning"
  | "traversal_complete"
  | "chunk_retrieval"
  | "vector_search"
  | "fallback"
  | "answer_generation";

export interface ChunkData {
  id: string;
  content: string;
  sourcefile: string;
  document_type: string;
  vendor?: string;
  project?: string;
}

export interface EntityRelationship {
  entity_id: string;
  entity_name?: string;
  relationship_type: string;
}

export interface EntityData {
  entity_id: string;
  entity_name: string;
  entity_type: string;
  relationships: EntityRelationship[];
}

export interface ThoughtStepEvent {
  type: "thought_step";
  title: string;
  description: string;
  step_type: StepType;
  entities?: EntityData[];
  chunks?: ChunkData[];
}

export interface AnswerEvent {
  type: "answer";
  content: string;
  citations: string[];
  query_type: string;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type StreamEvent = ThoughtStepEvent | AnswerEvent | ErrorEvent;

export interface ThoughtStep extends ThoughtStepEvent {
  timestamp: Date;
  index: number;
}

export interface SearchResult {
  query: string;
  mode: string;
  steps: ThoughtStep[];
  answer: string | null;
  citations: string[];
  queryType: string | null;
  isStreaming: boolean;
  error: string | null;
}
