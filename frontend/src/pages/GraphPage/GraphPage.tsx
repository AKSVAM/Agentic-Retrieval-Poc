import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { ForceGraphMethods } from "react-force-graph-2d";
import { fetchGraph } from "../../api/graphClient";
import type { GraphNode } from "../../api/types";
import styles from "./GraphPage.module.css";

const TYPE_COLORS: Record<string, string> = {
  vendor: "#8b5cf6",
  customer: "#3b82f6",
  po: "#f97316",
  invoice: "#ef4444",
  grn: "#22c55e",
  project: "#14b8a6",
  item: "#f59e0b",
  contact: "#ec4899",
  unknown: "#9ca3af",
};

function typeColor(type: string): string {
  return TYPE_COLORS[type] ?? TYPE_COLORS.unknown;
}

interface FGNode extends GraphNode {
  x?: number;
  y?: number;
}

interface FGLink {
  source: string | FGNode;
  target: string | FGNode;
  relationship_type: string;
}

function nodeId(n: string | FGNode): string {
  return typeof n === "string" ? n : n.id;
}

const NODE_BASE_R = 6;

export default function GraphPage() {
  const [allNodes, setAllNodes] = useState<GraphNode[]>([]);
  const [allEdges, setAllEdges] = useState<FGLink[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTypes, setActiveTypes] = useState<Set<string>>(new Set());
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const fgRef = useRef<ForceGraphMethods<FGNode, FGLink>>(undefined);

  useEffect(() => {
    fetchGraph()
      .then((data) => {
        setAllNodes(data.nodes);
        setAllEdges(data.edges as FGLink[]);
        setActiveTypes(new Set(data.nodes.map((n) => n.type)));
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const allTypes = useMemo(
    () => Array.from(new Set(allNodes.map((n) => n.type))).sort(),
    [allNodes]
  );

  // Build graphData once from allNodes/allEdges, then never recreate it.
  // Filtering is handled by hiding nodes via nodeVisibility instead of
  // re-slicing the array, which avoids breaking the force simulation's
  // internal source/target object references.
  const graphData = useMemo(
    () => ({ nodes: allNodes as FGNode[], links: allEdges }),
    [allNodes, allEdges]
  );

  const toggleType = (type: string) => {
    setActiveTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  const visibleCount = allNodes.filter((n) => activeTypes.has(n.type)).length;
  const visibleEdgeCount = allEdges.filter((e) => {
    const s = allNodes.find((n) => n.id === nodeId(e.source));
    const t = allNodes.find((n) => n.id === nodeId(e.target));
    return s && activeTypes.has(s.type) && t && activeTypes.has(t.type);
  }).length;

  const handleNodeClick = useCallback((node: object) => {
    setSelectedNode(node as GraphNode);
  }, []);

  const handleReset = () => fgRef.current?.zoomToFit(400);

  if (loading) return <div className={styles.center}>Loading graph…</div>;
  if (error) return <div className={styles.center}>Error: {error}</div>;
  if (allNodes.length === 0)
    return <div className={styles.center}>No entities found — run ingestion first.</div>;

  return (
    <div className={styles.page}>
      <div className={styles.toolbar}>
        <span className={styles.filterLabel}>Filter</span>
        {allTypes.map((type) => (
          <span
            key={type}
            className={`${styles.filterChip} ${activeTypes.has(type) ? "" : styles.inactive}`}
            style={{ borderColor: typeColor(type) }}
            onClick={() => toggleType(type)}
          >
            <span className={styles.dot} style={{ background: typeColor(type) }} />
            {type}
          </span>
        ))}
        <div className={styles.spacer} />
        <span className={styles.stats}>
          {visibleCount} nodes · {visibleEdgeCount} edges
        </span>
        <button className={styles.resetBtn} onClick={handleReset}>
          Reset zoom
        </button>
      </div>

      <div className={styles.body}>
        <div className={styles.canvas}>
          <ForceGraph2D
            ref={fgRef}
            graphData={graphData}
            nodeId="id"
            nodeVisibility={(n) => activeTypes.has((n as FGNode).type)}
            linkVisibility={(l) => {
              const link = l as FGLink;
              const sType = allNodes.find((n) => n.id === nodeId(link.source))?.type ?? "";
              const tType = allNodes.find((n) => n.id === nodeId(link.target))?.type ?? "";
              return activeTypes.has(sType) && activeTypes.has(tType);
            }}
            linkColor={() => "#64748b"}
            linkWidth={1.5}
            linkDirectionalArrowLength={5}
            linkDirectionalArrowRelPos={1}
            linkDirectionalParticles={1}
            linkDirectionalParticleWidth={2}
            onNodeClick={handleNodeClick}
            nodeCanvasObject={(node, ctx, globalScale) => {
              const n = node as FGNode;
              const x = n.x ?? 0;
              const y = n.y ?? 0;
              const r = NODE_BASE_R + Math.sqrt(n.mention_count ?? 1);
              const color = typeColor(n.type);

              // glow ring for selected node
              if (selectedNode && n.id === selectedNode.id) {
                ctx.beginPath();
                ctx.arc(x, y, r + 4, 0, 2 * Math.PI);
                ctx.fillStyle = color + "40";
                ctx.fill();
              }

              // node circle
              ctx.beginPath();
              ctx.arc(x, y, r, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.fill();
              ctx.strokeStyle = "#ffffff";
              ctx.lineWidth = 1.5;
              ctx.stroke();

              // always show label
              const label = n.name;
              const fontSize = Math.max(10, 12 / globalScale);
              ctx.font = `${fontSize}px sans-serif`;
              ctx.textAlign = "center";
              ctx.textBaseline = "top";

              // label background pill
              const textWidth = ctx.measureText(label).width;
              const padding = 3 / globalScale;
              const bx = x - textWidth / 2 - padding;
              const by = y + r + 4 / globalScale;
              const bw = textWidth + padding * 2;
              const bh = fontSize + padding * 2;
              ctx.fillStyle = "rgba(255,255,255,0.85)";
              ctx.fillRect(bx, by, bw, bh);

              ctx.fillStyle = "#1f2937";
              ctx.fillText(label, x, by + padding);
            }}
            nodePointerAreaPaint={(node, color, ctx) => {
              const n = node as FGNode;
              const r = NODE_BASE_R + Math.sqrt(n.mention_count ?? 1) + 6;
              ctx.beginPath();
              ctx.arc(n.x ?? 0, n.y ?? 0, r, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.fill();
            }}
          />
        </div>

        <div className={styles.sidebar}>
          {selectedNode ? (
            <>
              <p className={styles.sidebarTitle}>{selectedNode.name}</p>
              <span
                className={styles.sidebarBadge}
                style={{ background: typeColor(selectedNode.type) }}
              >
                {selectedNode.type}
              </span>
              <div className={styles.sidebarRow}>
                <span className={styles.sidebarRowLabel}>Mentioned</span>
                <span className={styles.sidebarRowValue}>{selectedNode.mention_count}×</span>
              </div>
              <div className={styles.sidebarRow}>
                <span className={styles.sidebarRowLabel}>Source files</span>
                {selectedNode.source_files.length > 0 ? (
                  <ul className={styles.fileList}>
                    {selectedNode.source_files.map((f) => (
                      <li key={f} className={styles.fileItem}>{f}</li>
                    ))}
                  </ul>
                ) : (
                  <span className={styles.sidebarRowValue}>—</span>
                )}
              </div>
              <div className={styles.sidebarRow}>
                <span className={styles.sidebarRowLabel}>Entity ID</span>
                <span className={styles.entityId}>{selectedNode.id}</span>
              </div>
            </>
          ) : (
            <p className={styles.sidebarEmpty}>Click a node to inspect it</p>
          )}
        </div>
      </div>
    </div>
  );
}
