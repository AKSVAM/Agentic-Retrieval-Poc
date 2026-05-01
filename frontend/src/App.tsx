import { useState } from "react";
import SearchPage from "./pages/SearchPage/SearchPage";
import GraphPage from "./pages/GraphPage/GraphPage";
import "./App.css";

type Tab = "search" | "graph";

export default function App() {
  const [tab, setTab] = useState<Tab>("search");
  return (
    <>
      <nav className="tab-bar">
        <button
          className={`tab-btn ${tab === "search" ? "active" : ""}`}
          onClick={() => setTab("search")}
        >
          Search
        </button>
        <button
          className={`tab-btn ${tab === "graph" ? "active" : ""}`}
          onClick={() => setTab("graph")}
        >
          Graph Explorer
        </button>
      </nav>
      {tab === "search" ? <SearchPage /> : <GraphPage />}
    </>
  );
}
