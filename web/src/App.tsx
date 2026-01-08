import { Routes, Route } from "react-router-dom";
import { RunList } from "./pages/RunList";
import { RunDetail } from "./pages/RunDetail";
import { Compare } from "./pages/Compare";

export default function App() {
  return (
    <div style={{ minHeight: "100vh", padding: "20px", maxWidth: "1400px", margin: "0 auto" }}>
      <header style={{ marginBottom: "24px", borderBottom: "1px solid #30363d", paddingBottom: "16px" }}>
        <h1 style={{ fontSize: "24px", fontWeight: 600 }}>
          <a href="/" style={{ color: "#e6edf3" }}>extty</a>
        </h1>
      </header>
      <Routes>
        <Route path="/" element={<RunList />} />
        <Route path="/runs/:id" element={<RunDetail />} />
        <Route path="/compare" element={<Compare />} />
      </Routes>
    </div>
  );
}
