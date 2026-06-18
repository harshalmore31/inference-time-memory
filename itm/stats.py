import numpy as np

from itm.core import MemoryLayer
from itm.embeddings import EmbeddingService


class MemoryStats:
    """Memory statistics and search utilities."""

    def summary(self, layer: MemoryLayer) -> str:
        """Return formatted summary statistics."""
        n = len(layer.memories)
        if n == 0:
            return f"Memory empty | timestep: {layer.timestep}"

        strengths = np.array([m.strength for m in layer.memories])

        lines = [
            "--- Memory Stats ---",
            f"Total memories:  {n}",
            f"Timestep:        {layer.timestep}",
            f"Avg strength:    {np.mean(strengths):.4f}",
            f"Max strength:    {np.max(strengths):.4f}",
            f"Min strength:    {np.min(strengths):.4f}",
            f"Median strength: {np.median(strengths):.4f}",
            "",
        ]

        # Top 5 strongest
        sorted_mems = sorted(layer.memories, key=lambda m: m.strength, reverse=True)
        lines.append("Top 5 strongest:")
        for mem in sorted_mems[:5]:
            lines.append(f"  s={mem.strength:.4f} | {mem.input_text[:60]}")

        # Bottom 5 weakest
        lines.append("\nTop 5 weakest:")
        for mem in sorted_mems[-5:]:
            lines.append(f"  s={mem.strength:.4f} | {mem.input_text[:60]}")

        # Strength distribution
        lines.append("\nStrength distribution:")
        buckets = [(0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float("inf"))]
        for lo, hi in buckets:
            count = int(np.sum((strengths >= lo) & (strengths < hi)))
            bar = "#" * min(count, 40)
            label = f"{lo:.1f}-{hi:.1f}" if hi != float("inf") else f"{lo:.1f}+"
            lines.append(f"  [{label:>8}] {count:>4} {bar}")

        # Memory graph info
        graph = layer.graph
        lines.append("\nMemory Graph:")
        lines.append(f"  Nodes: {graph.node_count()}  Edges: {graph.edge_count()}")

        # Show connections for top memories
        lines.append("\nGraph connections (top memories):")
        for i, mem in enumerate(sorted_mems[:5]):
            idx = layer.memories.index(mem)
            neighbors = graph.get_neighbors(idx)
            if neighbors:
                neighbor_strs = []
                for j, w in neighbors[:3]:
                    neighbor_strs.append(
                        f"{layer.memories[j].input_text[:30]}(w={w:.2f})"
                    )
                lines.append(f"  [{mem.input_text[:40]}]")
                for ns in neighbor_strs:
                    lines.append(f"    -> {ns}")
            else:
                lines.append(f"  [{mem.input_text[:40]}] (no connections)")

        return "\n".join(lines)

    def search(self, layer: MemoryLayer, query: str, embedder: EmbeddingService) -> str:
        """Execute a graph-based recall query and display detailed results."""
        if not layer.memories:
            return "No memories to search."

        query_emb = embedder.embed(query)
        results = layer.recall_graph(query_emb)

        if not results:
            return f"No relevant memories found for: '{query}'"

        lines = [f"Graph recall results for: '{query}'", ""]
        for i, (mem, activation) in enumerate(results):
            idx = layer.memories.index(mem)
            neighbors = layer.graph.get_neighbors(idx)
            n_edges = len(neighbors)
            lines.append(
                f"  #{i+1} activation={activation:.4f} strength={mem.strength:.4f} edges={n_edges}"
            )
            lines.append(f"      Input:  {mem.input_text[:80]}")
            lines.append(f"      Output: {mem.output_text[:80]}")
            lines.append("")

        return "\n".join(lines)
