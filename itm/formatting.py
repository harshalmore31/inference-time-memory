from itm.core import MemoryEntry, RecallResult


class MemoryFormatter:
    """Format recalled memories for LLM prompt injection and CLI display."""

    def _format_entry(self, mem: MemoryEntry, relevance: float) -> str:
        """Format a single memory entry."""
        if relevance >= 1.0:
            level = "HIGH"
        elif relevance >= 0.3:
            level = "MEDIUM"
        else:
            level = "LOW"

        user_said = mem.input_text[:120]
        your_response = mem.output_text[:100]
        cat_tag = f" [{mem.category}]" if mem.category else ""
        return (
            f'[{level}]{cat_tag} User said: "{user_said}" → You replied: '
            f'"{your_response}" (strength: {mem.strength:.2f})'
        )

    def format_for_prompt(self, recalled: list[tuple[MemoryEntry, float]]) -> str:
        """Format recalled memories for injection into the system prompt.

        Confidence levels based on relevance score (s_i * sim):
          HIGH:   >= 1.0
          MEDIUM: >= 0.3
          LOW:    < 0.3
        """
        if not recalled:
            return ""

        lines = ["=== MEMORY CONTEXT ==="]
        lines.append(
            "Use these memories to answer accurately. Distinguish between "
            "different relationship types (friends vs partners vs colleagues)."
        )
        for mem, relevance in recalled:
            if relevance >= 1.0:
                level = "HIGH"
            elif relevance >= 0.3:
                level = "MEDIUM"
            else:
                level = "LOW"

            # Show both user statement and your response for full context
            user_said = mem.input_text[:120]
            your_response = mem.output_text[:100]
            lines.append(
                f'[{level}] User said: "{user_said}" → You replied: "{your_response}" '
                f"(strength: {mem.strength:.2f})"
            )
        lines.append("=== END MEMORY ===")
        return "\n".join(lines)

    def format_with_tension(self, result: RecallResult) -> str:
        """Equation 13: Format recall with tension detection.

        Shows supporting memories normally and conflicting memories
        under a separate section, forcing the LLM to reconcile.
        """
        if not result.all_results:
            return ""

        lines = ["=== MEMORY CONTEXT ==="]
        lines.append("Use these memories to answer accurately.")

        if result.supporting:
            for mem, rel in result.supporting:
                lines.append(self._format_entry(mem, rel))

        if result.conflicting:
            lines.append("")
            lines.append("=== CONFLICTING MEMORIES (consider both sides) ===")
            for mem, rel in result.conflicting:
                lines.append(self._format_entry(mem, rel))

        lines.append("=== END MEMORY ===")
        return "\n".join(lines)

    def format_channeled(self, recalled: list[tuple[MemoryEntry, float]]) -> str:
        """Equation 14: Format memories grouped by category channel.

        Separates facts from preferences so the LLM can distinguish
        between objective information and user opinions.
        """
        if not recalled:
            return ""

        # Group by category
        channels: dict[str, list[tuple[MemoryEntry, float]]] = {}
        for mem, rel in recalled:
            cat = mem.category or "uncategorized"
            channels.setdefault(cat, []).append((mem, rel))

        lines = ["=== MEMORY CONTEXT ==="]

        # Facts first, then others
        order = ["fact", "event", "instruction", "preference", "uncategorized"]
        for cat in order:
            if cat not in channels:
                continue
            lines.append(f"\n[{cat.upper()}]")
            for mem, rel in channels[cat]:
                lines.append(self._format_entry(mem, rel))

        # Any categories not in the order list
        for cat, entries in channels.items():
            if cat not in order:
                lines.append(f"\n[{cat.upper()}]")
                for mem, rel in entries:
                    lines.append(self._format_entry(mem, rel))

        lines.append("\n=== END MEMORY ===")
        return "\n".join(lines)

    def format_memory_list(self, memories: list[MemoryEntry]) -> str:
        """Format all memories for the 'memories' CLI command."""
        if not memories:
            return "No memories stored."

        # Sort by strength descending
        sorted_mems = sorted(memories, key=lambda m: m.strength, reverse=True)

        lines = [f"{'#':<4} {'Strength':>10} {'Accessed':>10} {'Input Text':<60}"]
        lines.append("-" * 88)

        for i, mem in enumerate(sorted_mems):
            text = mem.input_text[:58]
            lines.append(
                f"{i+1:<4} {mem.strength:>10.4f} {mem.access_count:>10} {text:<60}"
            )

        return "\n".join(lines)
