import os

from dotenv import load_dotenv
from openai import OpenAI

from itm.config import MemoryConfig
from itm.patch import enable_memory

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# All 29 equations enabled — full ITM architecture
config = MemoryConfig(
    # Phase 2: Equations 12-17 (Kahneman + ACT-R)
    prospect_enabled=True,  # Eq 12: asymmetric strength (λ=2.25)
    tension_enabled=True,  # Eq 13: WYSIATI-breaking conflict detection
    channels_enabled=True,  # Eq 14: fact/preference/instruction/event
    adaptive_k_enabled=True,  # Eq 15: harder queries get more memories
    feedback_enabled=True,  # Eq 16: post-response edge learning
    actr_enabled=True,  # Eq 17: power-law decay (ACT-R)
    # Phase 3: Equations 18-21 (MemWire-Inspired)
    displacement_edges_enabled=True,  # Eq 18: ResNet displacement edges
    sparse_recall_enabled=True,  # Eq 19: BM25 hybrid dense+sparse
    multiscale_enabled=True,  # Eq 20: JK-Net multi-scale activation
    query_aware_spread_enabled=True,  # Eq 21: GAT query-aware spreading
    # Phase 4: Equations 22-28 (Research.md §14-18)
    hierarchy_enabled=True,  # Eq 22-23: L1→L2→L3 hierarchy + skip connections
    multichannel_enabled=True,  # Eq 24: multi-channel contradiction detection
    soft_decisions_enabled=True,  # Eq 25-27: sigmoid create-vs-strengthen
    cold_start_enabled=True,  # Eq 28: inverted warmup for first memories
)

# One line — memory is now active for all client.responses.create calls
mem = enable_memory(client, config)

print("GPT Chat (type 'quit' to exit)")
print("Commands: stats | memories | recall <query>")
print("-" * 50)

while 1:
    user_input = input("\nYou: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ("quit", "exit"):
        mem.flush()  # wait for any pending background updates
        mem.shutdown()
        print("Goodbye!")
        break

    # CLI commands — these bypass the LLM
    if user_input.lower() == "stats":
        mem.flush()  # ensure latest state
        print(f"\n{mem.stats.summary(mem.layer)}")
        continue

    if user_input.lower() == "memories":
        mem.flush()
        print(f"\n{mem.formatter.format_memory_list(mem.layer.memories)}")
        continue

    if user_input.lower().startswith("recall "):
        mem.flush()
        query = user_input[7:]
        print(f"\n{mem.stats.search(mem.layer, query, mem.embedder)}")
        continue

    # Normal chat — memory injection happens transparently
    # Response returns immediately; memory update runs in background thread
    response = client.responses.create(
        model="gpt-4.1-nano",
        instructions="You are a helpful coding assistant.",
        input=user_input,
    )

    # Show response immediately (memory update may still be processing)
    print(f"\nGPT: {response.output_text}")

    # Wait for background memory update, then show status
    mem.flush()

    action = mem.last_action
    if action:
        n = len(mem.layer.memories)
        t = mem.layer.timestep
        act = action["action"]
        details = action.get("details", "")
        if act == "filtered":
            print(f"  [Memory: filtered | {details} | memories: {n} | t={t}]")
        else:
            print(f"  [Memory: {act} | {details} | memories: {n} | t={t}]")
