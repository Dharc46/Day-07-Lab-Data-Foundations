from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.strategy_ui_server import ENGINE, STRATEGIES


def main() -> None:
    print("Precomputing strategy embedding cache...")
    for strategy in STRATEGIES:
        records, embeddings = ENGINE.load_index(strategy.key)
        print(f"{strategy.label}: {len(records)} chunks, embeddings={embeddings.shape}")
    print("Done. The web UI can now compare all strategies much faster.")


if __name__ == "__main__":
    main()
