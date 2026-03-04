from __future__ import annotations

from pathlib import Path

from .config import Settings
from .facts import FactStore
from .query import QueryAgent
from .storage import ArtifactStore
from .vector_store import VectorIndex


def _build_graph():
    settings = Settings(workspace_root=Path.cwd())
    store = ArtifactStore(settings)
    vector_index = VectorIndex(store)
    fact_store = FactStore(store.db_dir / "facts.db")
    agent = QueryAgent(store, vector_index, fact_store)
    graph = agent._build_langgraph_app()
    if graph is None:
        raise RuntimeError("LangGraph is not available. Install langgraph and enable REFINERY_QUERY_USE_LANGGRAPH=true")
    return graph


graph = _build_graph()
