from __future__ import annotations

def stack_graph_windows(graphs: list[dict], window_size: int = 5) -> list[dict]:
    windows = []
    for i in range(max(0, len(graphs) - window_size + 1)):
        chunk = graphs[i:i + window_size]
        windows.append({
            "start_frame": chunk[0].get("frame_id"),
            "end_frame": chunk[-1].get("frame_id"),
            "graphs": chunk,
            "window_size": window_size,
        })
    return windows
