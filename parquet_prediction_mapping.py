"""Plot the 30 longest voyages from the AIS feature parquet file."""

from pathlib import Path

import folium
import pandas as pd
from folium import plugins


# Data sources
TRACKS_PATH = Path(__file__).parent / "AIS_Dataset" / "filtered_trawlers.parquet"  # Dataset with voyage_id, timestamp, lat, lon
LABELS_PATH = Path(__file__).parent / "AIS_Label" / "labels_is_fishing.parquet"  # Fishing labels (-1, 0, 1)
TOP_N_VOYAGES = 30
OUTPUT_DIR = Path(__file__).parent / "AIS_Output"


def _get_output_path(labels_path: Path, top_n: int = TOP_N_VOYAGES) -> Path:
    """Generate output path: AIS_Output/{labels_parquet_name}_{top_n}.html"""
    labels_name = labels_path.stem  # e.g., "labels_is_fishing"
    return OUTPUT_DIR / f"{labels_name}_{top_n}.html"


def _get_label_column_name(labels_path: Path) -> str:
    """Get label column name from parquet filename: {name}_labels or {name}.parquet"""
    stem = labels_path.stem  # e.g., "labels_is_fishing" or "Random_Forest_Trawler_labels"
    return stem


REQUIRED_COLUMNS = {
    "voyage_id",
    "timestamp",
    "lat",
    "lon",
}


def _state_color(state: int) -> str:
    """Return the map color for the trawler fishing label."""
    return {1: "red", 0: "blue", -1: "gray"}.get(state, "gray")


def _add_colored_track(group: folium.FeatureGroup, track: pd.DataFrame, label_col: str) -> None:
    """Add continuous red/blue polylines split only when fishing state changes."""
    points = track[["lat", "lon"]].to_numpy(dtype=float)
    states = track[label_col].to_numpy(dtype=int)
    run_start = 0

    for index in range(len(points) - 1):
        state_changed = states[index + 1] != states[index]
        if state_changed:
            run_end = index + 2
            if run_end - run_start >= 2:
                color = _state_color(states[run_start])
                folium.PolyLine(
                    locations=points[run_start:run_end].tolist(),
                    color=color,
                    weight=2.5,
                    opacity=0.85,
                ).add_to(group)
            run_start = index + 1

    if len(points) - run_start >= 2:
        color = _state_color(states[run_start])
        folium.PolyLine(
            locations=points[run_start:].tolist(),
            color=color,
            weight=2.5,
            opacity=0.85,
        ).add_to(group)


def build_map(
    tracks_path: Path = TRACKS_PATH,
    labels_path: Path = LABELS_PATH,
    top_n: int = TOP_N_VOYAGES,
) -> list[str]:
    """Select the longest voyages and save their interactive Leaflet map."""
    output_path = _get_output_path(labels_path, top_n)
    label_col = _get_label_column_name(labels_path)
    data = pd.read_parquet(tracks_path, columns=sorted(REQUIRED_COLUMNS))
    labels = pd.read_parquet(labels_path)
    if len(data) != len(labels):
        raise ValueError(
            "Track and label Parquet files must have the same number of rows "
            "for positional label alignment."
        )
    data[label_col] = pd.to_numeric(labels.iloc[:, 0], errors="coerce").to_numpy()
    data["lat"] = pd.to_numeric(data["lat"], errors="coerce")
    data["lon"] = pd.to_numeric(data["lon"], errors="coerce")
    data["timestamp"] = pd.to_numeric(data["timestamp"], errors="coerce")
    data[label_col] = pd.to_numeric(data[label_col], errors="coerce")
    data = data.dropna(subset=[*REQUIRED_COLUMNS, label_col])
    data[label_col] = data[label_col].astype(int)
    data = data[data["lat"].between(-90, 90) & data["lon"].between(-180, 180)]
    if data.empty:
        raise ValueError("No valid trawler voyage rows remain after validation.")

    voyage_durations = data.groupby("voyage_id")["timestamp"].agg(
        lambda timestamps: timestamps.max() - timestamps.min()
    )
    voyage_ids = voyage_durations.nlargest(top_n).index.tolist()
    selected = data[data["voyage_id"].isin(voyage_ids)].sort_values(
        ["voyage_id", "timestamp"], kind="stable"
    )

    center = [float(selected["lat"].mean()), float(selected["lon"].mean())]
    leaflet_map = folium.Map(
        location=center,
        zoom_start=3,
        tiles="CartoDB positron",
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(leaflet_map)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Ocean",
        name="Esri Ocean",
    ).add_to(leaflet_map)

    legend = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 10px 12px; border: 1px solid #999;
                border-radius: 4px; font: 13px/1.5 Arial, sans-serif;
                box-shadow: 0 1px 4px rgba(0,0,0,.25);">
    <b>{label_col}</b><br>
    <span style="color: red; font-size: 20px;">&#8212;</span> 1: Fishing<br>
    <span style="color: blue; font-size: 20px;">&#8212;</span> 0: Not fishing<br>
    <span style="color: gray; font-size: 20px;">&#8212;</span> -1: Unknown
    </div>
    """
    leaflet_map.get_root().html.add_child(folium.Element(legend))

    for voyage_id, track in selected.groupby("voyage_id", sort=False):
        voyage_group = folium.FeatureGroup(
            name=f"Voyage {voyage_id} ({len(track):,} points)",
            show=True,
        )
        track = track.reset_index(drop=True)
        _add_colored_track(voyage_group, track, label_col)
        first, last = track.iloc[0], track.iloc[-1]
        folium.CircleMarker(
            [float(first["lat"]), float(first["lon"])],
            radius=3,
            color="green",
            fill=True,
            fill_color="green",
            tooltip=f"Start: {voyage_id}",
        ).add_to(voyage_group)
        folium.CircleMarker(
            [float(last["lat"]), float(last["lon"])],
            radius=3,
            color="black",
            fill=True,
            fill_color="black",
            tooltip=f"End: {voyage_id}",
        ).add_to(voyage_group)
        voyage_group.add_to(leaflet_map)

    folium.LayerControl(collapsed=False).add_to(leaflet_map)
    plugins.Fullscreen().add_to(leaflet_map)
    leaflet_map.fit_bounds(
        [
            [float(selected["lat"].min()), float(selected["lon"].min())],
            [float(selected["lat"].max()), float(selected["lon"].max())],
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    leaflet_map.save(output_path)
    print(f"Plotted {len(voyage_ids)} voyages and {len(selected):,} points")
    print(f"Map written to {output_path.resolve()}")
    return [str(voyage_id) for voyage_id in voyage_ids]


if __name__ == "__main__":
    build_map()
