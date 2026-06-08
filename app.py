import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import savgol_filter

# ------------------------------------------------------------
# Public Battery Test Viewer
# Folder layout:
#   app.py
#   requirements.txt
#   data/
#     pack_index.csv
#     your_test_files.csv
#
# pack_index.csv columns:
#   pack,test,current_a,file,chemistry,series,parallel,cell,notes
# Required: pack,test,file
# Optional: current_a,chemistry,series,parallel,cell,notes
# ------------------------------------------------------------

APP_TITLE = "Battery Pack Test Results"
DATA_DIR = Path("data")
INDEX_FILE = DATA_DIR / "pack_index.csv"
PACK_INFO_FILE = DATA_DIR / "pack_info.csv"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Verified discharge test results for available battery packs.")

st.info("""
**Test Notes**

• Results are from individual sample packs and may vary slightly between packs.

• Capacity, voltage sag and temperature depend on ambient temperature, cooling airflow and discharge profile.

• All tests are done without airflow, so in real world use the packs will typically run much cooler and for longer at higher currents.

• Tests are performed using constant-current discharge and should be used for comparison purposes.

• Cells are discharged to 2.5V per cell, or until the cell temperature limit.
""")

# ----------------------------
# CSV loader: supports your logger format + ATORCH exports
# ----------------------------
def read_test_csv(file_bytes: bytes) -> pd.DataFrame:
    text = file_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    first_nonempty = next((ln for ln in lines if ln.strip()), "").strip().lower()

    keep = [
        "Voltage(V)",
        "Current(A)",
        "Power(W)",
        "mAh",
        "Wh",
        "mAh_calc",
        "Wh_calc",
        "Temperature(C)",
        "RestingVoltage(V)",
    ]

    # New logger CSV
    if first_nonempty.startswith("t_s,") or "iso_time" in first_nonempty:
        df = pd.read_csv(io.StringIO(text))
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

        if "iso_time" in df.columns:
            df["Time"] = pd.to_datetime(df["iso_time"], errors="coerce")
        elif "t_s" in df.columns:
            start = pd.Timestamp("2000-01-01")
            df["Time"] = start + pd.to_timedelta(pd.to_numeric(df["t_s"], errors="coerce"), unit="s")
        else:
            raise ValueError("CSV missing iso_time or t_s column.")

        mapping = {
            "voltage_v": "Voltage(V)",
            "current_a": "Current(A)",
            "power_w": "Power(W)",
            "mAh": "mAh",
            "Wh": "Wh",
            "mAh_calc": "mAh_calc",
            "Wh_calc": "Wh_calc",
            "temp_c": "Temperature(C)",
            "resting_voltage_v": "RestingVoltage(V)",
        }
        for src, dst in mapping.items():
            if src in df.columns:
                df[dst] = pd.to_numeric(df[src], errors="coerce")

        out = df[["Time"] + [c for c in keep if c in df.columns]].copy()
        return out.dropna(subset=["Time"]).sort_values("Time")

    # Old ATORCH format
    header_idx = None
    header_line = ""
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Time,") or s.startswith("Time\t"):
            header_idx = i
            header_line = s
            break
        if s.startswith("(时间)DATE,") or s.startswith("DATE,") or "(时间)DATE" in s:
            header_idx = i
            header_line = s
            break

    if header_idx is None:
        raise ValueError("Could not find a supported CSV header.")

    csv_text = "\n".join(lines[header_idx:])
    sep = "," if "," in header_line else "\t"
    df = pd.read_csv(io.StringIO(csv_text), sep=sep)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    time_col = None
    for cand in ["Time", "(时间)DATE", "DATE", "Date", "date"]:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        time_col = df.columns[0]

    df["Time"] = pd.to_datetime(df[time_col], format="%Y-%m-%d_%H:%M:%S", errors="coerce")
    if df["Time"].isna().all():
        df["Time"] = pd.to_datetime(df[time_col], errors="coerce")

    old_map = {
        "Voltage": "Voltage(V)",
        "Current": "Current(A)",
        "Power": "Power(W)",
        "Temperature": "Temperature(C)",
        "Temp": "Temperature(C)",
        "RestingVoltage": "RestingVoltage(V)",
        "(电压)VOLTAGE(V)": "Voltage(V)",
        "(电流)CURRENT(A)": "Current(A)",
        "(功率)POWER(W)": "Power(W)",
        "(容量)E_CAPACITY(mAh)": "mAh",
        "(电量)E_QUANTITY(Wh)": "Wh",
        "(探头温度)NTC_TEMP(℃)": "Temperature(C)",
    }
    for src, dst in old_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce")

    if "Current(A)" in df.columns:
        cur = pd.to_numeric(df["Current(A)"], errors="coerce")
        if np.nanmedian(cur) < 0:
            df["Current(A)"] = -cur

    out = df[["Time"] + [c for c in keep if c in df.columns]].copy()
    return out.dropna(subset=["Time"]).sort_values("Time")


def fmt_runtime(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def smooth_series(series: pd.Series, window: int = 21, poly: int = 3) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").ffill().bfill()
    n = len(s)
    if n < 7:
        return s
    w = window if window % 2 else window + 1
    if w > n:
        w = n if n % 2 else n - 1
    w = max(w, 5)
    p = min(poly, w - 1)
    return pd.Series(savgol_filter(s.to_numpy(), w, p, mode="interp"), index=series.index)


def elapsed_seconds(df: pd.DataFrame) -> np.ndarray:
    return (df["Time"] - df["Time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)


@st.cache_data(show_spinner=False)
def load_index() -> pd.DataFrame:
    if not INDEX_FILE.exists():
        return pd.DataFrame()
    idx = pd.read_csv(INDEX_FILE)
    required = {"pack", "test", "file"}
    missing = required - set(idx.columns)
    if missing:
        raise ValueError(f"pack_index.csv is missing columns: {', '.join(sorted(missing))}")
    return idx


@st.cache_data(show_spinner=False)
def load_pack_info() -> pd.DataFrame:
    """Load optional pack/cell specification data from data/pack_info.csv."""
    if not PACK_INFO_FILE.exists():
        return pd.DataFrame()
    info = pd.read_csv(PACK_INFO_FILE)
    if "pack" not in info.columns:
        raise ValueError("pack_info.csv is missing required column: pack")
    return info


@st.cache_data(show_spinner=False)
def load_test_file(relative_file: str) -> pd.DataFrame:
    path = Path(relative_file)
    if not path.is_absolute():
        path = Path(relative_file)
    if not path.exists():
        # Allow paths listed as just filename in pack_index.csv
        fallback = DATA_DIR / relative_file
        if fallback.exists():
            path = fallback
    with open(path, "rb") as f:
        return read_test_csv(f.read())


def summarize(df: pd.DataFrame) -> dict:
    secs = elapsed_seconds(df)
    summary = {"Runtime": fmt_runtime(float(np.nanmax(secs))) if len(secs) else "—"}

    if "mAh_calc" in df.columns and df["mAh_calc"].notna().any():
        summary["Delivered mAh"] = float(pd.to_numeric(df["mAh_calc"], errors="coerce").max())
    elif "mAh" in df.columns and df["mAh"].notna().any():
        summary["Delivered mAh"] = float(pd.to_numeric(df["mAh"], errors="coerce").max())

    if "Wh_calc" in df.columns and df["Wh_calc"].notna().any():
        summary["Delivered Wh"] = float(pd.to_numeric(df["Wh_calc"], errors="coerce").max())
    elif "Wh" in df.columns and df["Wh"].notna().any():
        summary["Delivered Wh"] = float(pd.to_numeric(df["Wh"], errors="coerce").max())

    if "Voltage(V)" in df.columns:
        v = pd.to_numeric(df["Voltage(V)"], errors="coerce")
        summary["Min voltage"] = float(v.min())
        summary["Start voltage"] = float(v.iloc[0])

    if "Temperature(C)" in df.columns:
        t = pd.to_numeric(df["Temperature(C)"], errors="coerce")
        summary["Max temp"] = float(t.max())

    if "Current(A)" in df.columns:
        c = pd.to_numeric(df["Current(A)"], errors="coerce")
        summary["Avg current"] = float(c.mean())

    return summary


def value_text(key: str, value) -> str:
    if value == "—" or pd.isna(value):
        return "—"
    if key == "Runtime":
        return str(value)
    if key == "Delivered mAh":
        return f"{value:.0f} mAh"
    if key == "Delivered Wh":
        return f"{value:.2f} Wh"
    if key in ["Min voltage", "Start voltage"]:
        return f"{value:.2f} V"
    if key == "Max temp":
        return f"{value:.1f} °C"
    if key == "Avg current":
        return f"{value:.1f} A"
    return str(value)


PLOT_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def trace_key(row) -> tuple[str, str]:
    return (str(row.get("pack", "Pack")), str(row.get("test", "Test")))


def build_trace_colors(rows: pd.DataFrame) -> dict:
    colours = {}
    for i, (_, row) in enumerate(rows.reset_index(drop=True).iterrows()):
        colours[trace_key(row)] = PLOT_COLORS[i % len(PLOT_COLORS)]
    return colours


def color_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
    hc = str(hex_color).lstrip("#")
    if len(hc) != 6:
        return f"rgba(255,255,255,{alpha})"
    r, g, b = int(hc[0:2], 16), int(hc[2:4], 16), int(hc[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def html_escape(value) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def metric_card_html(title: str, subtitle: str, metrics: list[tuple[str, str]], color: str) -> str:
    metric_blocks = ""
    for label, value in metrics:
        metric_blocks += (
            f'<div>'
            f'<div style="font-size: 0.78rem; opacity: 0.72; font-weight: 600;">{html_escape(label)}</div>'
            f'<div style="font-size: 1.65rem; line-height: 1.25; margin-top: 0.15rem;">{html_escape(value)}</div>'
            f'</div>'
        )

    # Keep the HTML left-aligned. Leading indentation can make Streamlit render it as a code block.
    return (
        f'<div style="'
        f'border: 1px solid rgba(255,255,255,0.14); '
        f'border-left: 7px solid {color}; '
        f'border-radius: 10px; '
        f'padding: 18px 18px 16px 18px; '
        f'margin: 0 0 14px 0; '
        f'background: linear-gradient(90deg, {color_to_rgba(color, 0.11)}, rgba(255,255,255,0.018) 34%);'
        f'">'
        f'<div style="font-size: 1.15rem; font-weight: 750; margin-bottom: 0.35rem;">{html_escape(title)}</div>'
        f'<div style="font-size: 0.82rem; opacity: 0.75; margin-bottom: 1.1rem;">{html_escape(subtitle)}</div>'
        f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 1.15rem 2rem;">'
        f'{metric_blocks}'
        f'</div>'
        f'</div>'
    )


try:
    index_df = load_index()
except Exception as e:
    st.error(str(e))
    st.stop()

if index_df.empty:
    st.warning("No data found. Create data/pack_index.csv and add your test CSVs to the data folder.")
    st.code(
        """pack,test,current_a,file,cell,notes
JP30 6S1P,20A run,20,data/JP30_6S1P_20A.csv,Ampace JP30,Room temp test
JP30 6S1P,50A run,50,data/JP30_6S1P_50A.csv,Ampace JP30,High current test""",
        language="csv",
    )
    st.stop()

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Select results")

    packs = sorted(index_df["pack"].dropna().astype(str).unique().tolist())

    selected_packs = st.multiselect(
        "Battery packs",
        packs,
        default=packs[:1],
        help="Select one pack for normal viewing, or multiple packs to compare cells/packs on the same graph.",
    )

    if not selected_packs:
        st.info("Select at least one battery pack.")
        st.stop()

    pack_rows = index_df[index_df["pack"].astype(str).isin(selected_packs)].copy()

    # Use unique test names across the selected packs. This works well when each pack
    # has matching test names like 20A run, 30A run, 50A run, etc.
    tests = sorted(pack_rows["test"].dropna().astype(str).unique().tolist())
    selected_tests = st.multiselect(
        "Tests",
        tests,
        default=tests[: min(3, len(tests))],
        help="When comparing packs, select the same current test, e.g. 20A run, to compare them directly.",
    )

    graph_options = {
        "Voltage": "Voltage(V)",
        "Temperature": "Temperature(C)",
        "Power": "Power(W)",
        "Current": "Current(A)",
        "Capacity": "mAh_calc",
        "Energy": "Wh_calc",
    }
    graph_type = st.selectbox("Primary graph", list(graph_options.keys()))

    secondary_options = ["None"] + list(graph_options.keys())
    graph_type_2 = st.selectbox(
        "Secondary graph",
        secondary_options,
        index=0,
        help="Optional second metric shown on the right axis, e.g. Temperature over Voltage.",
    )

    st.header("Display")
    # Smoothing removed for public viewer: show the real measured data by default.
    smooth = False
    show_markers = st.checkbox("Show points", value=False)
    show_summary = st.checkbox("Show summary cards", value=True)
    line_width = st.slider("Line width", 1.0, 5.0, 2.6, 0.1)
    chart_height = st.slider("Chart height", 450, 900, 650, 25)

if not selected_tests:
    st.info("Select at least one test.")
    st.stop()

selected_rows = pack_rows[pack_rows["test"].astype(str).isin(selected_tests)]

# ----------------------------
# Load selected tests
# ----------------------------
loaded = []
errors = []
for _, row in selected_rows.iterrows():
    try:
        df = load_test_file(str(row["file"]))
        loaded.append((row, df))
    except Exception as e:
        errors.append(f"{row.get('test', 'Unknown test')}: {e}")

if errors:
    st.error("Some tests failed to load:\n\n" + "\n".join(errors))

if not loaded:
    st.stop()

# Prefer calculated columns but fall back to raw logged columns
selected_col = graph_options[graph_type]
if selected_col == "mAh_calc":
    fallback_col = "mAh"
elif selected_col == "Wh_calc":
    fallback_col = "Wh"
else:
    fallback_col = selected_col

secondary_col = None
secondary_fallback_col = None
if graph_type_2 != "None":
    secondary_col = graph_options[graph_type_2]
    if secondary_col == "mAh_calc":
        secondary_fallback_col = "mAh"
    elif secondary_col == "Wh_calc":
        secondary_fallback_col = "Wh"
    else:
        secondary_fallback_col = secondary_col

trace_colors = build_trace_colors(selected_rows)

# ----------------------------
# Pack info
# ----------------------------
pack_title = selected_packs[0] if len(selected_packs) == 1 else "Pack comparison"
st.subheader(pack_title)

# Optional customer-facing pack/cell specs from data/pack_info.csv.
# Match rows by the "pack" column.
try:
    pack_info_df = load_pack_info()
except Exception as e:
    pack_info_df = pd.DataFrame()
    st.warning(f"Could not load pack_info.csv: {e}")

if not pack_info_df.empty:
    matching_info = pack_info_df[pack_info_df["pack"].astype(str).isin(selected_packs)].copy()

    if not matching_info.empty:
        st.markdown("### Pack / Cell Specifications")

        # Show one colour-coded spec box per selected pack so comparisons stay readable.
        for _, info in matching_info.iterrows():
            pack_name_for_info = str(info.get("pack", "Pack"))
            pack_colour = next(
                (
                    colour for (pack_name, _test_name), colour in trace_colors.items()
                    if pack_name == pack_name_for_info
                ),
                "#636EFA",
            )

            hidden_cols = {"pack"}
            long_text_cols = {"notes", "description", "summary", "datasheet_notes", "capabilities"}

            metric_items = []
            text_items = []

            for col in pack_info_df.columns:
                if col in hidden_cols:
                    continue

                value = info.get(col)
                if pd.isna(value):
                    continue

                label = str(col).replace("_", " ").title()
                value_text_display = str(value)

                if col.lower() in long_text_cols or len(value_text_display) > 80:
                    text_items.append((label, value_text_display))
                else:
                    metric_items.append((label, value_text_display))

            card_html = metric_card_html(
                title=pack_name_for_info,
                subtitle="Pack / cell specifications",
                metrics=metric_items,
                color=pack_colour,
            )
            st.markdown(card_html, unsafe_allow_html=True)

            for label, value in text_items:
                st.markdown(
                    f'<div style="border-left: 7px solid {pack_colour}; padding: 0.15rem 0 0.45rem 1rem; margin: -0.45rem 0 0.85rem 0; opacity: 0.95;">'
                    f'<b>{html_escape(label)}:</b> {html_escape(value)}</div>',
                    unsafe_allow_html=True,
                )

# Compact metadata from pack_index.csv, grouped per selected pack.
for pack_name in selected_packs:
    pack_meta_rows = selected_rows[selected_rows["pack"].astype(str) == str(pack_name)]
    if pack_meta_rows.empty:
        continue

    first_row = pack_meta_rows.iloc[0]
    meta_bits = []
    for col in ["cell", "chemistry", "series", "parallel"]:
        if col in selected_rows.columns and pd.notna(first_row.get(col)):
            meta_bits.append(f"**{col.title()}:** {first_row.get(col)}")

    if meta_bits:
        st.markdown(f"**{pack_name}:** " + "  |  ".join(meta_bits))

notes = selected_rows["notes"].dropna().unique().tolist() if "notes" in selected_rows.columns else []
if notes:
    with st.expander("Test notes"):
        for note in notes:
            st.write(f"- {note}")

# ----------------------------
# Summary cards
# ----------------------------
if show_summary:
    st.markdown("### Test summaries")
    card_order = ["Delivered mAh", "Delivered Wh", "Runtime", "Max temp", "Min voltage", "Avg current"]

    for row, df in loaded:
        row_colour = trace_colors.get(trace_key(row), "#636EFA")
        s = summarize(df)

        details = []
        if "current_a" in row and pd.notna(row.get("current_a")):
            details.append(f"Current: {row.get('current_a')}A")
        if "notes" in row and pd.notna(row.get("notes")):
            details.append(f"Notes: {row.get('notes')}")
        subtitle = "  |  ".join(details) if details else "Test result"

        metrics = [(key, value_text(key, s.get(key, "—"))) for key in card_order]

        st.markdown(
            metric_card_html(
                title=f"{row.get('pack', '')} — {row['test']}",
                subtitle=subtitle,
                metrics=metrics,
                color=row_colour,
            ),
            unsafe_allow_html=True,
        )

# ----------------------------
# Graph
# ----------------------------
fig = go.Figure()
mode = "lines+markers" if show_markers else "lines"

for row, df in loaded:
    col = selected_col if selected_col in df.columns else fallback_col
    if col not in df.columns:
        continue

    x = elapsed_seconds(df) / 60.0
    y = pd.to_numeric(df[col], errors="coerce")
    if smooth:
        y = smooth_series(y)

    label_base = f"{row.get('pack', 'Pack')} — {row['test']}"
    if "current_a" in row and pd.notna(row["current_a"]):
        label_base = f"{label_base} ({row['current_a']}A)"

    row_colour = trace_colors.get(trace_key(row), "#636EFA")

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=f"{label_base} — {graph_type}",
            legendgroup=label_base,
            line=dict(width=float(line_width), color=row_colour),
            hovertemplate=f"{label_base}<br>Time: %{{x:.2f}} min<br>{graph_type}: %{{y:.3f}}<extra></extra>",
            yaxis="y",
        )
    )

    if secondary_col is not None:
        col2 = secondary_col if secondary_col in df.columns else secondary_fallback_col
        if col2 in df.columns:
            y2 = pd.to_numeric(df[col2], errors="coerce")
            if smooth:
                y2 = smooth_series(y2)

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y2,
                    mode=mode,
                    name=f"{label_base} — {graph_type_2}",
                    legendgroup=label_base,
                    line=dict(width=max(1.0, float(line_width) * 0.78), color=row_colour, dash="dot"),
                    opacity=0.75,
                    hovertemplate=f"{label_base}<br>Time: %{{x:.2f}} min<br>{graph_type_2}: %{{y:.3f}}<extra></extra>",
                    yaxis="y2",
                )
            )

if not fig.data:
    st.warning(f"No selected files contain data for: {graph_type}")
else:
    y_titles = {
        "Voltage": "Voltage (V)",
        "Temperature": "Temperature (°C)",
        "Power": "Power (W)",
        "Current": "Current (A)",
        "Capacity": "Capacity (mAh)",
        "Energy": "Energy (Wh)",
    }

    chart_title = f"{pack_title} — {graph_type}"
    if graph_type_2 != "None":
        chart_title = f"{pack_title} — {graph_type} + {graph_type_2}"

    layout_kwargs = dict(
        title=chart_title,
        height=int(chart_height),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=70, r=70 if graph_type_2 != "None" else 30, t=90, b=70),
        xaxis=dict(title="Elapsed time (minutes)", showgrid=True),
        yaxis=dict(title=y_titles.get(graph_type, graph_type), showgrid=True),
    )

    if graph_type_2 != "None":
        layout_kwargs["yaxis2"] = dict(
            title=y_titles.get(graph_type_2, graph_type_2),
            overlaying="y",
            side="right",
            showgrid=False,
        )

    fig.update_layout(**layout_kwargs)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Raw data download
# ----------------------------
with st.expander("Download raw test files"):
    for row, _df in loaded:
        path = Path(str(row["file"]))
        if not path.exists():
            path = DATA_DIR / str(row["file"])
        if path.exists():
            st.download_button(
                label=f"Download {row.get('pack', 'Pack')} — {row['test']} CSV",
                data=path.read_bytes(),
                file_name=path.name,
                mime="text/csv",
            )
