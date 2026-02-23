import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime

DB_PATH = "match.db"

REQUIRED_COLS = ["nombre", "zona", "budget", "inicio", "fin"]

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            zona TEXT NOT NULL,
            budget REAL NOT NULL,
            inicio TEXT NOT NULL,
            fin TEXT NOT NULL,
            notas TEXT
        )
    """)
    conn.commit()
    conn.close()

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Falta la columna obligatoria: {col}")

    # Limpieza básica
    df["nombre"] = df["nombre"].astype(str).str.strip()
    df["zona"] = df["zona"].astype(str).str.strip()
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["inicio"] = pd.to_datetime(df["inicio"], errors="coerce").dt.date
    df["fin"] = pd.to_datetime(df["fin"], errors="coerce").dt.date

    if "notas" not in df.columns:
        df["notas"] = ""

    # Eliminar filas inválidas
    df = df.dropna(subset=["nombre", "zona", "budget", "inicio", "fin"])
    df = df[df["fin"] >= df["inicio"]]

    return df

def upsert_clients(df: pd.DataFrame):
    # MVP simple: insertamos todo (si quieres deduplicación, lo mejoramos por nombre+inicio+fin o un id externo)
    conn = get_conn()
    df2 = df.copy()
    df2["inicio"] = df2["inicio"].astype(str)
    df2["fin"] = df2["fin"].astype(str)
    df2[["nombre","zona","budget","inicio","fin","notas"]].to_sql("clientes", conn, if_exists="append", index=False)
    conn.close()

def load_clients() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM clientes", conn)
    conn.close()
    if df.empty:
        return df
    df["inicio"] = pd.to_datetime(df["inicio"]).dt.date
    df["fin"] = pd.to_datetime(df["fin"]).dt.date
    return df

def zones_set(zona: str):
    # Permite múltiples zonas separadas por | o , 
    parts = [z.strip().lower() for z in zona.replace(",", "|").split("|")]
    return set([p for p in parts if p])

def overlap_days(a_start, a_end, b_start, b_end) -> int:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if end < start:
        return 0
    return (end - start).days + 1

def budget_score(b1, b2, tolerance=0.20):
    # score 1 si están dentro de ±tolerance, cae linealmente hasta 0
    if b1 <= 0 or b2 <= 0:
        return 0.0
    ratio = abs(b1 - b2) / max(b1, b2)
    if ratio <= tolerance:
        return 1.0
    # penalización suave
    return max(0.0, 1.0 - (ratio - tolerance) / 0.50)  # a 70% diff => 0 aprox.

def date_score(days_overlap, min_days=30):
    # 1 si overlap >= min_days, si no, proporcional
    return min(1.0, days_overlap / float(min_days))

def zone_score(z1, z2):
    return 1.0 if len(z1.intersection(z2)) > 0 else 0.0

def compute_matches(df: pd.DataFrame, target_id: int, top_n=10,
                    w_zone=0.45, w_budget=0.25, w_dates=0.30,
                    min_overlap_days=30, require_zone=True):
    if df.empty:
        return df

    target = df[df["id"] == target_id].iloc[0]
    tz = zones_set(target["zona"])

    rows = []
    for _, other in df.iterrows():
        if int(other["id"]) == int(target_id):
            continue

        oz = zones_set(other["zona"])
        zs = zone_score(tz, oz)

        if require_zone and zs == 0:
            continue

        odays = overlap_days(target["inicio"], target["fin"], other["inicio"], other["fin"])
        ds = date_score(odays, min_days=min_overlap_days)
        bs = budget_score(target["budget"], other["budget"])

        score = (w_zone * zs) + (w_budget * bs) + (w_dates * ds)

        rows.append({
            "match_id": int(other["id"]),
            "nombre": other["nombre"],
            "zona": other["zona"],
            "budget": other["budget"],
            "inicio": other["inicio"],
            "fin": other["fin"],
            "overlap_dias": odays,
            "score_zona": round(zs, 2),
            "score_budget": round(bs, 2),
            "score_fechas": round(ds, 2),
            "score_total": round(score, 3),
            "notas": other.get("notas", "")
        })

    out = pd.DataFrame(rows).sort_values("score_total", ascending=False).head(top_n)
    return out

# ---------------- UI ----------------
st.set_page_config(page_title="Programa Match", layout="wide")
st.title("🏡 Programa Match — Clientes")

init_db()

with st.sidebar:
    st.header("Importar clientes")
    file = st.file_uploader("Sube tu Excel (.xlsx)", type=["xlsx"])
    sheet_name = st.text_input("Nombre de la hoja", value="clientes")
    if st.button("Importar a base de datos", type="primary"):
        if file is None:
            st.warning("Sube un Excel primero.")
        else:
            try:
                df_excel = pd.read_excel(file, sheet_name=sheet_name)
                df_clean = normalize_df(df_excel)
                upsert_clients(df_clean)
                st.success(f"Importados: {len(df_clean)} clientes.")
            except Exception as e:
                st.error(f"Error importando: {e}")

st.subheader("📋 Base de datos de clientes")
df = load_clients()
st.dataframe(df, use_container_width=True)

st.divider()

st.subheader("💞 Matching")
if df.empty:
    st.info("Importa clientes para empezar.")
else:
    col1, col2, col3, col4 = st.columns([2,1,1,1])

    with col1:
        target_id = st.selectbox(
            "Elige cliente",
            options=df["id"].tolist(),
            format_func=lambda x: f"{int(x)} — {df[df['id']==x].iloc[0]['nombre']}"
        )
    with col2:
        top_n = st.number_input("Top N", min_value=3, max_value=50, value=10)
    with col3:
        min_overlap_days = st.number_input("Mín. solape (días)", min_value=1, max_value=365, value=30)
    with col4:
        require_zone = st.checkbox("Exigir misma zona", value=True)

    w_zone = st.slider("Peso zona", 0.0, 1.0, 0.45)
    w_budget = st.slider("Peso budget", 0.0, 1.0, 0.25)
    w_dates = st.slider("Peso fechas", 0.0, 1.0, 0.30)

    # Normaliza pesos por si el usuario los cambia raro
    total_w = w_zone + w_budget + w_dates
    if total_w == 0:
        st.warning("Ajusta pesos (no pueden ser todos 0).")
    else:
        w_zone, w_budget, w_dates = w_zone/total_w, w_budget/total_w, w_dates/total_w

        matches = compute_matches(
            df, int(target_id),
            top_n=int(top_n),
            w_zone=w_zone, w_budget=w_budget, w_dates=w_dates,
            min_overlap_days=int(min_overlap_days),
            require_zone=require_zone
        )

        st.write("✅ Resultados")
        st.dataframe(matches, use_container_width=True)

        if not matches.empty:
            csv = matches.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar matches (CSV)", data=csv, file_name="matches.csv", mime="text/csv")
