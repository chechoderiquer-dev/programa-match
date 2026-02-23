import streamlit as st
import pandas as pd
import sqlite3
import re
from datetime import date

# ✅ Cambia este nombre si quieres "resetear" la base otra vez
DB_PATH = "match_v5.db"

VALID_PREF = {"mixto", "solo_ninas", "solo_ninos"}

# Columnas obligatorias (incluye teléfono + edad)
REQUIRED_COLS = [
    "nombre", "zona", "budget", "inicio", "fin",
    "max_compartir_con", "banos_min", "preferencia",
    "telefono", "edad"
]


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
            max_compartir_con INTEGER NOT NULL,
            banos_min INTEGER NOT NULL,
            preferencia TEXT NOT NULL,
            telefono TEXT NOT NULL,
            pais TEXT,
            edad INTEGER NOT NULL,
            notas TEXT
        )
    """)
    conn.commit()
    conn.close()


def detectar_pais(telefono: str) -> str:
    if telefono is None:
        return "Desconocido"
    t = str(telefono).strip()

    # Normaliza: deja + y dígitos
    t = re.sub(r"[^\d\+]", "", t)

    # Detección por prefijos comunes (puedes ampliar cuando quieras)
    if t.startswith("+34"):
        return "España"
    if t.startswith("+52"):
        return "México"
    if t.startswith("+57"):
        return "Colombia"
    if t.startswith("+1"):
        return "USA/Canadá"
    if t.startswith("+54"):
        return "Argentina"
    if t.startswith("+33"):
        return "Francia"
    if t.startswith("+39"):
        return "Italia"
    if t.startswith("+44"):
        return "Reino Unido"
    if t.startswith("+49"):
        return "Alemania"
    if t.startswith("+31"):
        return "Países Bajos"
    if t.startswith("+41"):
        return "Suiza"
    if t.startswith("+351"):
        return "Portugal"
    if t.startswith("+353"):
        return "Irlanda"
    return "Otro"


def normalize_phone(telefono: str) -> str:
    if telefono is None:
        return ""
    t = str(telefono).strip()
    t = re.sub(r"[^\d\+]", "", t)
    return t


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Falta la columna obligatoria: {col}")

    if "notas" not in df.columns:
        df["notas"] = ""

    # Limpieza básica
    df["nombre"] = df["nombre"].astype(str).str.strip()
    df["zona"] = df["zona"].astype(str).str.strip()
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")

    df["inicio"] = pd.to_datetime(df["inicio"], errors="coerce").dt.date
    df["fin"] = pd.to_datetime(df["fin"], errors="coerce").dt.date

    df["max_compartir_con"] = pd.to_numeric(df["max_compartir_con"], errors="coerce")
    df["banos_min"] = pd.to_numeric(df["banos_min"], errors="coerce")

    df["preferencia"] = df["preferencia"].astype(str).str.strip().str.lower()
    df.loc[~df["preferencia"].isin(VALID_PREF), "preferencia"] = "mixto"

    df["telefono"] = df["telefono"].apply(normalize_phone)
    df["pais"] = df["telefono"].apply(detectar_pais)

    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")

    df["notas"] = df["notas"].astype(str).fillna("").str.strip()

    # Eliminar filas inválidas
    df = df.dropna(subset=[
        "nombre", "zona", "budget", "inicio", "fin",
        "max_compartir_con", "banos_min", "telefono", "edad"
    ])

    df = df[df["fin"] >= df["inicio"]]

    # Cast a int + filtros lógicos
    df["max_compartir_con"] = df["max_compartir_con"].astype(int)
    df["banos_min"] = df["banos_min"].astype(int)
    df["edad"] = df["edad"].astype(int)

    df = df[(df["max_compartir_con"] >= 0) & (df["banos_min"] >= 1)]
    df = df[(df["edad"] >= 18) & (df["edad"] <= 80)]
    df = df[df["telefono"].str.startswith("+")]  # obliga prefijo país

    return df


def upsert_clients(df: pd.DataFrame):
    conn = get_conn()
    df2 = df.copy()
    df2["inicio"] = df2["inicio"].astype(str)
    df2["fin"] = df2["fin"].astype(str)

    df2[[
        "nombre", "zona", "budget", "inicio", "fin",
        "max_compartir_con", "banos_min", "preferencia",
        "telefono", "pais", "edad", "notas"
    ]].to_sql("clientes", conn, if_exists="append", index=False)

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
    parts = [z.strip().lower() for z in str(zona).replace(",", "|").split("|")]
    return set([p for p in parts if p])


def overlap_days(a_start, a_end, b_start, b_end) -> int:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if end < start:
        return 0
    return (end - start).days + 1


def budget_score(b1, b2, tolerance=0.20):
    if b1 <= 0 or b2 <= 0:
        return 0.0
    ratio = abs(b1 - b2) / max(b1, b2)
    if ratio <= tolerance:
        return 1.0
    return max(0.0, 1.0 - (ratio - tolerance) / 0.50)


def date_score(days_overlap, min_days=30):
    return min(1.0, days_overlap / float(min_days))


def zone_score(z1, z2):
    return 1.0 if len(z1.intersection(z2)) > 0 else 0.0


def preference_compatible(p1: str, p2: str) -> bool:
    p1 = (p1 or "mixto").lower()
    p2 = (p2 or "mixto").lower()
    if p1 == "mixto" or p2 == "mixto":
        return True
    return p1 == p2


def share_score(s1, s2):
    try:
        s1, s2 = int(s1), int(s2)
        diff = abs(s1 - s2)
        return max(0.0, 1.0 - diff / 3.0)  # diff 0 => 1, diff 3 => 0
    except Exception:
        return 0.5


def bath_score(b1, b2):
    try:
        d = abs(int(b1) - int(b2))
        if d == 0:
            return 1.0
        if d == 1:
            return 0.7
        return 0.3
    except Exception:
        return 0.5


def age_score(e1, e2):
    try:
        diff = abs(int(e1) - int(e2))
        return max(0.0, 1.0 - diff / 15.0)  # 15 años => score bajo
    except Exception:
        return 0.5


def compute_matches(
    df: pd.DataFrame,
    target_id: int,
    top_n=10,
    min_overlap_days=30,
    require_zone=True,
):
    if df.empty:
        return df

    target = df[df["id"] == target_id].iloc[0]
    tz = zones_set(target["zona"])

    rows = []
    for _, other in df.iterrows():
        if int(other["id"]) == int(target_id):
            continue

        # Preferencia (filtro duro)
        if not preference_compatible(target.get("preferencia"), other.get("preferencia")):
            continue

        # Zona
        oz = zones_set(other["zona"])
        zs = zone_score(tz, oz)
        if require_zone and zs == 0:
            continue

        # Fechas
        odays = overlap_days(target["inicio"], target["fin"], other["inicio"], other["fin"])
        ds = date_score(odays, min_days=min_overlap_days)

        # Budget
        bs = budget_score(float(target["budget"]), float(other["budget"]))

        # Roomies
        ss = share_score(target.get("max_compartir_con"), other.get("max_compartir_con"))
        baths = bath_score(target.get("banos_min"), other.get("banos_min"))
        age_s = age_score(target.get("edad"), other.get("edad"))

        # Score final (pesos)
        score = (
            0.25 * zs +
            0.15 * bs +
            0.20 * ds +
            0.10 * ss +
            0.10 * baths +
            0.20 * age_s
        )

        rows.append({
            "match_id": int(other["id"]),
            "nombre": other["nombre"],
            "telefono": other.get("telefono", ""),
            "pais": other.get("pais", ""),
            "edad": int(other.get("edad", 0)),
            "zona": other["zona"],
            "budget": other["budget"],
            "inicio": other["inicio"],
            "fin": other["fin"],
            "preferencia": other.get("preferencia", "mixto"),
            "max_compartir_con": int(other.get("max_compartir_con", 0)),
            "banos_min": int(other.get("banos_min", 1)),
            "overlap_dias": odays,
            "score_zona": round(zs, 2),
            "score_budget": round(bs, 2),
            "score_fechas": round(ds, 2),
            "score_compartir": round(ss, 2),
            "score_banos": round(baths, 2),
            "score_edad": round(age_s, 2),
            "score_total": round(score, 3),
            "notas": other.get("notas", "")
        })

    out = pd.DataFrame(rows).sort_values("score_total", ascending=False).head(top_n)
    return out


# ---------------- UI ----------------
st.set_page_config(page_title="Programa Match", layout="wide")
st.title("🏡 Programa Match — Clientes (Roomies)")

init_db()

with st.sidebar:
    st.header("Importar clientes")
    file = st.file_uploader("Sube tu Excel (.xlsx)", type=["xlsx"])
    sheet_name = st.text_input("Nombre de la hoja", value="clientes")

    st.caption("Columnas obligatorias:")
    st.code(
        "nombre, zona, budget, inicio, fin, max_compartir_con, banos_min, preferencia, telefono, edad, (notas opcional)",
        language="text"
    )
    st.caption("preferencia: mixto | solo_ninas | solo_ninos")
    st.caption("telefono: con prefijo país (ej: +34...)")

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
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        target_id = st.selectbox(
            "Elige cliente",
            options=df["id"].tolist(),
            format_func=lambda x: f"{int(x)} — {df[df['id'] == x].iloc[0]['nombre']} ({df[df['id'] == x].iloc[0].get('pais','')})"
        )

    with col2:
        top_n = st.number_input("Top N", min_value=3, max_value=50, value=10)

    with col3:
        min_overlap_days = st.number_input("Mín. solape (días)", min_value=1, max_value=365, value=30)

    with col4:
        require_zone = st.checkbox("Exigir misma zona", value=True)

    matches = compute_matches(
        df=df,
        target_id=int(target_id),
        top_n=int(top_n),
        min_overlap_days=int(min_overlap_days),
        require_zone=require_zone
    )

    st.write("✅ Resultados")
    st.dataframe(matches, use_container_width=True)

    if not matches.empty:
        csv = matches.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar matches (CSV)", data=csv, file_name="matches.csv", mime="text/csv")
