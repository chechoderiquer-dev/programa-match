import streamlit as st
import pandas as pd
import sqlite3
import re

# ✅ NEW DB => clean start
DB_PATH = "match_v8.db"

VALID_GENERO = {"mujer", "hombre", "otro"}
VALID_PREF_GENERO = {"mixto", "solo_mujeres", "solo_hombres"}

REQUIRED_COLS = [
    "nombre", "telefono", "edad", "genero", "pref_genero", "idioma",
    "zona", "budget", "inicio", "fin",
    "max_compartir_con", "banos_min"
]


# ---------------- DB ----------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            telefono TEXT NOT NULL,
            pais TEXT,
            edad INTEGER NOT NULL,
            genero TEXT NOT NULL,
            pref_genero TEXT NOT NULL,
            idioma TEXT NOT NULL,

            zona TEXT NOT NULL,
            budget REAL NOT NULL,
            inicio TEXT NOT NULL,
            fin TEXT NOT NULL,

            max_compartir_con INTEGER NOT NULL,
            banos_min INTEGER NOT NULL,

            notas TEXT
        )
    """)
    conn.commit()
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


def upsert_clients(df: pd.DataFrame):
    conn = get_conn()
    df2 = df.copy()
    df2["inicio"] = df2["inicio"].astype(str)
    df2["fin"] = df2["fin"].astype(str)

    cols = [
        "nombre", "telefono", "pais", "edad", "genero", "pref_genero", "idioma",
        "zona", "budget", "inicio", "fin", "max_compartir_con", "banos_min", "notas"
    ]
    df2[cols].to_sql("clientes", conn, if_exists="append", index=False)
    conn.close()


# ---------------- Normalization ----------------
def normalize_phone(telefono: str) -> str:
    if telefono is None:
        return ""
    t = str(telefono).strip()
    return re.sub(r"[^\d\+]", "", t)  # keep + and digits


def detectar_pais(telefono: str) -> str:
    if not telefono:
        return "Unknown"
    t = str(telefono).strip()
    t = re.sub(r"[^\d\+]", "", t)

    if t.startswith("+34"):
        return "Spain"
    if t.startswith("+52"):
        return "Mexico"
    if t.startswith("+57"):
        return "Colombia"
    if t.startswith("+1"):
        return "USA/Canada"
    if t.startswith("+54"):
        return "Argentina"
    if t.startswith("+33"):
        return "France"
    if t.startswith("+39"):
        return "Italy"
    if t.startswith("+44"):
        return "UK"
    if t.startswith("+49"):
        return "Germany"
    if t.startswith("+31"):
        return "Netherlands"
    if t.startswith("+41"):
        return "Switzerland"
    if t.startswith("+351"):
        return "Portugal"
    if t.startswith("+353"):
        return "Ireland"
    return "Other"


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if "notas" not in df.columns:
        df["notas"] = ""

    df["nombre"] = df["nombre"].astype(str).str.strip()
    df["telefono"] = df["telefono"].apply(normalize_phone)
    df["pais"] = df["telefono"].apply(detectar_pais)

    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df["genero"] = df["genero"].astype(str).str.strip().str.lower()
    df["pref_genero"] = df["pref_genero"].astype(str).str.strip().str.lower()
    df["idioma"] = df["idioma"].astype(str).str.strip()

    df.loc[~df["genero"].isin(VALID_GENERO), "genero"] = "otro"
    df.loc[~df["pref_genero"].isin(VALID_PREF_GENERO), "pref_genero"] = "mixto"

    df["zona"] = df["zona"].astype(str).str.strip()
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["inicio"] = pd.to_datetime(df["inicio"], errors="coerce").dt.date
    df["fin"] = pd.to_datetime(df["fin"], errors="coerce").dt.date

    df["max_compartir_con"] = pd.to_numeric(df["max_compartir_con"], errors="coerce")
    df["banos_min"] = pd.to_numeric(df["banos_min"], errors="coerce")

    df["notas"] = df["notas"].astype(str).fillna("").str.strip()

    # Drop invalid rows
    df = df.dropna(subset=[
        "nombre", "telefono", "edad", "idioma",
        "zona", "budget", "inicio", "fin",
        "max_compartir_con", "banos_min"
    ])

    df = df[df["fin"] >= df["inicio"]]
    df = df[df["telefono"].str.startswith("+")]

    df["edad"] = df["edad"].astype(int)
    df["max_compartir_con"] = df["max_compartir_con"].astype(int)
    df["banos_min"] = df["banos_min"].astype(int)

    # Basic sanity
    df = df[(df["edad"] >= 18) & (df["edad"] <= 80)]
    df = df[(df["banos_min"] >= 1) & (df["max_compartir_con"] >= 0)]

    return df


# ---------------- Matching helpers ----------------
def zones_set(zona: str):
    parts = [z.strip().lower() for z in str(zona).replace(",", "|").split("|")]
    return set([p for p in parts if p])


def overlap_days(a_start, a_end, b_start, b_end) -> int:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if end < start:
        return 0
    return (end - start).days + 1


def zone_score(z1, z2) -> float:
    return 1.0 if len(z1.intersection(z2)) > 0 else 0.0


def budget_score(b1, b2, tolerance=0.20) -> float:
    if b1 <= 0 or b2 <= 0:
        return 0.0
    ratio = abs(b1 - b2) / max(b1, b2)
    if ratio <= tolerance:
        return 1.0
    return max(0.0, 1.0 - (ratio - tolerance) / 0.50)


def date_score(days_overlap, min_days=30) -> float:
    return min(1.0, days_overlap / float(min_days))


def share_score(s1, s2) -> float:
    try:
        s1, s2 = int(s1), int(s2)
        diff = abs(s1 - s2)
        return max(0.0, 1.0 - diff / 3.0)
    except Exception:
        return 0.5


def bath_score(b1, b2) -> float:
    try:
        d = abs(int(b1) - int(b2))
        if d == 0:
            return 1.0
        if d == 1:
            return 0.7
        return 0.3
    except Exception:
        return 0.5


def age_score(e1, e2) -> float:
    try:
        diff = abs(int(e1) - int(e2))
        return max(0.0, 1.0 - diff / 15.0)
    except Exception:
        return 0.5


def genero_compatible(pref_target: str, genero_other: str) -> bool:
    pref_target = (pref_target or "mixto").lower()
    genero_other = (genero_other or "otro").lower()

    if pref_target == "mixto":
        return True
    if pref_target == "solo_mujeres":
        return genero_other == "mujer"
    if pref_target == "solo_hombres":
        return genero_other == "hombre"
    return True


def match_genero_bidireccional(t: pd.Series, o: pd.Series) -> bool:
    return (
        genero_compatible(t["pref_genero"], o["genero"]) and
        genero_compatible(o["pref_genero"], t["genero"])
    )


def compute_matches(
    df: pd.DataFrame,
    target_id: int,
    top_n: int = 15,
    min_overlap_days: int = 30,
    require_zone: bool = True,
    country_filter: str = "All",
    language_filter: str = "All",
    weights=None
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if weights is None:
        # Must sum ~1.0 (we do not strictly require; just keep consistent)
        weights = {
            "zone": 0.25,
            "budget": 0.15,
            "dates": 0.20,
            "share": 0.10,
            "bath": 0.10,
            "age": 0.20,
        }

    target = df[df["id"] == target_id].iloc[0]
    tz = zones_set(target["zona"])

    rows = []
    for _, other in df.iterrows():
        if int(other["id"]) == int(target_id):
            continue

        # Filters
        if country_filter != "All" and other.get("pais", "") != country_filter:
            continue
        if language_filter != "All" and str(other.get("idioma", "")).strip() != language_filter:
            continue

        # Gender compatibility (hard filter)
        if not match_genero_bidireccional(target, other):
            continue

        # Zone
        oz = zones_set(other["zona"])
        zs = zone_score(tz, oz)
        if require_zone and zs == 0:
            continue

        # Dates
        odays = overlap_days(target["inicio"], target["fin"], other["inicio"], other["fin"])
        ds = date_score(odays, min_days=min_overlap_days)

        # Budget
        bs = budget_score(float(target["budget"]), float(other["budget"]))

        # Roomies
        ss = share_score(target["max_compartir_con"], other["max_compartir_con"])
        baths = bath_score(target["banos_min"], other["banos_min"])
        ages = age_score(target["edad"], other["edad"])

        # Total score (0..1-ish)
        score_total = (
            weights["zone"] * zs +
            weights["budget"] * bs +
            weights["dates"] * ds +
            weights["share"] * ss +
            weights["bath"] * baths +
            weights["age"] * ages
        )

        rows.append({
            "match_id": int(other["id"]),
            "match_name": other["nombre"],
            "match_phone": other.get("telefono", ""),
            "match_country": other.get("pais", ""),
            "match_language": other.get("idioma", ""),
            "match_age": int(other.get("edad", 0)),
            "match_gender": other.get("genero", ""),
            "match_pref_gender": other.get("pref_genero", ""),
            "match_zone": other.get("zona", ""),
            "match_budget": other.get("budget", 0),
            "match_start": other.get("inicio"),
            "match_end": other.get("fin"),
            "match_max_share": int(other.get("max_compartir_con", 0)),
            "match_min_bath": int(other.get("banos_min", 1)),
            "overlap_days": int(odays),

            # breakdown scores
            "score_zone": round(zs, 2),
            "score_budget": round(bs, 2),
            "score_dates": round(ds, 2),
            "score_share": round(ss, 2),
            "score_bath": round(baths, 2),
            "score_age": round(ages, 2),

            "score_total": round(score_total, 3),
            "notes": other.get("notas", "")
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("score_total", ascending=False).head(int(top_n))
    return out


# ---------------- WhatsApp message generator ----------------
def generate_whatsapp_intro(target: pd.Series, match_row: pd.Series) -> str:
    # English template (you can customize brand/name)
    return (
        f"Hi {match_row['match_name']}! 👋\n\n"
        f"I'm connecting you with someone who could be a great roommate match.\n\n"
        f"✅ Potential roommate: {target['nombre']} ({target['edad']}), {target['pais']} — {target['telefono']}\n"
        f"• Preferred areas: {target['zona']}\n"
        f"• Budget: €{int(target['budget'])}\n"
        f"• Dates: {target['inicio']} to {target['fin']}\n"
        f"• Language: {target['idioma']}\n\n"
        f"About you (from our database):\n"
        f"• Areas: {match_row['match_zone']}\n"
        f"• Budget: €{int(match_row['match_budget'])}\n"
        f"• Dates: {match_row['match_start']} to {match_row['match_end']}\n"
        f"• Language: {match_row['match_language']}\n\n"
        f"If you're both happy, you can message each other directly and set up a quick call 😊"
    )


# ---------------- UI ----------------
st.set_page_config(page_title="Programa Match", layout="wide")
st.title("🏡 Roommate Matching System (Advanced Scoring + WhatsApp Message)")

init_db()

with st.sidebar:
    st.header("Upload Clients")
    file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    sheet_name = st.text_input("Sheet name", value="clientes")

    st.caption("Required columns:")
    st.code(
        "nombre, telefono, edad, genero, pref_genero, idioma, zona, budget, inicio, fin, max_compartir_con, banos_min (notas optional)",
        language="text"
    )
    st.caption("genero: mujer | hombre | otro")
    st.caption("pref_genero: mixto | solo_mujeres | solo_hombres")
    st.caption("telefono must start with + (country code)")

    if st.button("Import to database", type="primary"):
        if file is None:
            st.warning("Please upload an Excel file first.")
        else:
            try:
                df_excel = pd.read_excel(file, sheet_name=sheet_name)
                df_clean = normalize_df(df_excel)
                upsert_clients(df_clean)
                st.success(f"Imported {len(df_clean)} clients.")
            except Exception as e:
                st.error(f"Import error: {e}")

st.subheader("📋 Clients Database")
df = load_clients()
st.dataframe(df, use_container_width=True)

st.divider()

st.subheader("💞 Matchmaking")

if df.empty:
    st.info("Upload clients to start matching.")
else:
    # Target selector
    colA, colB, colC, colD = st.columns([2, 1, 1, 1])

    with colA:
        target_id = st.selectbox(
            "Select client",
            options=df["id"].tolist(),
            format_func=lambda x: f"{int(x)} — {df[df['id'] == x].iloc[0]['nombre']} ({df[df['id'] == x].iloc[0].get('pais','')})"
        )

    with colB:
        top_n = st.number_input("Top N", min_value=3, max_value=100, value=15)

    with colC:
        min_overlap_days = st.number_input("Min overlap (days)", min_value=1, max_value=365, value=30)

    with colD:
        require_zone = st.checkbox("Require same zone", value=True)

    # Filters
    countries = ["All"] + sorted([c for c in df["pais"].dropna().unique().tolist() if str(c).strip() != ""])
    languages = ["All"] + sorted([l for l in df["idioma"].dropna().unique().tolist() if str(l).strip() != ""])

    col1, col2 = st.columns([1, 1])
    with col1:
        country_filter = st.selectbox("Filter by country", options=countries)
    with col2:
        language_filter = st.selectbox("Filter by language", options=languages)

    # Visible scoring weights (advanced)
    st.markdown("### Scoring weights (advanced)")
    wcol1, wcol2, wcol3, wcol4, wcol5, wcol6 = st.columns(6)
    with wcol1:
        w_zone = st.slider("Zone", 0.0, 0.6, 0.25, 0.05)
    with wcol2:
        w_budget = st.slider("Budget", 0.0, 0.6, 0.15, 0.05)
    with wcol3:
        w_dates = st.slider("Dates", 0.0, 0.6, 0.20, 0.05)
    with wcol4:
        w_share = st.slider("Sharing", 0.0, 0.6, 0.10, 0.05)
    with wcol5:
        w_bath = st.slider("Bathrooms", 0.0, 0.6, 0.10, 0.05)
    with wcol6:
        w_age = st.slider("Age", 0.0, 0.6, 0.20, 0.05)

    # Normalize weights to sum to 1 (so score_total stays comparable)
    total_w = w_zone + w_budget + w_dates + w_share + w_bath + w_age
    if total_w == 0:
        st.warning("Set at least one weight > 0.")
        weights = {"zone": 0, "budget": 0, "dates": 0, "share": 0, "bath": 0, "age": 0}
    else:
        weights = {
            "zone": w_zone / total_w,
            "budget": w_budget / total_w,
            "dates": w_dates / total_w,
            "share": w_share / total_w,
            "bath": w_bath / total_w,
            "age": w_age / total_w,
        }

    # Compute matches
    matches = compute_matches(
        df=df,
        target_id=int(target_id),
        top_n=int(top_n),
        min_overlap_days=int(min_overlap_days),
        require_zone=bool(require_zone),
        country_filter=country_filter,
        language_filter=language_filter,
        weights=weights
    )

    if matches.empty:
        st.info("No matches found with the current filters/rules.")
    else:
        st.markdown("### ✅ Matches (with advanced scoring breakdown)")
        st.dataframe(matches, use_container_width=True)

        # Download
        csv = matches.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download matches (CSV)", data=csv, file_name="matches_advanced.csv", mime="text/csv")

        st.divider()

        # WhatsApp message generator
        st.markdown("### 📲 Generate WhatsApp introduction message")
        match_choice = st.selectbox(
            "Select one match to generate a WhatsApp message",
            options=matches["match_id"].tolist(),
            format_func=lambda mid: f"{int(mid)} — {matches[matches['match_id']==mid].iloc[0]['match_name']} ({matches[matches['match_id']==mid].iloc[0]['match_country']})"
        )

        target = df[df["id"] == int(target_id)].iloc[0]
        match_row = matches[matches["match_id"] == int(match_choice)].iloc[0]

        msg = generate_whatsapp_intro(target, match_row)
        st.text_area("WhatsApp message (copy & paste)", value=msg, height=260)

        st.download_button(
            "⬇️ Download message as .txt",
            data=msg.encode("utf-8"),
            file_name="whatsapp_intro.txt",
            mime="text/plain"
        )
