import streamlit as st
import pandas as pd
import sqlite3
import re
import itertools
from urllib.parse import quote

DB_PATH = "match_v11.db"

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
            telefono TEXT NOT NULL UNIQUE,
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


def reset_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS clientes")
    conn.commit()
    conn.close()
    init_db()


def load_clients() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM clientes", conn)
    conn.close()
    if df.empty:
        return df
    df["inicio"] = pd.to_datetime(df["inicio"]).dt.date
    df["fin"] = pd.to_datetime(df["fin"]).dt.date
    return df


def upsert_clients(df: pd.DataFrame) -> tuple[int, int]:
    """
    Inserta o actualiza por 'telefono' (UNIQUE).
    Retorna (inserted_or_updated, total_rows_processed)
    """
    conn = get_conn()
    cur = conn.cursor()

    df2 = df.copy()
    df2["inicio"] = df2["inicio"].astype(str)
    df2["fin"] = df2["fin"].astype(str)

    rows = df2.to_dict(orient="records")
    processed = 0
    for r in rows:
        processed += 1
        cur.execute("""
            INSERT INTO clientes (
                nombre, telefono, pais, edad, genero, pref_genero, idioma,
                zona, budget, inicio, fin,
                max_compartir_con, banos_min, notas
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(telefono) DO UPDATE SET
                nombre=excluded.nombre,
                pais=excluded.pais,
                edad=excluded.edad,
                genero=excluded.genero,
                pref_genero=excluded.pref_genero,
                idioma=excluded.idioma,
                zona=excluded.zona,
                budget=excluded.budget,
                inicio=excluded.inicio,
                fin=excluded.fin,
                max_compartir_con=excluded.max_compartir_con,
                banos_min=excluded.banos_min,
                notas=excluded.notas
        """, (
            r.get("nombre", ""),
            r.get("telefono", ""),
            r.get("pais", ""),
            int(r.get("edad", 0)),
            r.get("genero", "otro"),
            r.get("pref_genero", "mixto"),
            r.get("idioma", ""),
            r.get("zona", ""),
            float(r.get("budget", 0)),
            r.get("inicio", ""),
            r.get("fin", ""),
            int(r.get("max_compartir_con", 0)),
            int(r.get("banos_min", 1)),
            r.get("notas", "")
        ))

    conn.commit()

    # SQLite no da fácil cuántos fueron inserts vs updates con ON CONFLICT.
    # Pero te devolvemos "processed".
    conn.close()
    return processed, processed


# ---------------- Normalization (Excel-proof phones) ----------------
def normalize_phone(telefono: str) -> str:
    if telefono is None:
        return ""
    t = str(telefono).strip()

    if "E+" in t.upper():
        try:
            t = str(int(float(t)))
        except Exception:
            pass

    t = re.sub(r"[^\d]", "", t)
    if not t:
        return ""

    country_codes = [
        "351", "353",
        "34", "52", "57", "54", "33", "39", "44", "49", "31", "41",
        "1"
    ]
    for cc in country_codes:
        if t.startswith(cc) and len(t) >= len(cc) + 7:
            return "+" + t

    if len(t) == 9:
        return "+34" + t

    if len(t) > 9:
        return "+" + t

    return ""


def detectar_pais(telefono: str) -> str:
    if not telefono:
        return "Unknown"
    if telefono.startswith("+34"):
        return "Spain"
    if telefono.startswith("+52"):
        return "Mexico"
    if telefono.startswith("+57"):
        return "Colombia"
    if telefono.startswith("+1"):
        return "USA/Canada"
    if telefono.startswith("+54"):
        return "Argentina"
    if telefono.startswith("+33"):
        return "France"
    if telefono.startswith("+39"):
        return "Italy"
    if telefono.startswith("+44"):
        return "UK"
    if telefono.startswith("+49"):
        return "Germany"
    if telefono.startswith("+31"):
        return "Netherlands"
    if telefono.startswith("+41"):
        return "Switzerland"
    if telefono.startswith("+351"):
        return "Portugal"
    if telefono.startswith("+353"):
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

    df = df.dropna(subset=[
        "nombre", "telefono", "edad", "idioma",
        "zona", "budget", "inicio", "fin",
        "max_compartir_con", "banos_min"
    ])

    df = df[df["telefono"].str.len() >= 10]
    df = df[df["fin"] >= df["inicio"]]

    df["edad"] = df["edad"].astype(int)
    df["max_compartir_con"] = df["max_compartir_con"].astype(int)
    df["banos_min"] = df["banos_min"].astype(int)

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


def score_pair(target: pd.Series, other: pd.Series, min_overlap_days: int, require_zone: bool, weights: dict):
    if not match_genero_bidireccional(target, other):
        return None

    tz = zones_set(target["zona"])
    oz = zones_set(other["zona"])
    zs = zone_score(tz, oz)
    if require_zone and zs == 0:
        return None

    odays = overlap_days(target["inicio"], target["fin"], other["inicio"], other["fin"])
    ds = date_score(odays, min_days=min_overlap_days)
    bs = budget_score(float(target["budget"]), float(other["budget"]))
    ss = share_score(target["max_compartir_con"], other["max_compartir_con"])
    baths = bath_score(target["banos_min"], other["banos_min"])
    ages = age_score(target["edad"], other["edad"])

    total = (
        weights["zone"] * zs +
        weights["budget"] * bs +
        weights["dates"] * ds +
        weights["share"] * ss +
        weights["bath"] * baths +
        weights["age"] * ages
    )
    return {
        "score_total": float(total),
        "score_zone": float(zs),
        "score_budget": float(bs),
        "score_dates": float(ds),
        "score_share": float(ss),
        "score_bath": float(baths),
        "score_age": float(ages),
        "overlap_days": int(odays),
    }


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
        weights = {"zone": 0.25, "budget": 0.15, "dates": 0.20, "share": 0.10, "bath": 0.10, "age": 0.20}

    target = df[df["id"] == target_id].iloc[0]
    rows = []

    for _, other in df.iterrows():
        if int(other["id"]) == int(target_id):
            continue

        if country_filter != "All" and other.get("pais", "") != country_filter:
            continue
        if language_filter != "All" and str(other.get("idioma", "")).strip() != language_filter:
            continue

        scored = score_pair(target, other, min_overlap_days, require_zone, weights)
        if scored is None:
            continue

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
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in scored.items()},
            "notes": other.get("notas", "")
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("score_total", ascending=False).head(int(top_n))
    return out


# ---------------- WhatsApp helpers ----------------
def wa_digits(phone: str) -> str:
    if not phone:
        return ""
    return re.sub(r"[^\d]", "", str(phone))


def generate_whatsapp_intro(target: pd.Series, match_row: pd.Series) -> str:
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


def whatsapp_web_link(phone: str, message: str) -> str:
    digits = wa_digits(phone)
    if not digits:
        return ""
    return f"https://wa.me/{digits}?text={quote(message)}"


# ---------------- Group matching (3–4 people) ----------------
def group_compatible(a: pd.Series, b: pd.Series) -> bool:
    return match_genero_bidireccional(a, b)


def group_score(members: list[pd.Series], weights: dict, min_overlap_days: int, require_zone: bool) -> float:
    pairs = list(itertools.combinations(members, 2))
    if not pairs:
        return 0.0
    scores = []
    for a, b in pairs:
        scored = score_pair(a, b, min_overlap_days, require_zone, weights)
        if scored is None:
            return -1.0
        scores.append(scored["score_total"])
    return float(sum(scores) / len(scores))


def generate_groups(
    df: pd.DataFrame,
    target_id: int,
    matches_df: pd.DataFrame,
    group_size: int,
    max_groups: int,
    weights: dict,
    min_overlap_days: int,
    require_zone: bool
) -> pd.DataFrame:
    if matches_df.empty:
        return pd.DataFrame()

    pool_ids = [int(x) for x in matches_df["match_id"].tolist()]
    members_map = {int(r["id"]): r for _, r in df[df["id"].isin([target_id] + pool_ids)].iterrows()}
    target = members_map.get(int(target_id))
    if target is None:
        return pd.DataFrame()

    combos = itertools.combinations(pool_ids, group_size - 1)

    rows = []
    for combo in combos:
        ids = [int(target_id)] + [int(x) for x in combo]
        members = [members_map[i] for i in ids if i in members_map]
        if len(members) != group_size:
            continue

        # Hard gender constraints for all pairs
        ok = True
        for a, b in itertools.combinations(members, 2):
            if not group_compatible(a, b):
                ok = False
                break
        if not ok:
            continue

        roommates_needed = group_size - 1
        if any(int(m["max_compartir_con"]) < roommates_needed for m in members):
            continue

        gscore = group_score(members, weights, min_overlap_days, require_zone)
        if gscore < 0:
            continue

        names = [f"{m['nombre']} ({m['edad']})" for m in members]
        phones = [str(m.get("telefono", "")) for m in members]
        countries = [str(m.get("pais", "")) for m in members]
        langs = sorted(set(str(m.get("idioma", "")).strip() for m in members if str(m.get("idioma", "")).strip()))

        rows.append({
            "group_size": group_size,
            "group_score": round(gscore, 3),
            "members": " | ".join(names),
            "phones": " | ".join(phones),
            "countries": " | ".join(countries),
            "languages": ", ".join(langs),
            "member_ids": ",".join(str(i) for i in ids),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("group_score", ascending=False).head(int(max_groups))
    return out


# ---------------- Dashboard (FIXED) ----------------
def dashboard(df: pd.DataFrame):
    st.subheader("📊 Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total clients", int(len(df)))
    with c2:
        st.metric("Countries", int(df["pais"].nunique()) if "pais" in df.columns else 0)
    with c3:
        st.metric("Languages", int(df["idioma"].nunique()) if "idioma" in df.columns else 0)
    with c4:
        st.metric("Avg age", round(df["edad"].mean(), 1) if "edad" in df.columns and len(df) else "—")

    if "zona" in df.columns and len(df):
        z = df["zona"].astype(str).str.replace(",", "|")
        z = z.str.split("|").explode().str.strip()
        z = z[z != ""]
        top_z = z.value_counts().head(10)
        st.markdown("**Top zones (Top 10)**")
        st.bar_chart(top_z)

    if "pais" in df.columns and len(df):
        st.markdown("**Clients by country**")
        st.bar_chart(df["pais"].value_counts())

    if "idioma" in df.columns and len(df):
        st.markdown("**Clients by language**")
        st.bar_chart(df["idioma"].value_counts())

    # ✅ FIX Interval -> string
    if "edad" in df.columns and len(df):
        st.markdown("**Age distribution**")
        age_bins = pd.cut(df["edad"], bins=[17, 20, 25, 30, 35, 40, 50, 80], right=True)
        counts = age_bins.value_counts().sort_index()
        counts.index = counts.index.astype(str)
        st.bar_chart(counts)


# ---------------- UI ----------------
st.set_page_config(page_title="Programa Match", layout="wide")
st.title("🏡 Roommate Matching System — Dashboard + Groups + WhatsApp Export")

init_db()

with st.sidebar:
    st.header("Upload Clients")
    file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    sheet_name = st.text_input("Sheet name", value="clientes")

    if st.button("Reset database (delete all)", type="secondary"):
        reset_db()
        st.success("Database reset. Upload again.")

    if st.button("Import to database", type="primary"):
        if file is None:
            st.warning("Please upload an Excel file first.")
        else:
            try:
                df_excel = pd.read_excel(file, sheet_name=sheet_name)
                df_clean = normalize_df(df_excel)
                processed, _ = upsert_clients(df_clean)
                st.success(f"Upserted {processed} clients (no duplicates by phone).")
            except Exception as e:
                st.error(f"Import error: {e}")

df = load_clients()

tab1, tab2, tab3 = st.tabs(["Dashboard", "Match (1-to-1)", "Group Match (3–4)"])

with tab1:
    if df.empty:
        st.info("Upload clients to see the dashboard.")
    else:
        dashboard(df)

with tab2:
    st.subheader("💞 Matchmaking (1-to-1)")
    if df.empty:
        st.info("Upload clients to start matching.")
    else:
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

        countries = ["All"] + sorted([c for c in df["pais"].dropna().unique().tolist() if str(c).strip() != ""])
        languages = ["All"] + sorted([l for l in df["idioma"].dropna().unique().tolist() if str(l).strip() != ""])

        f1, f2 = st.columns([1, 1])
        with f1:
            country_filter = st.selectbox("Filter by country", options=countries)
        with f2:
            language_filter = st.selectbox("Filter by language", options=languages)

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
            st.markdown("### ✅ Matches (advanced score breakdown)")
            st.dataframe(matches, use_container_width=True)

            target = df[df["id"] == int(target_id)].iloc[0]
            wa_rows = []
            for _, r in matches.iterrows():
                msg = generate_whatsapp_intro(target, r)
                link = whatsapp_web_link(r.get("match_phone", ""), msg)
                wa_rows.append({
                    "match_id": r["match_id"],
                    "match_name": r["match_name"],
                    "match_phone": r.get("match_phone", ""),
                    "whatsapp_link": link
                })
            wa_df = pd.DataFrame(wa_rows)

            st.markdown("### 📲 WhatsApp Web links (click to open)")
            st.dataframe(wa_df, use_container_width=True)

with tab3:
    st.subheader("👥 Group Matching (3–4 people)")
    if df.empty:
        st.info("Upload clients to start group matching.")
    else:
        colA, colB, colC, colD = st.columns([2, 1, 1, 1])
        with colA:
            target_id_g = st.selectbox(
                "Select client (group anchor)",
                options=df["id"].tolist(),
                format_func=lambda x: f"{int(x)} — {df[df['id'] == x].iloc[0]['nombre']} ({df[df['id'] == x].iloc[0].get('pais','')})",
                key="target_group"
            )
        with colB:
            pool_top = st.number_input("Pool size (from top matches)", min_value=10, max_value=100, value=30)
        with colC:
            group_size = st.selectbox("Group size", options=[3, 4], index=1)
        with colD:
            max_groups = st.number_input("Max groups to show", min_value=5, max_value=100, value=20)

        min_overlap_days_g = st.number_input("Min overlap (days)", min_value=1, max_value=365, value=30, key="min_overlap_group")
        require_zone_g = st.checkbox("Require same zone", value=True, key="req_zone_group")

        pool_matches = compute_matches(
            df=df,
            target_id=int(target_id_g),
            top_n=int(pool_top),
            min_overlap_days=int(min_overlap_days_g),
            require_zone=bool(require_zone_g),
            weights={"zone": 0.25, "budget": 0.15, "dates": 0.20, "share": 0.10, "bath": 0.10, "age": 0.20}
        )

        groups = generate_groups(
            df=df,
            target_id=int(target_id_g),
            matches_df=pool_matches,
            group_size=int(group_size),
            max_groups=int(max_groups),
            weights={"zone": 0.25, "budget": 0.15, "dates": 0.20, "share": 0.10, "bath": 0.10, "age": 0.20},
            min_overlap_days=int(min_overlap_days_g),
            require_zone=bool(require_zone_g)
        )

        if groups.empty:
            st.info("No compatible groups found from the current pool.")
        else:
            st.markdown("### ✅ Best groups")
            st.dataframe(groups, use_container_width=True)
