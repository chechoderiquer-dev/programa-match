import streamlit as st
import pandas as pd
import sqlite3
import re
from urllib.parse import quote

DB_PATH = "match_v12_excel.db"

# =========================================================
# CONFIG / NORMALIZATION FOR YOUR REAL EXCEL
# =========================================================
RAW_TO_STD_COLS = {
    "id": "source_id",
    "created_at": "created_at",
    "telefono": "telefono",
    "telefono_raw": "telefono_raw",
    "telefono_region": "telefono_region",
    "nombre": "nombre",
    "edad": "edad",
    "genero": "genero",
    "pref_genero": "pref_genero",
    "idioma": "idioma",
    "zona": "zona",
    "budget": "budget",
    "inicio": "inicio",
    "fin": "fin",
    "perfil": "perfil",
    "Perfil": "perfil",
    "banos_min": "banos_min",
    "habitaciones": "habitaciones",
    "notas": "notas",
    "zonas_preferidas": "zonas_preferidas",
    "pais_origen": "pais_origen",
    "pais_origen_codigo": "pais_origen_codigo",
    "estilo_vida": "estilo_vida",
    "urgencia_mudanza": "urgencia_mudanza",
    "urgencia_mudanza_codigo": "urgencia_mudanza_codigo",
    "busqueda_vivienda": "busqueda_vivienda",
    "busqueda_vivienda_codigo": "busqueda_vivienda_codigo",
    "teletrabajo": "teletrabajo",
    "zona_otra": "zona_otra",
    "preferencia_rutina_hogar": "preferencia_rutina_hogar",
}

REQUIRED_COLS = [
    "nombre", "telefono", "edad", "genero", "pref_genero",
    "idioma", "zona", "budget", "inicio", "fin"
]

VALID_GENERO = {"mujer", "hombre", "otro"}
VALID_PREF_GENERO = {"mixto", "solo_mujeres", "solo_hombres"}

GENERO_MAP = {
    "mujer": "mujer",
    "woman": "mujer",
    "female": "mujer",
    "hombre": "hombre",
    "man": "hombre",
    "male": "hombre",
    "otro": "otro",
    "other": "otro",
    "non-binary": "otro",
    "non binary": "otro",
}

PREF_GENERO_MAP = {
    "mixto": "mixto",
    "mixed": "mixto",
    "any": "mixto",
    "all": "mixto",
    "solo mujeres": "solo_mujeres",
    "solo_mujeres": "solo_mujeres",
    "women only": "solo_mujeres",
    "female only": "solo_mujeres",
    "girls only": "solo_mujeres",
    "solo hombres": "solo_hombres",
    "solo_hombres": "solo_hombres",
    "men only": "solo_hombres",
    "male only": "solo_hombres",
    "boys only": "solo_hombres",
}

# =========================================================
# DB
# =========================================================
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS clientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT,
            created_at TEXT,
            nombre TEXT NOT NULL,
            telefono TEXT NOT NULL UNIQUE,
            telefono_raw TEXT,
            telefono_region TEXT,
            pais TEXT,
            pais_origen TEXT,
            pais_origen_codigo TEXT,
            edad INTEGER NOT NULL,
            genero TEXT NOT NULL,
            pref_genero TEXT NOT NULL,
            idioma TEXT NOT NULL,
            zona TEXT NOT NULL,
            zonas_preferidas TEXT,
            zona_otra TEXT,
            budget REAL NOT NULL,
            inicio TEXT NOT NULL,
            fin TEXT NOT NULL,
            perfil TEXT,
            banos_min INTEGER,
            habitaciones INTEGER,
            estilo_vida TEXT,
            urgencia_mudanza TEXT,
            urgencia_mudanza_codigo TEXT,
            busqueda_vivienda TEXT,
            busqueda_vivienda_codigo TEXT,
            teletrabajo TEXT,
            preferencia_rutina_hogar TEXT,
            notas TEXT
        )
        """
    )
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

    for col in ["inicio", "fin", "created_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "inicio" in df.columns:
        df["inicio"] = df["inicio"].dt.date
    if "fin" in df.columns:
        df["fin"] = df["fin"].dt.date

    return df


def upsert_clients(df: pd.DataFrame) -> tuple[int, int]:
    conn = get_conn()
    cur = conn.cursor()

    df2 = df.copy()
    for c in ["created_at", "inicio", "fin"]:
        if c in df2.columns:
            df2[c] = df2[c].astype(str)

    rows = df2.to_dict(orient="records")
    processed = 0

    for r in rows:
        processed += 1
        cur.execute(
            """
            INSERT INTO clientes (
                source_id, created_at, nombre, telefono, telefono_raw, telefono_region,
                pais, pais_origen, pais_origen_codigo,
                edad, genero, pref_genero, idioma,
                zona, zonas_preferidas, zona_otra,
                budget, inicio, fin,
                perfil, banos_min, habitaciones,
                estilo_vida, urgencia_mudanza, urgencia_mudanza_codigo,
                busqueda_vivienda, busqueda_vivienda_codigo,
                teletrabajo, preferencia_rutina_hogar,
                notas
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(telefono) DO UPDATE SET
                source_id=excluded.source_id,
                created_at=excluded.created_at,
                nombre=excluded.nombre,
                telefono_raw=excluded.telefono_raw,
                telefono_region=excluded.telefono_region,
                pais=excluded.pais,
                pais_origen=excluded.pais_origen,
                pais_origen_codigo=excluded.pais_origen_codigo,
                edad=excluded.edad,
                genero=excluded.genero,
                pref_genero=excluded.pref_genero,
                idioma=excluded.idioma,
                zona=excluded.zona,
                zonas_preferidas=excluded.zonas_preferidas,
                zona_otra=excluded.zona_otra,
                budget=excluded.budget,
                inicio=excluded.inicio,
                fin=excluded.fin,
                perfil=excluded.perfil,
                banos_min=excluded.banos_min,
                habitaciones=excluded.habitaciones,
                estilo_vida=excluded.estilo_vida,
                urgencia_mudanza=excluded.urgencia_mudanza,
                urgencia_mudanza_codigo=excluded.urgencia_mudanza_codigo,
                busqueda_vivienda=excluded.busqueda_vivienda,
                busqueda_vivienda_codigo=excluded.busqueda_vivienda_codigo,
                teletrabajo=excluded.teletrabajo,
                preferencia_rutina_hogar=excluded.preferencia_rutina_hogar,
                notas=excluded.notas
            """,
            (
                str(r.get("source_id", "")),
                r.get("created_at", ""),
                r.get("nombre", ""),
                r.get("telefono", ""),
                r.get("telefono_raw", ""),
                r.get("telefono_region", ""),
                r.get("pais", ""),
                r.get("pais_origen", ""),
                r.get("pais_origen_codigo", ""),
                int(r.get("edad", 0)),
                r.get("genero", "otro"),
                r.get("pref_genero", "mixto"),
                r.get("idioma", ""),
                r.get("zona", ""),
                r.get("zonas_preferidas", ""),
                r.get("zona_otra", ""),
                float(r.get("budget", 0)),
                r.get("inicio", ""),
                r.get("fin", ""),
                r.get("perfil", ""),
                int(r.get("banos_min", 1)) if pd.notna(r.get("banos_min")) else 1,
                int(r.get("habitaciones", 0)) if pd.notna(r.get("habitaciones")) else 0,
                r.get("estilo_vida", ""),
                r.get("urgencia_mudanza", ""),
                r.get("urgencia_mudanza_codigo", ""),
                r.get("busqueda_vivienda", ""),
                r.get("busqueda_vivienda_codigo", ""),
                r.get("teletrabajo", ""),
                r.get("preferencia_rutina_hogar", ""),
                r.get("notas", ""),
            ),
        )

    conn.commit()
    conn.close()
    return processed, processed


# =========================================================
# NORMALIZATION
# =========================================================
def normalize_phone(telefono) -> str:
    if telefono is None or (isinstance(telefono, float) and pd.isna(telefono)):
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
        "351", "353", "34", "52", "57", "54", "33", "39", "44", "49", "31", "41", "1"
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


def normalize_genero(value: str) -> str:
    v = str(value).strip().lower()
    return GENERO_MAP.get(v, "otro")


def normalize_pref_genero(value: str) -> str:
    v = str(value).strip().lower()
    return PREF_GENERO_MAP.get(v, "mixto")


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def first_non_empty_sheet(uploaded_file) -> str:
    xls = pd.ExcelFile(uploaded_file)
    for sheet in xls.sheet_names:
        try:
            test = pd.read_excel(uploaded_file, sheet_name=sheet, nrows=5)
            if not test.empty and len(test.columns) > 0:
                return sheet
        except Exception:
            continue
    return xls.sheet_names[0]


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    original_cols = list(df.columns)
    lowered_map = {c: str(c).strip().lower() for c in original_cols}
    df = df.rename(columns=lowered_map)

    rename_map = {}
    for c in df.columns:
        if c in RAW_TO_STD_COLS:
            rename_map[c] = RAW_TO_STD_COLS[c]
    df = df.rename(columns=rename_map)

    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column in Excel: {col}")

    optional_defaults = {
        "source_id": "",
        "created_at": "",
        "telefono_raw": "",
        "telefono_region": "",
        "perfil": "",
        "banos_min": 1,
        "habitaciones": 0,
        "notas": "",
        "zonas_preferidas": "",
        "pais_origen": "",
        "pais_origen_codigo": "",
        "estilo_vida": "",
        "urgencia_mudanza": "",
        "urgencia_mudanza_codigo": "",
        "busqueda_vivienda": "",
        "busqueda_vivienda_codigo": "",
        "teletrabajo": "",
        "zona_otra": "",
        "preferencia_rutina_hogar": "",
    }

    for col, default_val in optional_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    df["source_id"] = df["source_id"].apply(normalize_text)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    df["nombre"] = df["nombre"].apply(normalize_text)
    df["telefono"] = df["telefono"].apply(normalize_phone)
    df["telefono_raw"] = df["telefono_raw"].apply(normalize_text)
    df["telefono_region"] = df["telefono_region"].apply(normalize_text)

    df["pais"] = df["telefono"].apply(detectar_pais)
    df["pais_origen"] = df["pais_origen"].apply(normalize_text)
    df["pais_origen_codigo"] = df["pais_origen_codigo"].apply(normalize_text)

    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df["genero"] = df["genero"].apply(normalize_genero)
    df["pref_genero"] = df["pref_genero"].apply(normalize_pref_genero)
    df["idioma"] = df["idioma"].apply(normalize_text)

    df["zona"] = df["zona"].apply(normalize_text)
    df["zonas_preferidas"] = df["zonas_preferidas"].apply(normalize_text)
    df["zona_otra"] = df["zona_otra"].apply(normalize_text)

    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["inicio"] = pd.to_datetime(df["inicio"], errors="coerce").dt.date
    df["fin"] = pd.to_datetime(df["fin"], errors="coerce").dt.date

    df["perfil"] = df["perfil"].apply(normalize_text)
    df["banos_min"] = pd.to_numeric(df["banos_min"], errors="coerce").fillna(1)
    df["habitaciones"] = pd.to_numeric(df["habitaciones"], errors="coerce").fillna(0)

    df["notas"] = df["notas"].apply(normalize_text)
    df["estilo_vida"] = df["estilo_vida"].apply(normalize_text)
    df["urgencia_mudanza"] = df["urgencia_mudanza"].apply(normalize_text)
    df["urgencia_mudanza_codigo"] = df["urgencia_mudanza_codigo"].apply(normalize_text)
    df["busqueda_vivienda"] = df["busqueda_vivienda"].apply(normalize_text)
    df["busqueda_vivienda_codigo"] = df["busqueda_vivienda_codigo"].apply(normalize_text)
    df["teletrabajo"] = df["teletrabajo"].apply(normalize_text)
    df["preferencia_rutina_hogar"] = df["preferencia_rutina_hogar"].apply(normalize_text)

    df = df.dropna(subset=[
        "nombre", "telefono", "edad", "idioma", "zona", "budget", "inicio", "fin"
    ])

    df = df[df["telefono"].str.len() >= 10]
    df = df[df["fin"] >= df["inicio"]]

    df["edad"] = df["edad"].astype(int)
    df["banos_min"] = df["banos_min"].astype(int)
    df["habitaciones"] = df["habitaciones"].astype(int)

    df = df[(df["edad"] >= 18) & (df["edad"] <= 80)]
    df = df[(df["banos_min"] >= 1)]
    df = df[(df["budget"] > 0)]

    return df


# =========================================================
# MATCHING HELPERS
# =========================================================
def zones_set(zona: str):
    text = str(zona).replace("/", ",").replace("|", ",")
    parts = [z.strip().lower() for z in text.split(",")]
    return {p for p in parts if p}


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


def room_score(h1, h2) -> float:
    try:
        d = abs(int(h1) - int(h2))
        if d == 0:
            return 1.0
        if d == 1:
            return 0.7
        return 0.4
    except Exception:
        return 0.5


def age_score(e1, e2) -> float:
    try:
        diff = abs(int(e1) - int(e2))
        return max(0.0, 1.0 - diff / 15.0)
    except Exception:
        return 0.5


def exact_or_soft_text_score(v1, v2) -> float:
    a = normalize_text(v1).lower()
    b = normalize_text(v2).lower()

    if not a or not b:
        return 0.5
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.75
    return 0.0


def teletrabajo_score(v1, v2) -> float:
    a = normalize_text(v1).lower()
    b = normalize_text(v2).lower()

    if not a or not b:
        return 0.5
    if a == b:
        return 1.0
    return 0.3


def idioma_score(i1, i2) -> float:
    a = normalize_text(i1).lower()
    b = normalize_text(i2).lower()

    if not a or not b:
        return 0.5
    if a == b:
        return 1.0

    sa = {x.strip() for x in re.split(r"[,/|]", a) if x.strip()}
    sb = {x.strip() for x in re.split(r"[,/|]", b) if x.strip()}

    if sa.intersection(sb):
        return 1.0

    return 0.0


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
    ages = age_score(target["edad"], other["edad"])
    baths = bath_score(target.get("banos_min", 1), other.get("banos_min", 1))
    rooms = room_score(target.get("habitaciones", 0), other.get("habitaciones", 0))
    langs = idioma_score(target.get("idioma", ""), other.get("idioma", ""))
    life = exact_or_soft_text_score(target.get("estilo_vida", ""), other.get("estilo_vida", ""))
    routine = exact_or_soft_text_score(
        target.get("preferencia_rutina_hogar", ""),
        other.get("preferencia_rutina_hogar", "")
    )
    remote = teletrabajo_score(target.get("teletrabajo", ""), other.get("teletrabajo", ""))

    total = (
        weights["zone"] * zs +
        weights["budget"] * bs +
        weights["dates"] * ds +
        weights["age"] * ages +
        weights["bath"] * baths +
        weights["rooms"] * rooms +
        weights["language"] * langs +
        weights["lifestyle"] * life +
        weights["routine"] * routine +
        weights["remote"] * remote
    )

    return {
        "score_total": float(total),
        "score_zone": float(zs),
        "score_budget": float(bs),
        "score_dates": float(ds),
        "score_age": float(ages),
        "score_bath": float(baths),
        "score_rooms": float(rooms),
        "score_language": float(langs),
        "score_lifestyle": float(life),
        "score_routine": float(routine),
        "score_remote": float(remote),
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
        weights = {
            "zone": 0.20,
            "budget": 0.15,
            "dates": 0.15,
            "age": 0.10,
            "bath": 0.05,
            "rooms": 0.05,
            "language": 0.10,
            "lifestyle": 0.08,
            "routine": 0.07,
            "remote": 0.05,
        }

    target = df[df["id"] == target_id].iloc[0]
    rows = []

    for _, other in df.iterrows():
        if int(other["id"]) == int(target_id):
            continue

        if country_filter != "All" and other.get("pais_origen", "") != country_filter:
            continue

        if language_filter != "All":
            other_lang = normalize_text(other.get("idioma", ""))
            if other_lang != language_filter:
                continue

        scored = score_pair(target, other, min_overlap_days, require_zone, weights)
        if scored is None:
            continue

        rows.append({
            "match_id": int(other["id"]),
            "match_name": other["nombre"],
            "match_phone": other.get("telefono", ""),
            "match_country": other.get("pais_origen", other.get("pais", "")),
            "match_language": other.get("idioma", ""),
            "match_age": int(other.get("edad", 0)),
            "match_gender": other.get("genero", ""),
            "match_pref_gender": other.get("pref_genero", ""),
            "match_zone": other.get("zona", ""),
            "match_budget": other.get("budget", 0),
            "match_start": other.get("inicio"),
            "match_end": other.get("fin"),
            "match_banos_min": int(other.get("banos_min", 1)) if pd.notna(other.get("banos_min", 1)) else 1,
            "match_habitaciones": int(other.get("habitaciones", 0)) if pd.notna(other.get("habitaciones", 0)) else 0,
            "match_estilo_vida": other.get("estilo_vida", ""),
            "match_teletrabajo": other.get("teletrabajo", ""),
            "match_rutina": other.get("preferencia_rutina_hogar", ""),
            "match_busqueda_vivienda": other.get("busqueda_vivienda", ""),
            "match_perfil": other.get("perfil", ""),
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in scored.items()},
            "notes": other.get("notas", "")
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("score_total", ascending=False).head(int(top_n))
    return out


# =========================================================
# WHATSAPP HELPERS
# =========================================================
def wa_digits(phone: str) -> str:
    if not phone:
        return ""
    return re.sub(r"[^\d]", "", str(phone))


def generate_whatsapp_intro(target: pd.Series, match_row: pd.Series) -> str:
    target_country = target.get("pais_origen", "") or target.get("pais", "")
    return (
        f"Hi {match_row['match_name']}! 👋\n\n"
        f"I'm connecting you with someone who could be a great roommate match.\n\n"
        f"✅ Potential roommate: {target['nombre']} ({target['edad']}), {target_country} — {target['telefono']}\n"
        f"• Preferred areas: {target['zona']}\n"
        f"• Budget: €{int(target['budget'])}\n"
        f"• Dates: {target['inicio']} to {target['fin']}\n"
        f"• Language: {target['idioma']}\n"
        f"• Lifestyle: {target.get('estilo_vida', '')}\n"
        f"• Home routine: {target.get('preferencia_rutina_hogar', '')}\n\n"
        f"About you (from our database):\n"
        f"• Areas: {match_row['match_zone']}\n"
        f"• Budget: €{int(match_row['match_budget'])}\n"
        f"• Dates: {match_row['match_start']} to {match_row['match_end']}\n"
        f"• Language: {match_row['match_language']}\n"
        f"• Lifestyle: {match_row.get('match_estilo_vida', '')}\n"
        f"• Home routine: {match_row.get('match_rutina', '')}\n\n"
        f"If you're both happy, you can message each other directly and set up a quick call 😊"
    )


def whatsapp_web_link(phone: str, message: str) -> str:
    digits = wa_digits(phone)
    if not digits:
        return ""
    return f"https://wa.me/{digits}?text={quote(message)}"


# =========================================================
# DASHBOARD
# =========================================================
def dashboard(df: pd.DataFrame):
    st.subheader("📊 Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total clients", int(len(df)))
    with c2:
        st.metric("Countries", int(df["pais_origen"].nunique()) if "pais_origen" in df.columns else 0)
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

    if "pais_origen" in df.columns and len(df):
        st.markdown("**Clients by country of origin**")
        st.bar_chart(df["pais_origen"].fillna("Unknown").replace("", "Unknown").value_counts())

    if "idioma" in df.columns and len(df):
        st.markdown("**Clients by language**")
        st.bar_chart(df["idioma"].fillna("Unknown").replace("", "Unknown").value_counts())

    if "edad" in df.columns and len(df):
        st.markdown("**Age distribution**")
        age_bins = pd.cut(df["edad"], bins=[17, 20, 25, 30, 35, 40, 50, 80], right=True)
        counts = age_bins.value_counts().sort_index()
        counts.index = counts.index.astype(str)
        st.bar_chart(counts)

    if "estilo_vida" in df.columns and len(df):
        clean_life = df["estilo_vida"].fillna("").astype(str).str.strip()
        clean_life = clean_life[clean_life != ""]
        if len(clean_life):
            st.markdown("**Lifestyle**")
            st.bar_chart(clean_life.value_counts())

    if "preferencia_rutina_hogar" in df.columns and len(df):
        clean_rutina = df["preferencia_rutina_hogar"].fillna("").astype(str).str.strip()
        clean_rutina = clean_rutina[clean_rutina != ""]
        if len(clean_rutina):
            st.markdown("**Home routine preference**")
            st.bar_chart(clean_rutina.value_counts())


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Programa Match", layout="wide")
st.title("🏡 Roommate Matching System — Excel-based Matching")

init_db()

with st.sidebar:
    st.header("Upload Clients")
    file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    sheet_name = st.text_input("Sheet name (optional)", value="")

    if st.button("Reset database (delete all)", type="secondary"):
        reset_db()
        st.success("Database reset. Upload again.")

    if st.button("Import to database", type="primary"):
        if file is None:
            st.warning("Please upload an Excel file first.")
        else:
            try:
                chosen_sheet = sheet_name.strip() if sheet_name.strip() else first_non_empty_sheet(file)
                df_excel = pd.read_excel(file, sheet_name=chosen_sheet)
                df_clean = normalize_df(df_excel)
                processed, _ = upsert_clients(df_clean)
                st.success(f"Upserted {processed} clients from sheet '{chosen_sheet}' (no duplicates by phone).")
            except Exception as e:
                st.error(f"Import error: {e}")

df = load_clients()

tab1, tab2, tab3 = st.tabs(["Dashboard", "Match (1-to-1)", "Database Preview"])

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
                format_func=lambda x: (
                    f"{int(x)} — {df[df['id'] == x].iloc[0]['nombre']} "
                    f"({df[df['id'] == x].iloc[0].get('pais_origen','')})"
                )
            )

        with colB:
            top_n = st.number_input("Top N", min_value=3, max_value=100, value=15)

        with colC:
            min_overlap_days = st.number_input("Min overlap (days)", min_value=1, max_value=365, value=30)

        with colD:
            require_zone = st.checkbox("Require same zone", value=True)

        countries = ["All"] + sorted([
            c for c in df["pais_origen"].dropna().unique().tolist()
            if str(c).strip() != ""
        ])
        languages = ["All"] + sorted([
            l for l in df["idioma"].dropna().unique().tolist()
            if str(l).strip() != ""
        ])

        f1, f2 = st.columns([1, 1])
        with f1:
            country_filter = st.selectbox("Filter by country of origin", options=countries)
        with f2:
            language_filter = st.selectbox("Filter by language", options=languages)

        st.markdown("### Scoring weights (advanced)")
        wcol1, wcol2, wcol3, wcol4, wcol5 = st.columns(5)
        with wcol1:
            w_zone = st.slider("Zone", 0.0, 0.5, 0.20, 0.05)
            w_budget = st.slider("Budget", 0.0, 0.5, 0.15, 0.05)
        with wcol2:
            w_dates = st.slider("Dates", 0.0, 0.5, 0.15, 0.05)
            w_age = st.slider("Age", 0.0, 0.5, 0.10, 0.05)
        with wcol3:
            w_bath = st.slider("Bathrooms", 0.0, 0.5, 0.05, 0.05)
            w_rooms = st.slider("Rooms", 0.0, 0.5, 0.05, 0.05)
        with wcol4:
            w_language = st.slider("Language", 0.0, 0.5, 0.10, 0.05)
            w_lifestyle = st.slider("Lifestyle", 0.0, 0.5, 0.08, 0.05)
        with wcol5:
            w_routine = st.slider("Home routine", 0.0, 0.5, 0.07, 0.05)
            w_remote = st.slider("Remote work", 0.0, 0.5, 0.05, 0.05)

        total_w = (
            w_zone + w_budget + w_dates + w_age + w_bath +
            w_rooms + w_language + w_lifestyle + w_routine + w_remote
        )

        if total_w == 0:
            st.warning("Set at least one weight > 0.")
            weights = {
                "zone": 0, "budget": 0, "dates": 0, "age": 0, "bath": 0,
                "rooms": 0, "language": 0, "lifestyle": 0, "routine": 0, "remote": 0
            }
        else:
            weights = {
                "zone": w_zone / total_w,
                "budget": w_budget / total_w,
                "dates": w_dates / total_w,
                "age": w_age / total_w,
                "bath": w_bath / total_w,
                "rooms": w_rooms / total_w,
                "language": w_language / total_w,
                "lifestyle": w_lifestyle / total_w,
                "routine": w_routine / total_w,
                "remote": w_remote / total_w,
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

            st.markdown("### 📲 WhatsApp Web links")
            st.dataframe(wa_df, use_container_width=True)

with tab3:
    st.subheader("🗂 Database preview")
    if df.empty:
        st.info("No data loaded yet.")
    else:
        st.dataframe(df, use_container_width=True)
