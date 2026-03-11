import streamlit as st
import pandas as pd
import sqlite3
import re
import itertools
from urllib.parse import quote

DB_PATH = "match_v13_group.db"

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

DEFAULT_WEIGHTS = {
    "dates": 0.40,
    "zone": 0.30,
    "pref_gender": 0.15,
    "budget": 0.06,
    "age": 0.03,
    "language": 0.03,
    "lifestyle": 0.01,
    "routine": 0.01,
    "remote": 0.01,
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

    country_codes = ["351", "353", "34", "52", "57", "54", "33", "39", "44", "49", "31", "41", "1"]

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
    if telefono.startswith("+351"):
        return "Portugal"
    if telefono.startswith("+353"):
        return "Ireland"
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
    return "Other"


def normalize_genero(value: str) -> str:
    return GENERO_MAP.get(str(value).strip().lower(), "otro")


def normalize_pref_genero(value: str) -> str:
    return PREF_GENERO_MAP.get(str(value).strip().lower(), "mixto")


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

    lowered_map = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=lowered_map)

    rename_map = {c: RAW_TO_STD_COLS[c] for c in df.columns if c in RAW_TO_STD_COLS}
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

    df = df.dropna(subset=["nombre", "telefono", "edad", "idioma", "zona", "budget", "inicio", "fin"])
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
# MATCH HELPERS
# =========================================================
def zones_set(zona: str):
    text = str(zona).replace("/", ",").replace("|", ",").replace(";", ",")
    parts = [z.strip().lower() for z in text.split(",")]
    return {p for p in parts if p}


def overlap_days(a_start, a_end, b_start, b_end) -> int:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if end < start:
        return 0
    return (end - start).days + 1


def period_length(start, end) -> int:
    return max(1, (end - start).days + 1)


def date_overlap_ratio(a_start, a_end, b_start, b_end) -> float:
    days = overlap_days(a_start, a_end, b_start, b_end)
    shorter = min(period_length(a_start, a_end), period_length(b_start, b_end))
    return days / shorter if shorter > 0 else 0.0


def zone_score(z1, z2) -> float:
    inter = len(z1.intersection(z2))
    if inter == 0:
        return 0.0
    union = len(z1.union(z2))
    return inter / union if union > 0 else 0.0


def budget_score(b1, b2) -> float:
    if b1 <= 0 or b2 <= 0:
        return 0.0
    ratio = abs(b1 - b2) / max(b1, b2)
    if ratio <= 0.10:
        return 1.0
    if ratio <= 0.20:
        return 0.90
    if ratio <= 0.30:
        return 0.75
    if ratio <= 0.40:
        return 0.55
    if ratio <= 0.50:
        return 0.35
    return 0.10


def age_score(e1, e2) -> float:
    diff = abs(int(e1) - int(e2))
    if diff <= 3:
        return 1.0
    if diff <= 6:
        return 0.85
    if diff <= 10:
        return 0.65
    if diff <= 15:
        return 0.45
    return 0.20


def idioma_score(i1, i2) -> float:
    a = normalize_text(i1).lower()
    b = normalize_text(i2).lower()

    if not a or not b:
        return 0.50
    if a == b:
        return 1.0

    sa = {x.strip() for x in re.split(r"[,/|;]", a) if x.strip()}
    sb = {x.strip() for x in re.split(r"[,/|;]", b) if x.strip()}

    if sa.intersection(sb):
        return 0.90
    return 0.0


def exact_or_soft_text_score(v1, v2) -> float:
    a = normalize_text(v1).lower()
    b = normalize_text(v2).lower()

    if not a or not b:
        return 0.50
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.75
    return 0.20


def teletrabajo_score(v1, v2) -> float:
    a = normalize_text(v1).lower()
    b = normalize_text(v2).lower()

    if not a or not b:
        return 0.50
    if a == b:
        return 1.0
    return 0.30


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


def match_genero_bidireccional(a: pd.Series, b: pd.Series) -> bool:
    return (
        genero_compatible(a.get("pref_genero", "mixto"), b.get("genero", "otro")) and
        genero_compatible(b.get("pref_genero", "mixto"), a.get("genero", "otro"))
    )


def preference_gender_score(a: pd.Series, b: pd.Series) -> float:
    return 1.0 if match_genero_bidireccional(a, b) else 0.0


def score_pair(target: pd.Series, other: pd.Series, min_overlap_days: int, require_zone: bool, weights: dict):
    gender_ok = match_genero_bidireccional(target, other)
    if not gender_ok:
        return None

    odays = overlap_days(target["inicio"], target["fin"], other["inicio"], other["fin"])
    if odays < min_overlap_days:
        return None

    tz = zones_set(target["zona"])
    oz = zones_set(other["zona"])
    zs = zone_score(tz, oz)

    if require_zone and zs == 0:
        return None

    ds = date_overlap_ratio(target["inicio"], target["fin"], other["inicio"], other["fin"])
    bs = budget_score(float(target["budget"]), float(other["budget"]))
    ages = age_score(target["edad"], other["edad"])
    langs = idioma_score(target.get("idioma", ""), other.get("idioma", ""))
    life = exact_or_soft_text_score(target.get("estilo_vida", ""), other.get("estilo_vida", ""))
    routine = exact_or_soft_text_score(
        target.get("preferencia_rutina_hogar", ""),
        other.get("preferencia_rutina_hogar", "")
    )
    remote = teletrabajo_score(target.get("teletrabajo", ""), other.get("teletrabajo", ""))
    pgs = preference_gender_score(target, other)

    total = (
        weights["dates"] * ds +
        weights["zone"] * zs +
        weights["pref_gender"] * pgs +
        weights["budget"] * bs +
        weights["age"] * ages +
        weights["language"] * langs +
        weights["lifestyle"] * life +
        weights["routine"] * routine +
        weights["remote"] * remote
    )

    return {
        "score_total": float(total),
        "score_dates": float(ds),
        "score_zone": float(zs),
        "score_pref_gender": float(pgs),
        "score_budget": float(bs),
        "score_age": float(ages),
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
        weights = DEFAULT_WEIGHTS.copy()

    target = df[df["id"] == target_id].iloc[0]
    rows = []

    for _, other in df.iterrows():
        if int(other["id"]) == int(target_id):
            continue

        if country_filter != "All":
            other_country = normalize_text(other.get("pais_origen", "")) or normalize_text(other.get("pais", ""))
            if other_country != country_filter:
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
            "match_estilo_vida": other.get("estilo_vida", ""),
            "match_teletrabajo": other.get("teletrabajo", ""),
            "match_rutina": other.get("preferencia_rutina_hogar", ""),
            "match_perfil": other.get("perfil", ""),
            "notes": other.get("notas", ""),
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in scored.items()}
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        by=["score_total", "score_dates", "score_zone", "score_pref_gender", "overlap_days"],
        ascending=[False, False, False, False, False]
    ).head(int(top_n))
    return out


# =========================================================
# GROUP MATCHING
# =========================================================
def pair_score_lookup(df: pd.DataFrame, ids: list[int], min_overlap_days: int, require_zone: bool, weights: dict):
    lookup = {}
    for i, j in itertools.combinations(ids, 2):
        a = df[df["id"] == i].iloc[0]
        b = df[df["id"] == j].iloc[0]
        s = score_pair(a, b, min_overlap_days, require_zone, weights)
        lookup[(i, j)] = s
        lookup[(j, i)] = s
    return lookup


def group_common_zone(group_df: pd.DataFrame) -> str:
    zone_sets = [zones_set(z) for z in group_df["zona"].tolist()]
    if not zone_sets:
        return ""
    inter = zone_sets[0].copy()
    for z in zone_sets[1:]:
        inter = inter.intersection(z)
    if inter:
        return ", ".join(sorted(inter))
    union = set()
    for z in zone_sets:
        union.update(z)
    return ", ".join(sorted(union))


def group_common_dates(group_df: pd.DataFrame):
    latest_start = max(group_df["inicio"].tolist())
    earliest_end = min(group_df["fin"].tolist())
    overlap = 0
    if earliest_end >= latest_start:
        overlap = (earliest_end - latest_start).days + 1
    return latest_start, earliest_end, overlap


def group_gender_compatible(group_df: pd.DataFrame) -> bool:
    members = list(group_df.to_dict(orient="records"))
    for a, b in itertools.combinations(members, 2):
        sa = pd.Series(a)
        sb = pd.Series(b)
        if not match_genero_bidireccional(sa, sb):
            return False
    return True


def group_avg_budget(group_df: pd.DataFrame) -> float:
    return float(group_df["budget"].mean())


def compute_group_matches(
    df: pd.DataFrame,
    target_id: int,
    group_size: int = 3,
    top_n: int = 10,
    min_overlap_days: int = 30,
    require_zone: bool = True,
    weights=None
) -> pd.DataFrame:
    if df.empty or group_size < 2:
        return pd.DataFrame()

    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    target_df = df[df["id"] == int(target_id)]
    if target_df.empty:
        return pd.DataFrame()

    target = target_df.iloc[0]
    other_ids = [int(x) for x in df[df["id"] != int(target_id)]["id"].tolist()]
    if len(other_ids) < group_size - 1:
        return pd.DataFrame()

    lookup = pair_score_lookup(df, [int(target_id)] + other_ids, min_overlap_days, require_zone, weights)
    rows = []

    for combo in itertools.combinations(other_ids, group_size - 1):
        member_ids = [int(target_id)] + list(combo)
        group_df = df[df["id"].isin(member_ids)].copy()

        if len(group_df) != group_size:
            continue

        if not group_gender_compatible(group_df):
            continue

        latest_start, earliest_end, overlap = group_common_dates(group_df)
        if overlap < min_overlap_days:
            continue

        common_zone = group_common_zone(group_df)
        if require_zone and not common_zone:
            continue

        pair_scores = []
        valid = True

        for a, b in itertools.combinations(member_ids, 2):
            s = lookup.get((a, b))
            if s is None:
                valid = False
                break
            pair_scores.append(s)

        if not valid or not pair_scores:
            continue

        avg_total = sum(x["score_total"] for x in pair_scores) / len(pair_scores)
        avg_dates = sum(x["score_dates"] for x in pair_scores) / len(pair_scores)
        avg_zone = sum(x["score_zone"] for x in pair_scores) / len(pair_scores)
        avg_pref_gender = sum(x["score_pref_gender"] for x in pair_scores) / len(pair_scores)
        avg_budget = sum(x["score_budget"] for x in pair_scores) / len(pair_scores)

        member_lines = []
        for _, row in group_df.sort_values("nombre").iterrows():
            member_lines.append(
                f"{row['nombre']} | {row['edad']} años | {row['telefono']} | {row['zona']}"
            )

        rows.append({
            "group_size": group_size,
            "member_ids": member_ids,
            "member_names": " | ".join(group_df.sort_values("nombre")["nombre"].tolist()),
            "member_contacts": " | ".join(group_df.sort_values("nombre")["telefono"].tolist()),
            "group_common_zone": common_zone,
            "group_start": latest_start,
            "group_end": earliest_end,
            "group_overlap_days": overlap,
            "group_avg_budget": round(group_avg_budget(group_df), 2),
            "score_total": round(avg_total, 4),
            "score_dates": round(avg_dates, 4),
            "score_zone": round(avg_zone, 4),
            "score_pref_gender": round(avg_pref_gender, 4),
            "score_budget": round(avg_budget, 4),
            "members_detail": "\n".join(member_lines),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        by=["score_total", "score_dates", "score_zone", "score_pref_gender", "group_overlap_days"],
        ascending=[False, False, False, False, False]
    ).head(int(top_n))

    return out


# =========================================================
# WHATSAPP / COPY TEXT
# =========================================================
def wa_digits(phone: str) -> str:
    if not phone:
        return ""
    return re.sub(r"[^\d]", "", str(phone))


def whatsapp_web_link(phone: str, message: str) -> str:
    digits = wa_digits(phone)
    if not digits:
        return ""
    return f"https://wa.me/{digits}?text={quote(message)}"


def build_copy_text_for_1to1(target: pd.Series, matches: pd.DataFrame) -> str:
    if matches.empty:
        return "No encontré matches con los filtros actuales."

    lines = []
    lines.append(f"Estos son tus matches para {target['nombre']}:")
    lines.append("")

    for idx, (_, r) in enumerate(matches.iterrows(), start=1):
        lines.append(f"{idx}. {r['match_name']}")
        lines.append(f"Contacto: {r['match_phone']}")
        lines.append(f"Edad: {r['match_age']}")
        lines.append(f"Zona match: {r['match_zone']}")
        lines.append(f"Fechas: {r['match_start']} a {r['match_end']}")
        lines.append(f"Overlap: {r['overlap_days']} días")
        lines.append(f"Score total: {round(float(r['score_total']) * 100, 1)}%")
        lines.append("")

    return "\n".join(lines).strip()


def build_copy_text_for_groups(target: pd.Series, groups_df: pd.DataFrame) -> str:
    if groups_df.empty:
        return "No encontré grupos compatibles con los filtros actuales."

    lines = []
    lines.append(f"Estos son tus matches en grupo para {target['nombre']}:")
    lines.append("")

    for idx, (_, g) in enumerate(groups_df.iterrows(), start=1):
        lines.append(f"Grupo {idx} ({g['group_size']} personas)")
        lines.append(f"Zona match: {g['group_common_zone']}")
        lines.append(f"Fechas en común: {g['group_start']} a {g['group_end']}")
        lines.append(f"Overlap grupo: {g['group_overlap_days']} días")
        lines.append(f"Score total grupo: {round(float(g['score_total']) * 100, 1)}%")
        lines.append("Integrantes:")
        for member_line in str(g["members_detail"]).split("\n"):
            lines.append(f"- {member_line}")
        lines.append("")

    return "\n".join(lines).strip()


def generate_whatsapp_intro_for_pair(target: pd.Series, match_row: pd.Series) -> str:
    return (
        f"Hola {match_row['match_name']} 👋\n\n"
        f"Te comparto un posible roommate match:\n\n"
        f"Nombre: {target['nombre']}\n"
        f"Contacto: {target['telefono']}\n"
        f"Edad: {target['edad']}\n"
        f"Zona: {target['zona']}\n"
        f"Fechas: {target['inicio']} a {target['fin']}\n"
        f"Budget: €{int(target['budget'])}\n"
        f"Idioma: {target['idioma']}\n\n"
        f"Creemos que puede haber buen match por fechas, zona y preferencias. 😊"
    )


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
        z = df["zona"].astype(str).str.replace(",", "|").str.replace("/", "|").str.replace(";", "|")
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


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Programa Match HAUS", layout="wide")
st.title("🏡 HAUS Mate — Matching 1-to-1 + Group Matching")
st.caption("Prioridad fuerte: fechas + zona + preferred gender")

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

tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard",
    "Match 1-to-1",
    "Group Matches (3-5)",
    "Database Preview"
])

with tab1:
    if df.empty:
        st.info("Upload clients to see the dashboard.")
    else:
        dashboard(df)

with tab2:
    st.subheader("💞 Matchmaking 1-to-1")

    if df.empty:
        st.info("Upload clients to start matching.")
    else:
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

        with c1:
            target_id = st.selectbox(
                "Select client",
                options=df["id"].tolist(),
                format_func=lambda x: (
                    f"{int(x)} — {df[df['id'] == x].iloc[0]['nombre']} "
                    f"({df[df['id'] == x].iloc[0].get('pais_origen', '')})"
                )
            )

        with c2:
            top_n = st.number_input("Top N", min_value=3, max_value=100, value=15)

        with c3:
            min_overlap_days = st.number_input("Min overlap (days)", min_value=1, max_value=365, value=30)

        with c4:
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

        st.markdown("### Priority weights")
        w1, w2, w3 = st.columns(3)
        with w1:
            w_dates = st.slider("Dates", 0.0, 1.0, 0.40, 0.01)
            w_zone = st.slider("Zone", 0.0, 1.0, 0.30, 0.01)
            w_pref_gender = st.slider("Preferred gender", 0.0, 1.0, 0.15, 0.01)
        with w2:
            w_budget = st.slider("Budget", 0.0, 1.0, 0.06, 0.01)
            w_age = st.slider("Age", 0.0, 1.0, 0.03, 0.01)
            w_language = st.slider("Language", 0.0, 1.0, 0.03, 0.01)
        with w3:
            w_lifestyle = st.slider("Lifestyle", 0.0, 1.0, 0.01, 0.01)
            w_routine = st.slider("Home routine", 0.0, 1.0, 0.01, 0.01)
            w_remote = st.slider("Remote work", 0.0, 1.0, 0.01, 0.01)

        total_w = (
            w_dates + w_zone + w_pref_gender + w_budget + w_age +
            w_language + w_lifestyle + w_routine + w_remote
        )

        if total_w == 0:
            st.warning("Set at least one weight > 0.")
            weights = {k: 0 for k in DEFAULT_WEIGHTS.keys()}
        else:
            weights = {
                "dates": w_dates / total_w,
                "zone": w_zone / total_w,
                "pref_gender": w_pref_gender / total_w,
                "budget": w_budget / total_w,
                "age": w_age / total_w,
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
            st.markdown("### ✅ Matches")
            st.dataframe(matches, use_container_width=True)

            target = df[df["id"] == int(target_id)].iloc[0]
            copy_text = build_copy_text_for_1to1(target, matches)

            st.markdown("### 📋 Mensaje listo para copiar y pegar")
            st.text_area("Copy text", value=copy_text, height=350)

            wa_rows = []
            for _, r in matches.iterrows():
                msg = generate_whatsapp_intro_for_pair(target, r)
                link = whatsapp_web_link(r.get("match_phone", ""), msg)
                wa_rows.append({
                    "match_id": r["match_id"],
                    "match_name": r["match_name"],
                    "match_phone": r.get("match_phone", ""),
                    "whatsapp_link": link
                })
            wa_df = pd.DataFrame(wa_rows)

            st.markdown("### 📲 WhatsApp links")
            st.dataframe(wa_df, use_container_width=True)

with tab3:
    st.subheader("👥 Group Matches (3-5 personas)")

    if df.empty:
        st.info("Upload clients to start matching.")
    else:
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

        with c1:
            target_id_group = st.selectbox(
                "Select client for group match",
                options=df["id"].tolist(),
                key="group_target_id",
                format_func=lambda x: (
                    f"{int(x)} — {df[df['id'] == x].iloc[0]['nombre']} "
                    f"({df[df['id'] == x].iloc[0].get('pais_origen', '')})"
                )
            )

        with c2:
            group_size = st.selectbox("Group size", options=[3, 4, 5], index=0)

        with c3:
            top_n_groups = st.number_input("Top groups", min_value=1, max_value=50, value=10)

        with c4:
            min_overlap_days_group = st.number_input("Min overlap days", min_value=1, max_value=365, value=30, key="group_days")

        require_zone_group = st.checkbox("Require same/common zone in group", value=True, key="group_zone")

        group_matches = compute_group_matches(
            df=df,
            target_id=int(target_id_group),
            group_size=int(group_size),
            top_n=int(top_n_groups),
            min_overlap_days=int(min_overlap_days_group),
            require_zone=bool(require_zone_group),
            weights=DEFAULT_WEIGHTS.copy()
        )

        if group_matches.empty:
            st.info("No compatible groups found.")
        else:
            st.markdown("### ✅ Group matches")
            st.dataframe(group_matches, use_container_width=True)

            target_group = df[df["id"] == int(target_id_group)].iloc[0]
            group_copy_text = build_copy_text_for_groups(target_group, group_matches)

            st.markdown("### 📋 Mensaje de grupos listo para copiar y pegar")
            st.text_area("Copy group text", value=group_copy_text, height=420)

with tab4:
    st.subheader("🗂 Database preview")
    if df.empty:
        st.info("No data loaded yet.")
    else:
        st.dataframe(df, use_container_width=True)
