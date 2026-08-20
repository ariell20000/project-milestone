"""Adds a `country` column to the escxtra mapping (outputs/escxtra_reviews_mapped.csv),
exploding posts that reference several countries into one row per (post, country)
pair - titles like "Serbia and Israel reviewed" or "Xtra Jury: We review Bosnia &
Herzegovina, Albania and Italy" are common, so a post -> country relationship is
many-to-many, not one-to-one.

Country is resolved in three tiers, tried in order per post, recorded in
`country_source` so low-confidence rows can be filtered separately:

1. title_country - the country name/alias/flag emoji is directly in the
   title (curated, short text - low false-positive risk).
2. title_artist - most rehearsal-day titles instead name the artist (e.g.
   "Day 6: Sevak gives us perfect vocals..." for Armenia's Sevak
   Khanagyan) with no country word at all. Matched against that year's
   actual entrant list from eurovision_enriched2.csv.
3. text_fallback - last resort, scans the article opening for country
   names. This tier is noisy (an article about one country's rehearsal
   routinely name-drops several others in passing) so it's kept separate
   for filtering rather than trusted at face value.

Only `contest == "senior"` rows are processed - see escxtra_mapping.py for
why Junior Eurovision rows are excluded.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MAPPED_PATH = BASE_DIR / "outputs" / "escxtra_reviews_mapped.csv"
ENRICHED_PATH = BASE_DIR / "eurovision_enriched2.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "escxtra_reviews_countries.csv"

# canonical_key -> (display name matching eurovision_enriched2.csv's most
# common spelling, [name aliases to match in text])
COUNTRIES: dict[str, tuple[str, list[str]]] = {
    "albania": ("Albania", []),
    "armenia": ("Armenia", []),
    "australia": ("Australia", []),
    "austria": ("Austria", []),
    "azerbaijan": ("Azerbaijan", []),
    "belarus": ("Belarus", []),
    "belgium": ("Belgium", []),
    "bosnia_herzegovina": ("Bosnia and Herzegovina", ["Bosnia & Herzegovina", "Bosnia"]),
    "bulgaria": ("Bulgaria", []),
    "croatia": ("Croatia", []),
    "cyprus": ("Cyprus", []),
    "czech_republic": ("Czech Republic", ["Czechia"]),
    "denmark": ("Denmark", []),
    "estonia": ("Estonia", []),
    "finland": ("Finland", []),
    "france": ("France", []),
    "georgia": ("Georgia", []),
    "germany": ("Germany", []),
    "greece": ("Greece", []),
    "hungary": ("Hungary", []),
    "iceland": ("Iceland", []),
    "ireland": ("Ireland", []),
    "israel": ("Israel", []),
    "italy": ("Italy", []),
    "latvia": ("Latvia", []),
    "lithuania": ("Lithuania", []),
    "luxembourg": ("Luxembourg", []),
    "macedonia": ("North Macedonia", ["Macedonia", "North Macedonia", "FYR Macedonia"]),
    "malta": ("Malta", []),
    "moldova": ("Moldova", []),
    "monaco": ("Monaco", []),
    "montenegro": ("Montenegro", []),
    "morocco": ("Morocco", []),
    "netherlands": ("Netherlands", ["The Netherlands"]),
    "norway": ("Norway", []),
    "poland": ("Poland", []),
    "portugal": ("Portugal", []),
    "romania": ("Romania", []),
    "russia": ("Russia", []),
    "san_marino": ("San Marino", []),
    "serbia": ("Serbia", []),
    "slovakia": ("Slovakia", []),
    "slovenia": ("Slovenia", []),
    "spain": ("Spain", []),
    "sweden": ("Sweden", []),
    "switzerland": ("Switzerland", []),
    "turkey": ("Turkey", []),
    "ukraine": ("Ukraine", []),
    "united_kingdom": ("United Kingdom", ["UK"]),
    "andorra": ("Andorra", []),
}

ISO2_TO_KEY: dict[str, str] = {
    "AL": "albania", "AM": "armenia", "AU": "australia", "AT": "austria",
    "AZ": "azerbaijan", "BY": "belarus", "BE": "belgium", "BA": "bosnia_herzegovina",
    "BG": "bulgaria", "HR": "croatia", "CY": "cyprus", "CZ": "czech_republic",
    "DK": "denmark", "EE": "estonia", "FI": "finland", "FR": "france",
    "GE": "georgia", "DE": "germany", "GR": "greece", "HU": "hungary",
    "IS": "iceland", "IE": "ireland", "IL": "israel", "IT": "italy",
    "LV": "latvia", "LT": "lithuania", "LU": "luxembourg", "MK": "macedonia",
    "MT": "malta", "MD": "moldova", "MC": "monaco", "ME": "montenegro",
    "MA": "morocco", "NL": "netherlands", "NO": "norway", "PL": "poland",
    "PT": "portugal", "RO": "romania", "RU": "russia", "SM": "san_marino",
    "RS": "serbia", "SK": "slovakia", "SI": "slovenia", "ES": "spain",
    "SE": "sweden", "CH": "switzerland", "TR": "turkey", "UA": "ukraine",
    "GB": "united_kingdom", "AD": "andorra",
}

FLAG_RE = re.compile(r"[\U0001F1E6-\U0001F1FF]{2}")


def _build_name_pattern() -> re.Pattern:
    all_names: list[str] = []
    for canon, aliases in COUNTRIES.values():
        all_names.append(canon)
        all_names.extend(aliases)
    all_names = sorted(set(all_names), key=len, reverse=True)
    escaped = [re.escape(n) for n in all_names]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.I)


NAME_PATTERN = _build_name_pattern()
ALIAS_TO_KEY: dict[str, str] = {}
for key, (canon, aliases) in COUNTRIES.items():
    ALIAS_TO_KEY[canon.lower()] = key
    for alias in aliases:
        ALIAS_TO_KEY[alias.lower()] = key


def countries_in_text(text: str) -> set[str]:
    found: set[str] = set()
    for match in NAME_PATTERN.finditer(text):
        found.add(ALIAS_TO_KEY[match.group(1).lower()])
    for flag in FLAG_RE.findall(text):
        codepoints = [ord(c) - 0x1F1E6 + ord("A") for c in flag]
        iso2 = "".join(chr(c) for c in codepoints)
        if iso2 in ISO2_TO_KEY:
            found.add(ISO2_TO_KEY[iso2])
    return found


NAME_STOPWORDS = {
    "the", "dj", "and", "feat", "feat.", "de", "van", "el", "of", "for",
}


def normalize_country(raw: str) -> str:
    """eurovision_enriched2.csv spells some countries differently across
    years (Macedonia/North Macedonia, Netherlands/The Netherlands, Czech
    Republic/Czechia); route through the same alias table used for
    title/text matching so every tier agrees on one display name."""
    key = ALIAS_TO_KEY.get(raw.strip().lower())
    return COUNTRIES[key][0] if key else raw


def build_artist_index(year: int, enriched: pd.DataFrame) -> list[tuple[str, list[str], str]]:
    """Per-year list of (full_artist_name, distinctive_tokens, country)."""
    year_rows = enriched[enriched["Year"] == year]
    index = []
    for _, r in year_rows.iterrows():
        full_name = re.sub(r"\s*\(\d+\)\s*$", "", str(r["Artist"])).strip()  # strip "(2)" repeat-entrant suffix
        tokens = [t for t in re.split(r"[\s'’\-]+", full_name) if len(t) >= 3 and t.lower() not in NAME_STOPWORDS]
        index.append((full_name, tokens, normalize_country(str(r["Country"]))))
    return index


def countries_by_artist(title: str, artist_index: list[tuple[str, list[str], str]]) -> set[str]:
    title_lower = title.lower()
    # tier A: full artist-name substring match (highest confidence)
    full_matches = {country for full_name, _, country in artist_index if full_name and full_name.lower() in title_lower}
    if full_matches:
        return full_matches
    # tier B: distinctive single-name-token match, only where the token is
    # unique to one artist that year (avoids "Lea" clashing with "Léa X")
    token_owners: dict[str, set[str]] = {}
    for full_name, tokens, country in artist_index:
        for tok in tokens:
            token_owners.setdefault(tok.lower(), set()).add(country)
    token_matches: set[str] = set()
    for tok, owners in token_owners.items():
        if len(owners) == 1 and re.search(r"\b" + re.escape(tok) + r"\b", title, re.I):
            token_matches |= owners
    return token_matches


def main() -> None:
    df = pd.read_csv(MAPPED_PATH)
    senior = df[df["contest"] == "senior"].copy()
    enriched = pd.read_csv(ENRICHED_PATH)

    rows = []
    for _, row in senior.iterrows():
        title = str(row["title"])
        title_countries = countries_in_text(title)
        if title_countries:
            keys_or_names, source = title_countries, "title_country"
            names = [COUNTRIES[k][0] for k in sorted(keys_or_names)]
        else:
            artist_index = build_artist_index(int(row["year"]), enriched)
            artist_countries = countries_by_artist(title, artist_index)
            if artist_countries:
                names, source = sorted(artist_countries), "title_artist"
            else:
                fallback_keys = countries_in_text(str(row["text"])[:800])
                if fallback_keys:
                    names = [COUNTRIES[k][0] for k in sorted(fallback_keys)]
                    source = "text_fallback"
                else:
                    names, source = [], "none"

        if not names:
            rows.append({**row.to_dict(), "country": None, "country_source": "none"})
            continue
        for name in names:
            rows.append({**row.to_dict(), "country": name, "country_source": source})

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)

    n_posts = len(senior)
    n_rows = len(out)
    n_no_country = (out["country"].isna()).sum()
    n_multi = out.groupby(["slug"]).size()
    n_multi_country_posts = (n_multi > 1).sum()

    print(f"Senior posts: {n_posts} -> exploded rows: {n_rows}")
    print(f"Posts with no country match at all: {n_no_country}")
    print(f"Posts matching >1 country: {n_multi_country_posts}")
    print("\ncountry_source counts:")
    print(out["country_source"].value_counts())
    print("\ntop countries by post count:")
    print(out["country"].value_counts().head(15))
    print(f"\nSaved to {OUTPUT_PATH}")

    print("\nsample of no-country posts (need manual look):")
    print(out.loc[out["country"].isna(), "title"].head(15).to_string())


if __name__ == "__main__":
    main()
