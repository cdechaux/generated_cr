#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import io
import json
import re
import sys
from typing import Iterable, List, Tuple, Dict, Any

# --- Regex codes CIM-10 ---
# Lettre A–T/V–Z + 2 chiffres + (suffixe alphanum 0–4) OU un point + 1–3 alphanum
CIM10_CODE = r'[A-TV-Z]\d{2}(?:[A-Z0-9]{0,4}|\.[A-Z0-9]{1,3})'
PAREN_CODE = re.compile(r'\(\s*(' + CIM10_CODE + r')\s*\)', flags=re.IGNORECASE)

# --- Repères de sections (insensibles à la casse, accents tolérés) ---
SEC_DP = re.compile(r'diagnostic\s*principal\s*:?', re.IGNORECASE | re.UNICODE)
SEC_DR = re.compile(r'diagnostic(?:s)?\s*reli[eé]s?\s*:?', re.IGNORECASE | re.UNICODE)
SEC_DA = re.compile(r'diagnostic(?:s)?\s*associ[eé]s?\s*:?', re.IGNORECASE | re.UNICODE)

SECTION_ORDER = [('dp', SEC_DP), ('dr', SEC_DR), ('da', SEC_DA)]

def slice_sections(text: str) -> Dict[str, str]:
    """
    Coupe le texte 'assistant' en tranches pour DP / DR / DA selon le premier match de chaque repère.
    Renvoie un dict {'dp': str, 'dr': str, 'da': str} (chaînes possiblement vides).
    """
    positions = []
    for key, pat in SECTION_ORDER:
        m = pat.search(text)
        if m:
            positions.append((key, m.start(), m.end()))
    # Ordonner par position d'apparition
    positions.sort(key=lambda x: x[1])

    # Construire les tranches
    out = {'dp': '', 'dr': '', 'da': ''}
    for i, (key, _start, endhdr) in enumerate(positions):
        start = endhdr
        end = len(text)
        if i + 1 < len(positions):
            end = positions[i + 1][1]
        out[key] = text[start:end]
    return out

def find_codes(chunk: str) -> List[str]:
    """Liste des codes CIM-10 entre parenthèses dans un bloc, normalisés en UPPER, dédoublonnés en conservant l'ordre."""
    seen = set()
    codes = []
    for m in PAREN_CODE.finditer(chunk or ''):
        code = m.group(1).upper()
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes

def extract_fields_from_assistant(assistant_text: str) -> Tuple[str, str, List[str]]:
    """
    Retourne (dp_code, dr_code, da_codes_list)
    1) On cherche dans les sections. DP/DR = 1er code de la section, DA = tous les codes de la section.
    2) Secours DP : si vide, 1er code trouvé n'importe où.
    """
    if not assistant_text:
        return '', '', []

    sections = slice_sections(assistant_text)

    dp_list = find_codes(sections['dp'])
    dr_list = find_codes(sections['dr'])
    da_list = find_codes(sections['da'])

    dp_code = dp_list[0] if dp_list else ''
    dr_code = dr_list[0] if dr_list else ''

    if not dp_code:
        any_codes = find_codes(assistant_text)
        if any_codes:
            dp_code = any_codes[0]

    return dp_code, dr_code, da_list

def last_of_role(messages: List[dict], role: str) -> str:
    """Renvoie le contenu du DERNIER message d'un rôle donné, sinon chaîne vide."""
    for msg in reversed(messages or []):
        if msg.get('role') == role:
            return msg.get('content', '') or ''
    return ''

def iter_conversations(raw: str) -> Iterable[dict]:
    """
    Accepte :
      - un unique objet JSON {"messages":[...]}
      - une liste JSON d'objets [{"messages":[...]} , ...]
      - un fichier JSON Lines (un objet JSON par ligne)
    """
    # JSON complet (dict ou liste)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and 'messages' in data:
            yield data
            return
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'messages' in item:
                    yield item
            return
    except json.JSONDecodeError:
        pass

    # JSONL
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and 'messages' in obj:
                yield obj
        except json.JSONDecodeError:
            continue

def extract_row(conv: dict) -> Dict[str, Any]:
    messages = conv.get('messages', [])
    user_text = last_of_role(messages, 'user')
    assistant_text = last_of_role(messages, 'assistant')
    dp_code, dr_code, da_list = extract_fields_from_assistant(assistant_text)
    return {
        "text": user_text,
        "dp_code": dp_code,
        "dr_code": dr_code,
        # On sérialise la liste sous forme JSON dans la colonne CSV
        "da_codes": json.dumps(da_list, ensure_ascii=False),
    }

def main():
    parser = argparse.ArgumentParser(
        description="Extrait text (user), dp_code (diagnostic principal), dr_code (diagnostic relié), da_codes (liste) vers un CSV."
    )
    parser.add_argument("input", help="Chemin du fichier d'entrée (JSON / JSONL). Utilise '-' pour stdin.")
    parser.add_argument("output", help="Chemin du CSV de sortie.")
    args = parser.parse_args()

    # Lecture
    raw = sys.stdin.read() if args.input == '-' else io.open(args.input, 'r', encoding='utf-8').read()

    rows = []
    for conv in iter_conversations(raw):
        row = extract_row(conv)
        if row["text"]:  # n'écrit que si on a un contenu user
            rows.append(row)

    # Écriture CSV
    with io.open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["text", "dp_code", "dr_code", "da_codes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Écrit {len(rows)} ligne(s) dans {args.output}")

if __name__ == "__main__":
    main()
