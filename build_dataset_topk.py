#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import io
import json
import re
import sys
from typing import Iterable, List, Dict, Any, Tuple
from collections import Counter

# --- Regex codes CIM-10 ---
CIM10_CODE = r'[A-TV-Z]\d{2}(?:[A-Z0-9]{0,4}|\.[A-Z0-9]{1,3})'
PAREN_CODE = re.compile(r'\(\s*(' + CIM10_CODE + r')\s*\)', flags=re.IGNORECASE)

# --- Repères de sections ---
SEC_DP = re.compile(r'diagnostic\s*principal\s*:?', re.IGNORECASE | re.UNICODE)
SEC_DR = re.compile(r'diagnostic(?:s)?\s*reli[eé]s?\s*:?', re.IGNORECASE | re.UNICODE)
SEC_DA = re.compile(r'diagnostic(?:s)?\s*associ[eé]s?\s*:?', re.IGNORECASE | re.UNICODE)

SECTION_ORDER = [('dp', SEC_DP), ('dr', SEC_DR), ('da', SEC_DA)]

def slice_sections(text: str) -> Dict[str, str]:
    positions = []
    for key, pat in SECTION_ORDER:
        m = pat.search(text or "")
        if m:
            positions.append((key, m.start(), m.end()))
    positions.sort(key=lambda x: x[1])
    out = {'dp': '', 'dr': '', 'da': ''}
    for i, (key, _start, endhdr) in enumerate(positions):
        start = endhdr
        end = positions[i + 1][1] if i + 1 < len(positions) else len(text)
        out[key] = (text or "")[start:end]
    return out

def find_codes(chunk: str) -> List[str]:
    seen = set()
    codes = []
    for m in PAREN_CODE.finditer(chunk or ''):
        code = m.group(1).upper()
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes

def extract_fields_from_assistant(assistant_text: str) -> Tuple[str, str, List[str]]:
    """Retourne (dp_code, dr_code, da_list) depuis le texte de l'assistant."""
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
    for msg in reversed(messages or []):
        if msg.get('role') == role:
            return msg.get('content', '') or ''
    return ''

def iter_conversations(raw: str) -> Iterable[dict]:
    """Accepte un JSON unique, une liste JSON, ou JSON Lines."""
    # Essai JSON complet
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
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and 'messages' in obj:
                yield obj
        except json.JSONDecodeError:
            continue

def normalize_and_truncate(code: str, n: int) -> str:
    """Upper + troncature à n>0; laisse vide si code vide."""
    if not code:
        return ''
    code = str(code).upper().strip()
    if n and n > 0:
        return code[:n]
    return code

def normalize_list_and_truncate(codes: List[str], n: int) -> List[str]:
    out = []
    for c in codes or []:
        c2 = normalize_and_truncate(c, n)
        if c2:
            out.append(c2)
    return out

def prepare_rows(raw: str, n_truncate: int) -> List[Dict[str, Any]]:
    """Lit les conversations -> lignes de base avec codes éventuellement tronqués."""
    rows = []
    for conv in iter_conversations(raw):
        messages = conv.get('messages', [])
        user_text = last_of_role(messages, 'user')
        assistant_text = last_of_role(messages, 'assistant')
        dp, dr, da = extract_fields_from_assistant(assistant_text)

        dp = normalize_and_truncate(dp, n_truncate)
        dr = normalize_and_truncate(dr, n_truncate)
        da = normalize_list_and_truncate(da, n_truncate)

        if user_text:  # on garde seulement si un texte user existe
            rows.append({
                "text": user_text,
                "dp_code": dp,
                "dr_code": dr,
                "da_list": da,  # temporaire (liste Python), sera sérialisé en fin
            })
    return rows

def collect_counts(rows: List[Dict[str, Any]], scope: str) -> Counter:
    """Compte les occurrences selon le scope: dp | dr | da | any."""
    cnt = Counter()
    for r in rows:
        if scope == 'dp':
            if r['dp_code']:
                cnt[r['dp_code']] += 1
        elif scope == 'dr':
            if r['dr_code']:
                cnt[r['dr_code']] += 1
        elif scope == 'da':
            for c in r['da_list']:
                cnt[c] += 1
        elif scope == 'any':
            if r['dp_code']:
                cnt[r['dp_code']] += 1
            if r['dr_code']:
                cnt[r['dr_code']] += 1
            for c in r['da_list']:
                cnt[c] += 1
        else:
            raise ValueError("scope invalide (dp|dr|da|any).")
    return cnt

def row_matches_topk(r: Dict[str, Any], scope: str, topk_set: set) -> bool:
    """Retourne True si la ligne possède au moins un code du Top-K selon le scope."""
    if scope == 'dp':
        return r['dp_code'] in topk_set
    if scope == 'dr':
        return r['dr_code'] in topk_set
    if scope == 'da':
        return any(c in topk_set for c in r['da_list'])
    # any
    return (r['dp_code'] in topk_set) or (r['dr_code'] in topk_set) or any(c in topk_set for c in r['da_list'])

def prune_row_to_topk(r: Dict[str, Any], topk_set: set) -> None:
    """Supprime des colonnes les codes qui ne sont pas dans le Top-K (in-place)."""
    if r['dp_code'] not in topk_set:
        r['dp_code'] = ''
    if r['dr_code'] not in topk_set:
        r['dr_code'] = ''
    r['da_list'] = [c for c in r['da_list'] if c in topk_set]

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Construit un CSV filtré par Top-K codes (après troncation éventuelle). "
            "Garde uniquement les séjours contenant >=1 code du Top-K selon le scope choisi."
        )
    )
    parser.add_argument("input", help="Fichier d'entrée JSON/JSONL ou '-' pour stdin.")
    parser.add_argument("output", help="Fichier CSV de sortie.")
    parser.add_argument("--truncate", type=int, default=0,
                        help="Nombre de caractères à conserver au début des codes (0 = pas de troncature).")
    parser.add_argument("--topk", type=int, required=True,
                        help="Nombre de codes les plus fréquents à conserver.")
    parser.add_argument("--scope", choices=["dp", "dr", "da", "any"], default="dp",
                        help="Champ utilisé pour calculer les fréquences et filtrer les séjours (par défaut: dp).")
    parser.add_argument("--prune-non-topk", action="store_true",
                        help="Si présent, supprime des colonnes les codes qui ne sont pas dans le Top-K.")
    args = parser.parse_args()

    # Lecture d'entrée
    raw = sys.stdin.read() if args.input == '-' else io.open(args.input, 'r', encoding='utf-8').read()

    # 1) Extraction + troncature
    rows = prepare_rows(raw, n_truncate=args.truncate)

    if not rows:
        # Écrit tout de même un CSV avec en-tête
        with io.open(args.output, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["code_sejour", "text", "dp_code", "dr_code", "da_codes"])
            writer.writeheader()
        print("0")
        return

    # 2) Comptage et Top-K
    counts = collect_counts(rows, scope=args.scope)
    if not counts:
        # Si aucun code dans le scope, rien à garder
        with io.open(args.output, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["code_sejour", "text", "dp_code", "dr_code", "da_codes"])
            writer.writeheader()
        print("0")
        return

    # Tri par fréquence desc puis lexicographique pour stabilité
    sorted_codes = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    topk_list = [c for c, _ in sorted_codes[:args.topk]]
    topk_set = set(topk_list)

    # 3) Filtrage des lignes : garder uniquement celles qui matchent Top-K selon le scope
    filtered = []
    for r in rows:
        if row_matches_topk(r, scope=args.scope, topk_set=topk_set):
            if args.prune_non_topk:
                prune_row_to_topk(r, topk_set)
            filtered.append(r)

    # 4) Numérotation et écriture CSV
    with io.open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["code_sejour", "text", "dp_code", "dr_code", "da_codes"])
        writer.writeheader()
        for idx, r in enumerate(filtered, start=1):
            writer.writerow({
                "code_sejour": idx,
                "text": r["text"],
                "dp_code": r["dp_code"],
                "dr_code": r["dr_code"],
                "da_codes": json.dumps(r["da_list"], ensure_ascii=False),
            })

    # 5) Impression du nombre de lignes
    print(str(len(filtered)))

if __name__ == "__main__":
    main()
