#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import sys
import warnings
from collections import Counter
from typing import Dict, List, Tuple, Set, Any

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*Sliding Window Attention.*")
warnings.filterwarnings("ignore", message=".*Setting `pad_token_id`.*")

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
except ImportError:
    AutoModelForCausalLM, AutoTokenizer, torch = None, None, None
    BitsAndBytesConfig = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------------
# Base color families
# ---------------------------

BASIC_COLORS = {
    "red","green","blue","yellow","black","white","orange","purple","brown","gray","grey","pink"
}

# Core extended roots (used for -ish, -colored and compounds resolution)
COLOR_ROOTS = {
    "red","green","blue","yellow","black","white","orange","purple","brown","gray","grey","pink",
    "cyan","magenta","teal","indigo","violet","maroon","burgundy","navy","beige","tan","taupe",
    "ivory","cream","gold","silver","bronze","copper","khaki","mustard","olive","lime",
    "turquoise","azure","cerulean","cobalt","sapphire","emerald","jade","ruby","amethyst","topaz","amber",
    "coral","peach","apricot","salmon","rose","lavender","lilac","mauve","fuchsia","orchid","plum","eggplant",
    "charcoal","slate","graphite","gunmetal","ash","smoke","mint","sage","moss","forest","sea","sky",
    "sand","ecru","ochre","ocher","cream","ivory"
}

# CSS/X11 named colors (complete list, lowercased)
CSS_X11 = {
    "aliceblue","antiquewhite","aqua","aquamarine","azure","beige","bisque","blanchedalmond","blueviolet",
    "burlywood","cadetblue","chartreuse","chocolate","coral","cornflowerblue","cornsilk","crimson","cyan",
    "darkblue","darkcyan","darkgoldenrod","darkgray","darkgreen","darkgrey","darkkhaki","darkmagenta","darkolivegreen",
    "darkorange","darkorchid","darkred","darksalmon","darkseagreen","darkslateblue","darkslategray","darkslategrey",
    "darkturquoise","darkviolet","deeppink","deepskyblue","dimgray","dimgrey","dodgerblue","firebrick","floralwhite",
    "forestgreen","fuchsia","gainsboro","ghostwhite","gold","goldenrod","greenyellow","honeydew","hotpink","indianred",
    "indigo","ivory","khaki","lavender","lavenderblush","lawngreen","lemonchiffon","lightblue","lightcoral",
    "lightcyan","lightgoldenrodyellow","lightgray","lightgreen","lightgrey","lightpink","lightsalmon","lightseagreen",
    "lightskyblue","lightslategray","lightslategrey","lightsteelblue","lightyellow","lime","limegreen","linen","magenta",
    "maroon","mediumaquamarine","mediumblue","mediumorchid","mediumpurple","mediumseagreen","mediumslateblue",
    "mediumspringgreen","mediumturquoise","mediumvioletred","midnightblue","mintcream","mistyrose","moccasin",
    "navajowhite","navy","oldlace","olive","olivedrab","orange","orangered","orchid","palegoldenrod","palegreen",
    "paleturquoise","palevioletred","papayawhip","peachpuff","peru","pink","plum","powderblue","rosybrown","royalblue",
    "saddlebrown","salmon","sandybrown","seagreen","seashell","sienna","silver","skyblue","slateblue","slategray",
    "slategrey","snow","springgreen","steelblue","tan","teal","thistle","tomato","turquoise","violet","wheat","whitesmoke",
    "yellowgreen","rebeccapurple"
}

# Rich nuanced list (thousands-style feel; curated to avoid egregious false positives)
NUANCED_EXTRA = {
    # deep/dark/light etc. handled by modifiers below; include popular standalones/roots
    "scarlet","vermilion","carmine","burgundy","mahogany","wine","rust","brick","terracotta","sepia",
    "amber","saffron","mustard","ochre","ocher","chartreuse","lime","olive","sage","fern","mint","seafoam","moss",
    "teal","turquoise","cyan","cerulean","azure","cobalt","navy","indigo","periwinkle","ultramarine","prussian",
    "violet","lilac","lavender","mauve","magenta","fuchsia","orchid","amethyst","heliotrope",
    "peach","apricot","coral","salmon","rose","blush","cerise","raspberry","strawberry","cherry",
    "beige","ecru","taupe","sand","tan","khaki","ivory","cream","eggshell","bone",
    "charcoal","slate","graphite","gunmetal","ash","smoke","pewter",
    "gold","silver","bronze","copper","brass","platinum","gunmetal",
    "obsidian","onyx","opal","topaz","ruby","sapphire","emerald","jade","garnet","quartz","peridot","zircon",
    "caramel","cinnamon","mocha","coffee","chocolate",
    "auburn","chestnut","hazel","hazelnut",
    "petrol","petroleum","ink","midnight",
    # standalone common multiword concepts (we also handle via COMPOUND_MAP):
    "eggplant","plum","mustard","rust",
}

# canonical aliases
ALIASES = {
    "grey": "gray",
    "darkgrey": "darkgray",
    "lightgrey": "lightgray",
    "slategrey": "slategray",
    "dimgrey": "dimgray",
    "lightslategrey": "lightslategray",
    "ocher": "ochre",
}

# Modifiers (prefixes) that may precede color roots/words
SHADE_PREFIXES = {
    "light","dark","pale","deep","bright","vivid","pastel","soft","bold","neon","matte","glossy","dusty","rich","vibrant","muted","warm","cool","burnt","electric","icy","metallic","saturated","unsaturated"
}

# Words to ignore in color sense
STOPLIKE_COLOR_ADJS = {"golden","silvery","bronzed","coppery","leaden"}

# Compound patterns map to canonical tokens (nuanced)
# including hyphen/space variants; we normalize hyphen->space before matching
COMPOUND_MAP = {
    # black/white extremes
    "jet black":"jet black","matte black":"matte black","ink black":"ink black","pitch black":"pitch black",
    "snow white":"snow white","pure white":"pure white","off white":"off white","bone white":"bone white","eggshell white":"eggshell white",
    # blue family
    "sky blue":"sky blue","baby blue":"baby blue","powder blue":"powder blue","electric blue":"electric blue",
    "royal blue":"royal blue","cobalt blue":"cobalt blue","navy blue":"navy blue","midnight blue":"midnight blue",
    "steel blue":"steel blue","cornflower blue":"cornflower blue","prussian blue":"prussian blue",
    # green family
    "forest green":"forest green","sea green":"sea green","hunter green":"hunter green","mint green":"mint green",
    "sage green":"sage green","olive green":"olive green","neon green":"neon green","emerald green":"emerald green","jade green":"jade green",
    # red/pink family
    "brick red":"brick red","blood red":"blood red","rose red":"rose red","rust red":"rust red","cherry red":"cherry red",
    "hot pink":"hot pink","dusty pink":"dusty pink","bubblegum pink":"bubblegum pink","rose pink":"rose pink",
    # orange/yellow family
    "burnt orange":"burnt orange","pumpkin orange":"pumpkin orange","apricot orange":"apricot orange",
    "lemon yellow":"lemon yellow","butter yellow":"butter yellow","mustard yellow":"mustard yellow","golden yellow":"golden yellow",
    # neutrals
    "blue gray":"blue gray","blue grey":"blue gray","green gray":"green gray","green grey":"green gray","brown gray":"brown gray","brown grey":"brown gray",
    "charcoal gray":"charcoal gray","charcoal grey":"charcoal gray","cool gray":"cool gray","warm gray":"warm gray",
    # metals/materials
    "rose gold":"rose gold","antique gold":"antique gold","brushed gold":"brushed gold","brushed silver":"brushed silver",
    "gunmetal gray":"gunmetal gray",
    # stones/gems
    "pearl white":"pearl white","ivory white":"ivory white",
    # foods/plants as colors
    "eggplant purple":"eggplant purple","plum purple":"plum purple","mocha brown":"mocha brown","chocolate brown":"chocolate brown","caramel brown":"caramel brown",
}

# From CSS_X11 + NUANCED_EXTRA build nuanced set excluding BASIC_COLORS
NUANCED_COLORS: Set[str] = set()
for c in CSS_X11 | NUANCED_EXTRA:
    if c not in BASIC_COLORS:
        NUANCED_COLORS.add(c)
# Also add compound canonical tokens
NUANCED_COLORS.update(COMPOUND_MAP.values())

# Regexes
WORD_RE = re.compile(r"[a-z]+", re.I)
ISH_RE = re.compile(r"^([a-z]+)ish$")  # bluish, reddish
COLOR_OF_RE = re.compile(r"^color\s+of\s+([a-z]+(?:\s+[a-z]+)?)$")  # "color of jade green" (rare but handle)

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("/", " ")
    text = text.replace("-", " ")  # hyphen to space for compound matching
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text)

def window_tokens(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def count_colors_in_caption(answer_text: str,
                            binary_mode: bool = False,
                            ambiguous_check: bool = False,
                            llm_classifier=None,
                            llm_full_scan=None) -> Dict[str, Any]:
    text = normalize_text(answer_text)

    # Full-scan LLM mode: boolean presence only
    if llm_full_scan is not None:
        has_basic, has_nuanced = llm_full_scan(text)
        out = {
            "basic_colors": {"count_total": 1 if has_basic else 0, "count_unique": 1 if has_basic else 0, "detected_words": [], "wordfreq": {}},
            "nuanced_colors": {"count_total": 1 if has_nuanced else 0, "count_unique": 1 if has_nuanced else 0, "detected_words": [], "wordfreq": {}},
            "binary_flags": {"has_basic": has_basic, "has_nuanced": has_nuanced},
            "llm_full_scan_used": True
        }
        if not binary_mode:
            # keep counts as booleans 0/1 even in frequency mode (no token data available here)
            pass
        return out

    tokens = tokenize_words(text)

    # First pass: detect 3-grams then 2-grams against COMPOUND_MAP (nuanced)
    consumed = [False]*len(tokens)
    set_basic: Set[str] = set()
    set_nuanced: Set[str] = set()
    wf_basic, wf_nuanced = Counter(), Counter()

    def consume_span(i, k):
        for j in range(i, i+k):
            consumed[j] = True

    # 3-gram compounds (e.g., 'cobalt blue', 'cherry red' can also be 2-gram; but handle longer first)
    for n in (3, 2):
        grams = window_tokens(tokens, n)
        for i, gram in enumerate(grams):
            if any(consumed[i:i+n]):
                continue
            phrase = " ".join(gram)
            if phrase in COMPOUND_MAP:
                can = COMPOUND_MAP[phrase]
                set_nuanced.add(can)
                wf_nuanced[can] += 1
                consume_span(i, n)

    # Next, handle "<color_root> colored/coloured" patterns as 2-grams
    grams2 = window_tokens(tokens, 2)
    for i, gram in enumerate(grams2):
        if any(consumed[i:i+2]):
            continue
        if gram[1] in ('colored', 'coloured'):
            root = ALIASES.get(gram[0], gram[0])
            if root in COLOR_ROOTS:
                if root in BASIC_COLORS:
                    has_basic = True
                    if not binary_mode:
                        set_basic.add(root)
                        wf_basic[root] += 1
                else: # It's a nuanced color root
                    has_nuanced = True
                    if not binary_mode:
                        set_nuanced.add(root)
                        wf_nuanced[root] += 1
                consume_span(i, 2)


    has_basic = False
    has_nuanced = bool(set_nuanced)

    # Single-token and prefix-modifier handling
    i = 0
    ambiguous: Set[str] = set()
    while i < len(tokens):
        if consumed[i]:
            i += 1
            continue
        t = tokens[i]

        # skip stop-like
        if t in STOPLIKE_COLOR_ADJS:
            i += 1
            continue

        # handle "<modifier> <color>"
        if t in SHADE_PREFIXES and i+1 < len(tokens) and not consumed[i+1]:
            nxt = ALIASES.get(tokens[i+1], tokens[i+1])
            if nxt in BASIC_COLORS:
                has_basic = True
                if not binary_mode:
                    set_basic.add(nxt)
                    wf_basic[nxt] += 1
                consumed[i] = consumed[i+1] = True
                i += 2
                continue
            if nxt in NUANCED_COLORS:
                has_nuanced = True
                if not binary_mode:
                    set_nuanced.add(nxt)
                    wf_nuanced[nxt] += 1
                consumed[i] = consumed[i+1] = True
                i += 2
                continue

        # "X-colored/coloured" -- this is now handled above as a 2-gram pattern

        # "-ish" colors
        m2 = ISH_RE.match(t)
        if m2:
            root = ALIASES.get(m2.group(1), m2.group(1))
            if root in BASIC_COLORS:
                has_basic = True
                if not binary_mode:
                    set_basic.add(root); wf_basic[root] += 1
            elif root in NUANCED_COLORS or root in COLOR_ROOTS:
                has_nuanced = True
                if not binary_mode:
                    set_nuanced.add(root); wf_nuanced[root] += 1
            consumed[i] = True
            i += 1
            continue

        # direct single-token matches
        can = ALIASES.get(t, t)
        if can in BASIC_COLORS:
            has_basic = True
            if not binary_mode:
                set_basic.add(can); wf_basic[can] += 1
            consumed[i] = True
            i += 1
            continue
        if can in NUANCED_COLORS:
            has_nuanced = True
            if not binary_mode:
                set_nuanced.add(can); wf_nuanced[can] += 1
            consumed[i] = True
            i += 1
            continue

        # ambiguous tokens to LLM (optional)
        if ambiguous_check and (can in COLOR_ROOTS or can in {"emerald","ruby","sapphire","amethyst","topaz","jade","opal","onyx","pearl","amber","aqua","aquamarine","beige","rose","sky","plum","eggplant"} or can.endswith("ish")):
            ambiguous.add(can)

        i += 1

    llm_classified = {}
    if ambiguous_check and ambiguous and llm_classifier is not None:
        try:
            llm_classified = llm_classifier(sorted(ambiguous))
        except Exception:
            llm_classified = {}
        for tok, cls in llm_classified.items():
            if cls == "basic":
                has_basic = True
                if not binary_mode:
                    set_basic.add(tok); wf_basic[tok] += 1
            elif cls == "nuanced":
                has_nuanced = True
                if not binary_mode:
                    set_nuanced.add(tok); wf_nuanced[tok] += 1

    # finalize counts
    if binary_mode:
        basic_total = 1 if has_basic else 0
        nuanced_total = 1 if has_nuanced else 0
    else:
        basic_total = sum(wf_basic.values())
        nuanced_total = sum(wf_nuanced.values())

    return {
        "basic_colors": {
            "count_total": basic_total,
            "count_unique": len(set_basic),
            "detected_words": sorted(set_basic),
            "wordfreq": dict(wf_basic)
        },
        "nuanced_colors": {
            "count_total": nuanced_total,
            "count_unique": len(set_nuanced),
            "detected_words": sorted(set_nuanced),
            "wordfreq": dict(wf_nuanced)
        },
        "binary_flags": {"has_basic": has_basic, "has_nuanced": has_nuanced},
        "llm_ambiguous_classified": llm_classified
    }


# ---------- LLM glue (now using Hugging Face) ----------

def build_llm_classifier(args, model, tokenizer):
    if not args.ambiguous_check or args.llm_full_scan or not model or not tokenizer:
        return None

    basic_list = ", ".join(sorted(BASIC_COLORS))
    nuanced_list = ", ".join(sorted(NUANCED_COLORS))

    def _classify(tokens: List[str]) -> Dict[str, str]:
        if not tokens: return {}
        sys_prompt = (
            "Classify each token into 'basic', 'nuanced', or 'none'. "
            "Use color sense only; whole-word semantics. "
            f"basic=[{basic_list}] ; nuanced includes [{nuanced_list}] and common shade/material/gem variants. "
            "Return JSON token->class."
        )
        user_prompt = "Tokens: " + json.dumps(tokens, ensure_ascii=False)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=args.llm_max_new_tokens
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # The model might return markdown ```json ... ```, so let's extract it.
            json_match = re.search(r"```json\s*([\s\S]+?)\s*```|({[\s\S]+})", response_text)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                return json.loads(json_str)
            return json.loads(response_text)
        except Exception as e:
            print(f"[WARN] LLM classifier failed: {e}", file=sys.stderr)
            return {}
    return _classify


def build_llm_full_scan(args, model, tokenizer):
    if not args.llm_full_scan or not model or not tokenizer:
        return None

    basic_list = ", ".join(sorted(BASIC_COLORS))
    nuanced_list = ", ".join(sorted(NUANCED_COLORS))

    sys_prompt = (
        "Read a caption and detect whether ANY color words appear. "
        "Return JSON booleans has_basic and has_nuanced. "
        f"basic_list=[{basic_list}]; nuanced_list includes [{nuanced_list}] plus common shades/materials/gems/compounds. "
        "Treat hyphens/slashes as separators; match whole words; ignore substrings like 'greenhouse'. "
        "Return ONLY JSON like {\"has_basic\":true/false, \"has_nuanced\":true/false}."
    )

    def _scan_batch(captions: List[str]) -> List[Tuple[bool, bool]]:
        prompts = []
        for caption_text in captions:
            user_prompt = "Caption: " + caption_text
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
        
        try:
            model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

            generated_ids = model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=args.llm_max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )
            # Important: Slice generated IDs to get only the response part
            generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]

            response_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            final_results = []
            for response_text in response_texts:
                try:
                    json_match = re.search(r"```json\s*([\s\S]+?)\s*```|({[\s\S]+})", response_text)
                    if json_match:
                        json_str = json_match.group(1) or json_match.group(2)
                        data = json.loads(json_str)
                    else:
                        data = json.loads(response_text)
                    final_results.append((bool(data.get("has_basic", False)), bool(data.get("has_nuanced", False))))
                except Exception:
                    final_results.append((False, False)) # JSON parsing failed for this item
            return final_results
        except Exception as e:
            print(f"[WARN] LLM full scan batch failed: {e}", file=sys.stderr)
            return [(False, False)] * len(captions)

    return _scan_batch


# ---------- File I/O ----------

def process_file(path: str, args, llm_classifier, llm_full_scan):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_entry = []
    agg_basic_total = 0
    agg_nuanced_total = 0
    agg_basic_set: Set[str] = set()
    agg_nuanced_set: Set[str] = set()
    agg_wf_basic = Counter()
    agg_wf_nuanced = Counter()
    agg_bin_basic = 0
    agg_bin_nuanced = 0

    if args.llm_full_scan and llm_full_scan:
        # --- BATCHED LLM FULL SCAN ---
        print(f"Running batched LLM full scan on {len(data)} entries with batch size {args.batch_size}...")
        captions = [item.get("answer") or "" for item in data]
        all_results = []

        # Create an iterator for mini-batches
        iterator = range(0, len(captions), args.batch_size)
        if tqdm:
            iterator = tqdm(iterator, desc=f"Scanning {os.path.basename(path)}", unit="batch")

        for i in iterator:
            batch_captions = captions[i:i + args.batch_size]
            batch_results = llm_full_scan(batch_captions)
            all_results.extend(batch_results)

        print("LLM processing complete. Aggregating results...")

        iterator = range(len(data))
        if tqdm:
            iterator = tqdm(iterator, desc=f"Aggregating {os.path.basename(path)}", unit="entry")

        for i in iterator:
            item = data[i]
            has_basic, has_nuanced = all_results[i]

            # Reconstruct the `res` dictionary for aggregation
            res = {
                "basic_colors": {
                    "count_total": 1 if has_basic else 0, "count_unique": 1 if has_basic else 0,
                    "detected_words": [], "wordfreq": {}
                },
                "nuanced_colors": {
                    "count_total": 1 if has_nuanced else 0, "count_unique": 1 if has_nuanced else 0,
                    "detected_words": [], "wordfreq": {}
                },
                "binary_flags": {"has_basic": has_basic, "has_nuanced": has_nuanced},
                "llm_full_scan_used": True,
                "llm_ambiguous_classified": {}
            }
            
            agg_basic_total += res["basic_colors"]["count_total"]
            agg_nuanced_total += res["nuanced_colors"]["count_total"]
            
            agg_bin_basic += 1 if res["binary_flags"]["has_basic"] else 0
            agg_bin_nuanced += 1 if res["binary_flags"]["has_nuanced"] else 0

            per_entry.append({
                "index": item.get("index"), "image": item.get("image"),
                "has_basic": res["binary_flags"]["has_basic"], "has_nuanced": res["binary_flags"]["has_nuanced"],
                "basic_count": res["basic_colors"]["count_total"], "nuanced_count": res["nuanced_colors"]["count_total"],
                "basic_detected": "", "nuanced_detected": "",
                "llm_full_scan_used": res.get("llm_full_scan_used", False)
            })

    else:
        # --- ORIGINAL SEQUENTIAL PROCESSING (for ambiguous_check or no LLM) ---
        iterator = data
        if tqdm:
            iterator = tqdm(data, desc=f"Processing {os.path.basename(path)}", unit="entry")

        for item in iterator:
            answer = item.get("answer") or ""
            res = count_colors_in_caption(
                answer_text=answer,
                binary_mode=args.binary_mode,
                ambiguous_check=args.ambiguous_check,
                llm_classifier=llm_classifier,
                llm_full_scan=None # full scan is handled above
            )
            agg_basic_total += res["basic_colors"]["count_total"]
            agg_nuanced_total += res["nuanced_colors"]["count_total"]
            agg_basic_set.update(res["basic_colors"]["detected_words"])
            agg_nuanced_set.update(res["nuanced_colors"]["detected_words"])
            agg_wf_basic.update(res["basic_colors"]["wordfreq"])
            agg_wf_nuanced.update(res["nuanced_colors"]["wordfreq"])

            agg_bin_basic += 1 if res["binary_flags"]["has_basic"] else 0
            agg_bin_nuanced += 1 if res["binary_flags"]["has_nuanced"] else 0

            per_entry.append({
                "index": item.get("index"),
                "image": item.get("image"),
                "has_basic": res["binary_flags"]["has_basic"],
                "has_nuanced": res["binary_flags"]["has_nuanced"],
                "basic_count": res["basic_colors"]["count_total"],
                "nuanced_count": res["nuanced_colors"]["count_total"],
                "basic_detected": ", ".join(res["basic_colors"]["detected_words"]),
                "nuanced_detected": ", ".join(res["nuanced_colors"]["detected_words"]),
                "llm_full_scan_used": res.get("llm_full_scan_used", False)
            })

    summary = {
        "file": os.path.basename(path),
        "mode": "binary" if args.binary_mode else "frequency",
        "basic_colors": {
            "count_total": agg_basic_total,
            "count_unique": len(agg_basic_set),
            "detected_words": sorted(agg_basic_set),
            "wordfreq": dict(agg_wf_basic)
        },
        "nuanced_colors": {
            "count_total": agg_nuanced_total,
            "count_unique": len(agg_nuanced_set),
            "detected_words": sorted(agg_nuanced_set),
            "wordfreq": dict(agg_wf_nuanced)
        },
        "binary_sums": {
            "has_basic_sum": agg_bin_basic,
            "has_nuanced_sum": agg_bin_nuanced,
            "num_entries": len(per_entry)
        }
    }
    return summary, per_entry

def save_outputs(summary: Dict[str, Any], per_entry: List[Dict[str, Any]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    tag = os.path.splitext(summary["file"])[0]

    summary_path = os.path.join(output_dir, f"colors_summary_{tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # wordfreq csv
    rows = []
    for w, c in summary["basic_colors"]["wordfreq"].items():
        rows.append({"word": w, "class": "basic", "count": c})
    for w, c in summary["nuanced_colors"]["wordfreq"].items():
        rows.append({"word": w, "class": "nuanced", "count": c})
    freq_path = os.path.join(output_dir, f"colors_wordfreq_{tag}.csv")
    if pd is not None and rows:
        import pandas as _pd
        _pd.DataFrame(rows).sort_values(["class","count","word"], ascending=[True, False, True]).to_csv(freq_path, index=False)
    else:
        with open(freq_path, "w", encoding="utf-8") as f:
            f.write("word,class,count\n")
            for r in sorted(rows, key=lambda r:(r["class"], -r["count"], r["word"])):
                f.write(f'{r["word"]},{r["class"]},{r["count"]}\n')

    # per-entry csv
    per_path = os.path.join(output_dir, f"colors_detected_{tag}.csv")
    if pd is not None:
        import pandas as _pd
        _pd.DataFrame(per_entry).to_csv(per_path, index=False)
    else:
        with open(per_path, "w", encoding="utf-8") as f:
            headers = ["index","image","has_basic","has_nuanced","basic_count","nuanced_count","basic_detected","nuanced_detected","llm_full_scan_used"]
            f.write(",".join(headers) + "\n")
            for row in per_entry:
                vals = [str(row.get(h,"")).replace(",", ";") for h in headers]
                f.write(",".join(vals) + "\n")

    return summary_path, freq_path, per_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="../dataset/text", help="Directory with JSONs")
    ap.add_argument("--output_dir", type=str, default="color/regex", help="Directory to save outputs")
    ap.add_argument("--input_files", nargs='+', default=["claude-3-5-sonnet_prompts.json","gemini-1.5-pro_prompts.json","gpt-4o_prompts.json"], help="Specific JSON file(s) to process inside input_dir")
    ap.add_argument("--binary_mode", action="store_true", help="Per-caption +1 if any basic/nuanced appears")
    ap.add_argument("--ambiguous_check", action="store_true", help="Enable LLM to classify ambiguous color-like words")
    ap.add_argument("--llm_full_scan", action="store_true", help="Enable LLM to scan full caption and return booleans (overrides token classifier)")
    ap.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-1.5B", help="Hugging Face model ID for Qwen model")
    ap.add_argument("--llm_max_new_tokens", type=int, default=128, help="Max new tokens for LLM generation")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for --llm_full_scan inference.")
    ap.add_argument("--quantization", type=str, default=None, choices=['4bit', '8bit'], help="Enable 4-bit or 8-bit quantization via bitsandbytes to speed up inference and reduce memory.")
    ap.add_argument("--device", type=str, default="auto", help="Device for Hugging Face model (e.g., 'cpu', 'cuda', 'mps', 'auto')")
    args = ap.parse_args()

    # Build LLM helpers
    model, tokenizer = None, None
    if args.ambiguous_check or args.llm_full_scan:
        if not all([AutoModelForCausalLM, AutoTokenizer, torch, BitsAndBytesConfig]):
            print("[ERROR] `transformers` and `torch` are required for LLM features. `bitsandbytes` is needed for quantization.", file=sys.stderr)
            args.ambiguous_check = False
            args.llm_full_scan = False
        else:
            try:
                print(f"Loading model {args.llm_model}...")
                
                quantization_config = None
                if args.quantization:
                    if args.quantization == '4bit':
                        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                    elif args.quantization == '8bit':
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    print(f"Applying {args.quantization} quantization...")

                tokenizer = AutoTokenizer.from_pretrained(args.llm_model, padding_side='left')
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # For apple silicon, mps backend does not support float16, use bfloat16 instead
                if torch.backends.mps.is_available():
                    model = AutoModelForCausalLM.from_pretrained(args.llm_model, device_map=args.device, torch_dtype=torch.bfloat16, quantization_config=quantization_config, attn_implementation="eager")
                else:
                    model = AutoModelForCausalLM.from_pretrained(args.llm_model, device_map=args.device, torch_dtype="auto", quantization_config=quantization_config, attn_implementation="eager")

                print("Model loaded.")
            except Exception as e:
                print(f"[ERROR] Failed to load Hugging Face model: {e}", file=sys.stderr)
                print("[INFO] Please ensure 'transformers', 'torch', and 'accelerate' are installed.", file=sys.stderr)
                args.ambiguous_check = False
                args.llm_full_scan = False

    llm_full_scan = build_llm_full_scan(args, model, tokenizer)
    llm_classifier = build_llm_classifier(args, model, tokenizer)

    all_outs = []
    for fname in args.input_files:
        fpath = os.path.join(args.input_dir, fname)
        if not os.path.exists(fpath):
            print(f"[WARN] Missing file: {fpath}", file=sys.stderr)
            continue
        summary, per_entry = process_file(fpath, args, llm_classifier, llm_full_scan)
        paths = save_outputs(summary, per_entry, args.output_dir)
        all_outs.append((fname, paths))

    if not all_outs:
        print("[INFO] No outputs written; check input paths.", file=sys.stderr)
        return

    print("Outputs:")
    for fname, (sp, fp, pp) in all_outs:
        print(f"  {fname}:")
        print(f"    Summary JSON  : {sp}")
        print(f"    WordFreq CSV  : {fp}")
        print(f"    Per-entry CSV : {pp}")

if __name__ == "__main__":
    main()
