# -*- coding: utf-8 -*-
"""
Pilot Study - Step 1: Minimal session collection (20–30 sessions)
Models: GPT-5 (OpenAI), Claude (Anthropic), DeepSeek, Baidu ERNIE (Qianfan)

Usage:
  python pilot_collect_sessions_gpt5_claude_deepseek_baidu.py \
      --csv motion2_csv.csv \
      --t1 categories \
      --outdir sessions \
      --n_draws 12 \
      --min_domains 4 \
      --seed 42 \
      --debug 1

Notes:
- API keys via env:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, BAIDU_ACCESS_TOKEN (or BAIDU_QIANFAN_AK)
- Set DRY_RUN=1 to skip real API calls (placeholders will be written).
"""
# 在文件开头的导入部分添加
from anthropic import Anthropic
from openai import OpenAI
import os
import re
import csv
import json
import time
import random
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd

# ==================== Global Configuration ====================
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

def env(k: str) -> str:
    return os.getenv(k, "")

API_KEYS = {
    "openai": env("OPENAI_API_KEY"),
    "anthropic": env("ANTHROPIC_API_KEY"),
    "deepseek": env("DEEPSEEK_API_KEY"),
    # 直接使用 bce-v3/... access token；兼容老变量名 BAIDU_QIANFAN_AK
    "baidu_token": env("BAIDU_ACCESS_TOKEN") or env("BAIDU_QIANFAN_AK"),
}

# Debate rounds：4回合×两方=8发言
DEBATE_ROUNDS = [
    ("Pro First Speaker", "Round 1: Pro Opening Statement"),
    ("Con First Speaker", "Round 1: Con Opening Statement"),
    ("Pro Second Speaker", "Round 2: Pro Rebuttal/Supplement"),
    ("Con Second Speaker", "Round 2: Con Rebuttal/Supplement"),
    ("Pro First Speaker", "Round 3: Pro Cross-examination"),
    ("Con First Speaker", "Round 3: Con Cross-examination"),
    ("Pro Second Speaker", "Round 4: Pro Closing Statement"),
    ("Con Second Speaker", "Round 4: Con Closing Statement"),
]

# ==================== Utilities ====================
def slugify(text: str, maxlen: int = 40) -> str:
    t = re.sub(r"\s+", "_", str(text).strip())
    t = re.sub(r"[^a-zA-Z0-9_\-]+", "", t)
    return t[:maxlen]

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def make_order_vector(n_draws: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    n_zh = n_draws // 2
    n_en = n_draws - n_zh
    arr = ["zh-first"] * n_zh + ["en-first"] * n_en
    rng.shuffle(arr)
    return arr

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["Motion"].astype(str).str.len() > 0]
    df = df[df["Motion_Chinese"].astype(str).str.len() > 0]
    return df

def stratified_sample_by_categories(
    df: pd.DataFrame, cat_col: str, n_draws: int, min_domains: int, seed: int
) -> pd.DataFrame:
    """
    10_categories 列：随机选 K∈[3,5] 个领域，均匀配额抽满 n_draws，缺口用剩余样本补齐。
    """
    rng = random.Random(seed)
    df = _clean_df(df)
    if cat_col not in df.columns:
        raise ValueError(f"Column '{cat_col}' not found.")

    uniq = list(df[cat_col].dropna().astype(str).unique())
    if not uniq:
        raise ValueError(f"No valid values in '{cat_col}'.")
    K = min(max(min_domains, 3), min(5, len(uniq)))
    rng.shuffle(uniq)
    chosen = uniq[:K]

    base = n_draws // K
    rem = n_draws - base * K
    pieces = []
    for i, d in enumerate(chosen):
        block = df[df[cat_col].astype(str) == d]
        take = min(len(block), base + (1 if i < rem else 0))
        if take > 0:
            pieces.append(block.sample(n=take, random_state=seed + i))
    sampled = pd.concat(pieces, axis=0) if pieces else pd.DataFrame(columns=df.columns)

    if len(sampled) < n_draws:
        rest = df.loc[~df.index.isin(sampled.index)]
        need = n_draws - len(sampled)
        if need > 0 and len(rest) > 0:
            sampled = pd.concat([sampled, rest.sample(n=min(need, len(rest)), random_state=seed+999)], axis=0)

    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def stratified_sample_by_controversy(
    df: pd.DataFrame, col: str, n_draws: int, seed: int
) -> pd.DataFrame:
    """
    High_or_low 列：尽量 High/Low 各占一半；类内不够则用另一类补齐；最后再用全局余量补齐。
    """
    df = _clean_df(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")
    hi = df[df[col].astype(str).str.lower().isin(["high", "highly", "1", "high_controversy", "highly_controversial"])]
    lo = df[df[col].astype(str).str.lower().isin(["low", "0", "low_controversy", "less_controversial"])]

    # 如果标记值比较杂，也允许把不在上述集合的都并入“其它”类先补位
    others = df.loc[~df.index.isin(hi.index.union(lo.index))]

    n_hi = n_draws // 2
    n_lo = n_draws - n_hi

    parts = []
    if len(hi) >= n_hi:
        parts.append(hi.sample(n=n_hi, random_state=seed+1))
    else:
        parts.append(hi)
        n_lo += (n_hi - len(hi))  # 需要在另一类补

    if len(lo) >= n_lo:
        parts.append(lo.sample(n=n_lo, random_state=seed+2))
    else:
        parts.append(lo)
        # 不足部分用 others 或 hi 补
        shortage = n_lo - len(lo)
        pool = others if len(others) > 0 else hi
        if shortage > 0 and len(pool) > 0:
            parts.append(pool.sample(n=min(shortage, len(pool)), random_state=seed+3))

    sampled = pd.concat(parts, axis=0).drop_duplicates()
    if len(sampled) < n_draws:
        rest = df.loc[~df.index.isin(sampled.index)]
        need = n_draws - len(sampled)
        if need > 0 and len(rest) > 0:
            sampled = pd.concat([sampled, rest.sample(n=min(need, len(rest)), random_state=seed+999)], axis=0)

    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

# ==================== Providers (HTTP) ====================

def call_openai_gpt5(prompt: str) -> str:
    """使用OpenAI官方SDK调用API"""
    if DRY_RUN:
        return f"[DRY_RUN GPT-5] {prompt[:140]} ..."
    if not API_KEYS["openai"]:
        return "[OpenAI API key missing]"

    try:
        client = OpenAI(api_key=API_KEYS["openai"])

        completion = client.chat.completions.create(
            model="gpt-5",  # 改为实际存在的模型
            messages=[
                {"role": "user", "content": prompt}
            ],
            timeout=90  # 可选：增加超时时间
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"[OpenAI error] {str(e)[:200]}"



# 修改 call_anthropic_claude 函数
def call_anthropic_claude(prompt: str, model: str = "claude-sonnet-4-5-20250929") -> str:
    """使用 Anthropic 官方 SDK 调用 API"""
    if DRY_RUN:
        return f"[DRY_RUN Claude] {prompt[:140]} ..."
    if not API_KEYS["anthropic"]:
        return "[Anthropic API key missing]"

    try:
        client = Anthropic(api_key=API_KEYS["anthropic"])

        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message.content[0].text

    except Exception as e:
        return f"[Anthropic error] {str(e)[:200]}"

def call_deepseek(prompt: str, model: str = "deepseek-chat") -> str:
    if DRY_RUN:
        return f"[DRY_RUN DeepSeek] {prompt[:140]} ..."
    if not API_KEYS["deepseek"]:
        return "[DeepSeek API key missing]"

    try:
        client = OpenAI(
            api_key=API_KEYS["deepseek"],
            base_url="https://api.deepseek.com"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=60
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"[DeepSeek error] {str(e)[:200]}"

def call_baidu_ernie(prompt: str, model: str = "ernie-4.5-turbo-128k") -> str:
    """
    使用已直接持有的 Baidu Qianfan access_token (形如 bce-v3/ALTAK-...)。无需 AK/SK。
    """
    if DRY_RUN:
        return f"[DRY_RUN Baidu] {prompt[:140]} ..."
    token = API_KEYS.get("baidu_token")
    if not token:
        return "[Baidu access_token missing] Set BAIDU_ACCESS_TOKEN"
    url = "https://qianfan.baidubce.com/v2/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            js = r.json()
            if "result" in js:
                return js["result"]
            if "choices" in js and js["choices"]:
                return js["choices"][0].get("message", {}).get("content", "")
            return json.dumps(js, ensure_ascii=False)[:200]
        return f"[Baidu error {r.status_code}] {r.text[:200]}"
    except Exception as e:
        return f"[Baidu exception] {e}"

# ==================== Debate Core (one session) ====================
ROLES_ORDER = ["Pro First Speaker", "Con First Speaker", "Pro Second Speaker", "Con Second Speaker"]
MODEL_POOL = ["GPT-5", "Claude", "DeepSeek", "Baidu"]

def role_permutation(seed: Optional[int] = None) -> Dict[str, str]:
    rng = random.Random(seed)
    names = MODEL_POOL[:]
    rng.shuffle(names)
    return dict(zip(ROLES_ORDER, names))

def strategy_hint(round_name: str) -> str:
    if "Opening" in round_name:
        return ("Opening strategy: state a clear claim, give your strongest reason "
                "or example, anticipate one counter-point.")
    if "Rebuttal" in round_name or "Cross" in round_name:
        return ("Rebuttal strategy: target a core point from the previous speaker "
                "with evidence; expose a contradiction or missing premise.")
    if "Closing" in round_name:
        return ("Closing strategy: synthesize the debate, highlight unanswered issues, "
                "acknowledge trade-offs, end with a decisive takeaway.")
    return ("General strategy: respond concisely and advance your side with logic.")

def build_prompt(topic: str, role: str, round_name: str, previous: str = "") -> str:
    rules = (
        "Output rules:\n"
        "- Plain text only. No markdown or special formatting.\n"
        "- No greetings, no role/topic restatement.\n"
        "- Opening/Rebuttal/Cross-examination: exactly 5 sentences (<=50 words each).\n"
        "- Closing: exactly 4 sentences (<=50 words each).\n"
        "Content:\n"
        "1) Start with a clear claim.\n"
        "2) Give your strongest reason or concrete example (no fabricated stats).\n"
        "3) Directly address one key point from the previous speaker (quote <=10 words), then rebut.\n"
        "4) Weigh risks/tradeoffs and explain why your side minimizes the worst plausible outcome.\n"
        "5) Final sentence: either a sharp question or decisive takeaway.\n"
    )
    pos = "Support the proposition" if "Pro" in role else "Oppose the proposition"
    prev = f"\nPrevious speaker said:\n{previous}\n" if previous else ""
    return (
        f"You are participating in a debate about: {topic}\n"
        f"Your role: {role}. {pos}.\n"
        f"Current round: {round_name}\n"
        f"{prev}\n"
        f"{rules}\n"
        f"Strategy: {strategy_hint(round_name)}\n"
        f"Please deliver your statement now."
    )

def call_model(model_name: str, prompt: str) -> str:
    if model_name == "GPT-5":
        return call_openai_gpt5(prompt)
    if model_name == "Claude":
        return call_anthropic_claude(prompt)
    if model_name == "DeepSeek":
        return call_deepseek(prompt)
    if model_name == "Baidu":
        return call_baidu_ernie(prompt)
    return "[Unknown model]"

@dataclass
class SessionResult:
    debate_id: str
    roles: Dict[str, str]
    records: List[Dict]

def run_one_session(topic_text: str, topic_number: str, lang: str, debug: bool=False) -> SessionResult:
    """单语言会话（8发言）；每场独立实例，保证会话隔离。"""
    debate_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    roles = role_permutation()  # 独立均匀置换
    if debug:
        print(f"[Session start] debate_id={debate_id}, lang={lang}, roles={roles}")

    history: List[Tuple[str, str]] = []  # (speaker, text)
    detailed: List[Dict] = []

    for i, (role, round_name) in enumerate(DEBATE_ROUNDS):
        model_name = roles[role]
        prev_text = history[-1][1] if i > 0 else ""
        prompt = build_prompt(topic_text, role, round_name, previous=prev_text)
        t0 = time.time()
        text = call_model(model_name, prompt).strip()
        dt = time.time() - t0

        detailed.append({
            "debate_id": debate_id,
            "topic_number": topic_number,
            "topic": topic_text,
            "round_number": i + 1,
            "round_name": round_name,
            "speaker_role": role,
            "ai_model": model_name,
            "position": "Pro" if "Pro" in role else "Con",
            "final_speech": text,
            "speech_length": len(text),
            "timestamp": datetime.now().isoformat(),
            "previous_speaker": history[-1][0] if i > 0 else "",
            "response_time_seconds": round(dt, 2),
        })
        history.append((f"{model_name}({role})", text))

        if debug:
            print(f"  - {round_name} | {model_name} -> {len(text)} chars in {dt:.2f}s")

        if not DRY_RUN:
            time.sleep(random.uniform(1.1, 2.0))  # 轻微限速

    return SessionResult(debate_id=debate_id, roles=roles, records=detailed)

# ==================== Text Export ====================
def render_transcript(records: List[Dict]) -> str:
    lines = []
    for r in records:
        lines.append(f"[{r['round_name']}] {r['ai_model']} ({r['speaker_role']}): {r['final_speech']}")
    return "\n".join(lines)

def write_index_row(index_csv: str, row: Dict):
    write_header = not os.path.exists(index_csv)
    with open(index_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "draw_id", "debate_id", "domain", "order", "session_lang",
            "roles", "models", "motion_en", "motion_zh", "outfile", "timestamp"
        ])
        if write_header:
            w.writeheader()
        w.writerow(row)

# ==================== Pilot Runner ====================
def run_pilot(csv_path: str, t1: str, outdir: str, n_draws: int, min_domains: int, seed: int, debug: bool):
    random.seed(seed)
    ensure_outdir(outdir)

    df = pd.read_csv(csv_path)
    if "Motion" not in df.columns or "Motion_Chinese" not in df.columns:
        raise ValueError("CSV must contain columns: 'Motion' and 'Motion_Chinese'.")

    # 选择分层列
    if t1 == "categories":
        strat_col = "10_categories"
        if strat_col not in df.columns:
            raise ValueError("CSV must contain column: '10_categories'.")
        draws = stratified_sample_by_categories(df, strat_col, n_draws=n_draws, min_domains=min_domains, seed=seed)
    elif t1 == "controversy":
        strat_col = "High_or_low"
        if strat_col not in df.columns:
            raise ValueError("CSV must contain column: 'High_or_low'.")
        draws = stratified_sample_by_controversy(df, strat_col, n_draws=n_draws, seed=seed)
    else:
        raise ValueError("--t1 must be one of {'categories', 'controversy'}")

    orders = make_order_vector(len(draws), seed=seed)

    index_csv = os.path.join(outdir, "sessions_index.csv")
    zh_first, en_first = 0, 0

    for i, row in draws.reset_index(drop=True).iterrows():
        motion_en = str(row["Motion"])
        motion_zh = str(row["Motion_Chinese"])
        domain = str(row[strat_col])  # 记录所用分层标签（领域或争议度）
        draw_id = f"{i+1:03d}"

        order = orders[i]
        if order == "zh-first":
            sequence = [("zh", motion_zh), ("en", motion_en)]
            zh_first += 1
        else:
            sequence = [("en", motion_en), ("zh", motion_zh)]
            en_first += 1

        domain_slug = slugify(domain) if domain else "NA"
        order_tag = "zhfirst" if order == "zh-first" else "enfirst"

        for lang, topic_text in sequence:
            topic_number = f"{draw_id}_{lang}"
            sess = run_one_session(topic_text=topic_text, topic_number=topic_number, lang=lang, debug=debug)

            transcript = render_transcript(sess.records)
            outfile = os.path.join(outdir, f"s{draw_id}_{lang}_{domain_slug}_{order_tag}_{sess.debate_id}.txt")
            with open(outfile, "w", encoding="utf-8") as f:
                f.write(transcript)

            write_index_row(index_csv, {
                "draw_id": draw_id,
                "debate_id": sess.debate_id,
                "domain": domain,
                "order": order,
                "session_lang": lang,
                "roles": json.dumps(sess.roles, ensure_ascii=False),
                "models": ",".join(sess.roles.values()),
                "motion_en": motion_en,
                "motion_zh": motion_zh,
                "outfile": outfile,
                "timestamp": datetime.now().isoformat(),
            })

            if debug:
                print(f"[Saved] {outfile}")

            if not DRY_RUN:
                time.sleep(random.uniform(0.8, 1.6))

    print("\n=== Pilot Summary ===")
    print(f"- Draws (pairs): {len(draws)}")
    print(f"- Sessions saved: {len(draws) * 2} -> in {outdir}/")
    vc = draws[strat_col].value_counts()
    print(f"- T1 ({strat_col}) coverage: {len(vc)} groups")
    print(vc.to_string())
    print(f"- Order split: zh-first={zh_first}, en-first={en_first}")
    print(f"- Index file : {index_csv}")
    if DRY_RUN:
        print("[NOTE] DRY_RUN=1 (API calls skipped; placeholder text written.)")

# ==================== CLI ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="motion2_csv.csv",
                        help="Path to CSV (must contain Motion, Motion_Chinese, and 10_categories/High_or_low).")
    parser.add_argument("--t1", default="categories", choices=["categories", "controversy"],
                        help="Use 10_categories (categories) or High_or_low (controversy) for stratification.")
    parser.add_argument("--outdir", default="sessions", help="Directory to save session .txt files")
    parser.add_argument("--n_draws", type=int, default=12, help="Number of motion pairs to run (10–15 recommended)")
    parser.add_argument("--min_domains", type=int, default=4, help="Minimal distinct categories to cover (3–5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", type=int, default=0, help="Debug prints (1=yes)")
    args = parser.parse_args()

    run_pilot(
        csv_path=args.csv,
        t1=args.t1,
        outdir=args.outdir,
        n_draws=args.n_draws,
        min_domains=args.min_domains,
        seed=args.seed,
        debug=bool(args.debug),
    )
