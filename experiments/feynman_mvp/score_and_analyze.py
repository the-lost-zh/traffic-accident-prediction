import csv
import json
import statistics
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "results_raw.csv"
SCORED_PATH = BASE_DIR / "results.csv"
SUMMARY_PATH = BASE_DIR / "summary_by_method.csv"
REPORT_PATH = BASE_DIR / "conclusion.md"


COMMON_CHARS = set("的一是不了在人有我他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长知民样现")
DOMAIN_TERMS = {
    "ML": ["模型", "参数", "损失", "梯度", "正则化", "泛化", "训练", "测试", "注意力"],
    "Physics": ["力", "能量", "动量", "熵", "电磁", "相对论", "量子", "振动"],
    "Medicine": ["免疫", "炎症", "细菌", "病毒", "血糖", "胰岛素", "肿瘤", "耐药"],
    "Law": ["合同", "证据", "侵权", "无罪", "程序", "举证", "正当防卫", "知识产权"],
}


def extract_final_answer(method: str, response: str) -> str:
    if method != "C_feynman_cot":
        return response
    try:
        obj = json.loads(response)
        return str(obj.get("final_answer_for_beginner", response))
    except json.JSONDecodeError:
        return response


def split_sentences(text: str) -> list[str]:
    separators = "。！？；\n"
    cur = []
    parts = []
    for ch in text:
        cur.append(ch)
        if ch in separators:
            s = "".join(cur).strip()
            if s:
                parts.append(s)
            cur = []
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def readability_metrics(text: str, domain: str) -> tuple[float, float, float]:
    clean = text.replace(" ", "")
    if not clean:
        return 0.0, 0.0, 0.0

    sentences = split_sentences(clean)
    avg_sentence_len = len(clean) / max(1, len(sentences))

    terms = DOMAIN_TERMS.get(domain, [])
    term_hits = sum(clean.count(t) for t in terms)
    term_density = term_hits / max(1, len(clean))

    difficulties = []
    for ch in clean:
        if "\u4e00" <= ch <= "\u9fff":
            difficulties.append(0 if ch in COMMON_CHARS else 1)
    freq_difficulty = statistics.median(difficulties) if difficulties else 0.5
    return avg_sentence_len, term_density, freq_difficulty


def normalize_simplicity(avg_sentence_len: float, term_density: float, freq_difficulty: float) -> float:
    # Higher score = easier to understand
    # Sentence score: ideal in [15, 28], beyond that decreases.
    if avg_sentence_len < 15:
        sentence_score = max(0.0, avg_sentence_len / 15)
    elif avg_sentence_len <= 28:
        sentence_score = 1.0
    else:
        sentence_score = max(0.0, 1.0 - (avg_sentence_len - 28) / 35)

    # Term density score: lower is better
    term_score = max(0.0, 1.0 - min(1.0, term_density * 8))
    # Frequency difficulty score: lower is better
    freq_score = max(0.0, 1.0 - freq_difficulty)

    # Weights from plan: sentence 30%, term density 40%, frequency 30%
    return round((0.3 * sentence_score + 0.4 * term_score + 0.3 * freq_score) * 100, 2)


def accuracy_label(answer: str, checkpoints: list[str]) -> tuple[float, str]:
    hit = 0
    for cp in checkpoints:
        cp_keywords = [w for w in cp.replace("，", "").replace("。", "").split(" ") if w]
        # Chinese checkpoints are short; simple substring match works for MVP.
        if cp in answer:
            hit += 1
        elif len(cp_keywords) > 1 and all(k in answer for k in cp_keywords[:2]):
            hit += 1
    ratio = hit / max(1, len(checkpoints))
    if ratio >= 0.67:
        label = "正确"
    elif ratio > 0:
        label = "部分正确"
    else:
        label = "错误"
    return round(ratio, 4), label


def load_rows():
    with RAW_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    methods = sorted({r["method"] for r in rows})
    out = []
    for m in methods:
        mr = [r for r in rows if r["method"] == m]
        acc = statistics.mean(float(r["accuracy_ratio"]) for r in mr)
        simp = statistics.mean(float(r["simplicity_score"]) for r in mr)
        tok = statistics.mean(float(r["total_tokens"]) for r in mr)
        lat = statistics.mean(float(r["latency_sec"]) for r in mr)
        out.append(
            {
                "method": m,
                "samples": len(mr),
                "accuracy_avg": round(acc, 4),
                "simplicity_avg": round(simp, 2),
                "avg_total_tokens": round(tok, 2),
                "avg_latency_sec": round(lat, 4),
            }
        )
    return out


def decision(summary_rows: list[dict]) -> dict:
    by_m = {r["method"]: r for r in summary_rows}
    b = by_m["B_standard_cot"]
    c = by_m["C_feynman_cot"]
    simplicity_gain_pct = (float(c["simplicity_avg"]) - float(b["simplicity_avg"])) / max(1e-9, float(b["simplicity_avg"])) * 100
    accuracy_drop_pct = (float(b["accuracy_avg"]) - float(c["accuracy_avg"])) * 100
    token_mult = float(c["avg_total_tokens"]) / max(1e-9, float(b["avg_total_tokens"]))
    passed = simplicity_gain_pct >= 10 and accuracy_drop_pct <= 3 and token_mult <= 2.5
    return {
        "simplicity_gain_pct": round(simplicity_gain_pct, 2),
        "accuracy_drop_pct": round(accuracy_drop_pct, 2),
        "token_multiplier": round(token_mult, 2),
        "passed": passed,
    }


def write_report(summary_rows: list[dict], d: dict):
    lines = []
    lines.append("# 费曼最小实验结论")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("- 样本量：30个概念，三组方法共90条输出")
    lines.append("- 方法：A=Zero-shot，B=标准CoT，C=四阶段Feynman-CoT")
    lines.append("- 指标：准确性（核对点命中）、通俗性综合分、Token成本")
    lines.append("")
    lines.append("## 方法均值")
    for r in summary_rows:
        lines.append(
            f"- {r['method']}: accuracy={r['accuracy_avg']}, simplicity={r['simplicity_avg']}, tokens={r['avg_total_tokens']}, latency={r['avg_latency_sec']}s"
        )
    lines.append("")
    lines.append("## 主判定阈值检验（C 相对 B）")
    lines.append(f"- 通俗性提升：{d['simplicity_gain_pct']}%（阈值 >= 10%）")
    lines.append(f"- 准确性下降：{d['accuracy_drop_pct']}%（阈值 <= 3%）")
    lines.append(f"- Token成本倍数：{d['token_multiplier']}x（阈值 <= 2.5x）")
    lines.append("")
    lines.append("## 结论")
    if d["passed"]:
        lines.append("- 通过MVP阈值：可进入下一轮消融实验（Level 2）。")
    else:
        lines.append("- 未通过MVP阈值：建议先优化提示词与阶段结构，再进入消融实验。")
    lines.append("- 注意：本轮准确性标注采用MVP轻量规则，可在下一轮替换为严格人工双人标注。")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    raw_rows = load_rows()
    scored_rows = []
    for r in raw_rows:
        answer = extract_final_answer(r["method"], r["response"])
        checkpoints = [r["checkpoint_1"], r["checkpoint_2"], r["checkpoint_3"]]
        acc_ratio, acc_label = accuracy_label(answer, checkpoints)
        avg_len, term_density, freq_diff = readability_metrics(answer, r["domain"])
        simp_score = normalize_simplicity(avg_len, term_density, freq_diff)
        row = dict(r)
        row.update(
            {
                "final_answer": answer,
                "accuracy_ratio": acc_ratio,
                "accuracy_label": acc_label,
                "avg_sentence_len": round(avg_len, 4),
                "term_density": round(term_density, 6),
                "freq_difficulty": round(freq_diff, 4),
                "simplicity_score": simp_score,
            }
        )
        scored_rows.append(row)

    fieldnames = list(scored_rows[0].keys())
    write_csv(SCORED_PATH, scored_rows, fieldnames)

    summary_rows = summarize(scored_rows)
    summary_fields = list(summary_rows[0].keys())
    write_csv(SUMMARY_PATH, summary_rows, summary_fields)

    d = decision(summary_rows)
    write_report(summary_rows, d)
    print(f"Scored {len(scored_rows)} rows. Summary={SUMMARY_PATH.name}, Report={REPORT_PATH.name}")


if __name__ == "__main__":
    main()
