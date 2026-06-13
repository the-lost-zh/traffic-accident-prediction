import csv
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "concepts_30.json"
PROMPT_PATH = BASE_DIR / "prompt_templates.json"
RAW_OUT = BASE_DIR / "results_raw.csv"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def approx_tokens(text: str) -> int:
    # Cheap approximation for non-API mode.
    return max(1, int(len(text) / 1.8))


def call_openai(system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI HTTPError: {detail}") from e

    content = parsed["choices"][0]["message"]["content"]
    usage = parsed.get("usage", {})
    in_tokens = int(usage.get("prompt_tokens", approx_tokens(system_prompt + user_prompt)))
    out_tokens = int(usage.get("completion_tokens", approx_tokens(content)))
    return content, in_tokens, out_tokens


def mock_answer(concept: str, checkpoints: list[str], method: str) -> str:
    base = f"{concept}可以理解为：{checkpoints[0]}。同时，{checkpoints[1]}。在实践中，{checkpoints[2]}。"
    if method == "A_zero_shot":
        return base + " 可以把它想成一个需要在规则和结果之间保持平衡的小系统。"
    if method == "B_standard_cot":
        return (
            f"定义：{concept}涉及{checkpoints[0]}。\n"
            f"关键点：{checkpoints[1]}，并且{checkpoints[2]}。\n"
            "例子：像学习骑车，先掌握平衡，再通过练习减少错误。"
        )

    obj = {
        "expert_draft": f"{concept}的核心是：{checkpoints[0]}；其次，{checkpoints[1]}；并且{checkpoints[2]}。",
        "knowledge_gaps_or_risks": [
            "容易只记结论而忽略适用条件",
            "可能把类比当成严格等价关系",
            "忽略边界情况导致误解",
        ],
        "analogy_explanation": "可类比为管理一支团队：目标、约束和反馈共同决定最终表现。",
        "consistency_check": {
            "passed_points": ["保留了核心定义", "保留了关键机制"],
            "risk_points": ["类比可能弱化数学或法律上的严格条件"],
        },
        "final_answer_for_beginner": base + " 你可以先记住“核心机制+限制条件+典型场景”这三件事。",
    }
    return json.dumps(obj, ensure_ascii=False)


def run():
    concepts = load_json(DATA_PATH)
    prompts = load_json(PROMPT_PATH)
    random.seed(42)

    methods = ["A_zero_shot", "B_standard_cot", "C_feynman_cot"]
    backend = "openai" if os.getenv("OPENAI_API_KEY") else "mock"

    fields = [
        "id",
        "domain",
        "concept",
        "method",
        "backend",
        "response",
        "latency_sec",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "checkpoint_1",
        "checkpoint_2",
        "checkpoint_3",
    ]

    with RAW_OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for item in concepts:
            for m in methods:
                user_prompt = prompts[m].replace("{concept}", item["concept"])
                system_prompt = prompts["shared_system"]
                start = time.perf_counter()

                if backend == "openai":
                    try:
                        response, in_tok, out_tok = call_openai(system_prompt, user_prompt)
                    except Exception:
                        # Fallback to mock to guarantee the pipeline can run end-to-end.
                        backend_used = "mock_fallback"
                        response = mock_answer(item["concept"], item["checkpoints"], m)
                        in_tok = approx_tokens(system_prompt + user_prompt)
                        out_tok = approx_tokens(response)
                    else:
                        backend_used = "openai"
                else:
                    backend_used = "mock"
                    response = mock_answer(item["concept"], item["checkpoints"], m)
                    in_tok = approx_tokens(system_prompt + user_prompt)
                    out_tok = approx_tokens(response)

                latency = round(time.perf_counter() - start, 4)
                writer.writerow(
                    {
                        "id": item["id"],
                        "domain": item["domain"],
                        "concept": item["concept"],
                        "method": m,
                        "backend": backend_used,
                        "response": response,
                        "latency_sec": latency,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "total_tokens": in_tok + out_tok,
                        "checkpoint_1": item["checkpoints"][0],
                        "checkpoint_2": item["checkpoints"][1],
                        "checkpoint_3": item["checkpoints"][2],
                    }
                )

    print(f"Done. backend={backend}, rows={len(concepts) * len(methods)}, file={RAW_OUT}")


if __name__ == "__main__":
    run()
