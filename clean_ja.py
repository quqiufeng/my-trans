#!/usr/bin/env python3
# clean_ja.py — 用社区日语幻觉正则清洗 ASS（原地改写）
# 规则来自 WhisperJAV regexp_v09.json。策略：
#   * 括号/方括号/星号类「音效标注」→ 全局剥离（(笑) (音楽) 【字幕】 ★...★）
#   * 其它短语类 → 仅当「整行完全匹配」时才丢弃，避免误删真台词里的 笑/拍手 等字
import json, re, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = os.path.join(HERE, "res", "hallucination_regexp_ja.json")
# 以字面括号/方括号/星号开头的模式，可安全全局剥离
STRIP = re.compile(r"^(\\\(|\\（|【|★)")


def main(path):
    data = json.load(open(RULE, encoding="utf-8"))
    strip_pats, drop_pats = [], []
    for p in data["patterns"]:
        rx = re.compile(p["pattern"])
        (strip_pats if STRIP.match(p["pattern"]) else drop_pats).append(rx)

    lines = open(path, encoding="utf-8").read().splitlines()
    out, dropped, changed = [], 0, 0
    for line in lines:
        if not line.startswith("Dialogue:"):
            out.append(line)
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            out.append(line)
            continue
        text = parts[9]
        cleaned = text
        for p in strip_pats:
            cleaned = p.sub("", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
        core = re.sub(r"\\N", "", cleaned).strip()
        if core == "" or any(p.fullmatch(core) for p in drop_pats):
            dropped += 1
            continue
        if cleaned != text:
            changed += 1
            parts[9] = cleaned
            out.append(",".join(parts))
        else:
            out.append(line)
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"[clean_ja] {os.path.basename(path)}: 丢弃 {dropped}, 修改 {changed}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法: clean_ja.py <file.ass>")
    main(sys.argv[1])
