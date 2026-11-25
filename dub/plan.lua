-- dub/plan.lua — 文案适配：把一份文案按视频播放时间整理成合适长度的字幕文案
-- 规则：以「整句」为最小取舍单位（绝不掐字断句）；时长不够就先轻微变速、
-- 再整句均匀抽稀；时长有余就先放慢、再把句间停顿拉长，把字幕铺满整条视频。
local M = {}

-- ── UTF-8 逐码点工具 ───────────────────────────────────────
local function chars(s)
    local t = {}
    for c in tostring(s):gmatch("[%z\1-\127\194-\244][\128-\191]*") do t[#t + 1] = c end
    return t
end
M.chars = chars

local function codepoint_at(s, i)
    local b1 = s:byte(i)
    if not b1 then return nil, i end
    if b1 < 0x80 then return b1, 1 end
    local need = (b1 >= 0xF0 and 3) or (b1 >= 0xE0 and 2) or 1
    local cp = b1 % (2 ^ (7 - need))
    for k = 1, need do
        local bk = s:byte(i + k)
        if not bk then return nil, i end
        cp = cp * 64 + (bk % 64)
    end
    return cp, need + 1
end

local PUNCT = {}
for _, cp in ipairs({
    0x20, 0x22, 0x27, 0x28, 0x29, 0x2c, 0x2e, 0x21, 0x3f, 0x3b, 0x3a, 0xb7,
    0x2018, 0x2019, 0x201c, 0x201d, 0x2014, 0x2013, 0x2026, 0x3000,
    0x3001, 0x3002, 0x3008, 0x3009, 0x300a, 0x300b, 0x300c, 0x300d, 0x300e, 0x300f,
    0x3010, 0x3011, 0xff01, 0xff08, 0xff09, 0xff0c, 0xff0d, 0xff1a, 0xff1b, 0xff1f,
}) do PUNCT[cp] = true end

local function clean_punct(s)
    s = tostring(s)
    local out, i = {}, 1
    while i <= #s do
        local cp, len = codepoint_at(s, i)
        if not cp then break end
        if not PUNCT[cp] then out[#out + 1] = s:sub(i, i + len - 1) end
        i = i + len
    end
    return table.concat(out)
end
M.clean_punct = clean_punct

-- ── 断句 ───────────────────────────────────────────────────
local SENT_END, CLAUSE_END = {}, {}
for _, cp in ipairs({ 0x3002, 0xff01, 0xff1f, 0xff1b, 0x2e, 0x21, 0x3f, 0x3b, 0x2026 }) do SENT_END[cp] = true end
-- 子句界：逗号/顿号/冒号，以及空格（口播稿常用空格代替逗号）
for _, cp in ipairs({ 0xff0c, 0x2c, 0x3001, 0xff1a, 0x3a, 0x2026, 0x20, 0x9, 0x3000 }) do CLAUSE_END[cp] = true end

local function cut_by(text, ends)
    local out, buf, i = {}, {}, 1
    while i <= #text do
        local cp, len = codepoint_at(text, i)
        if not cp then break end
        buf[#buf + 1] = text:sub(i, i + len - 1)
        i = i + len
        if ends[cp] then
            local t = table.concat(buf)
            if #chars(clean_punct(t)) > 0 then out[#out + 1] = t end
            buf = {}
        end
    end
    if #buf > 0 then
        local t = table.concat(buf)
        if #chars(clean_punct(t)) > 0 then out[#out + 1] = t end
    end
    return out
end

-- 文案 → 语义单元（句）；换行 / | 也当句界
local function sentences(text)
    return cut_by((tostring(text):gsub("[|\r\n]+", "。")), SENT_END)
end
M.sentences = sentences

-- 长句按子句合并成「一条字幕」那么长（默认 20 字），整子句进出，不切字
local function pack(text, max_chars)
    local cls = cut_by(text, CLAUSE_END)
    if #cls <= 1 then return { text } end
    local out, acc = {}, nil
    for _, c in ipairs(cls) do
        local cn = #chars(clean_punct(c))
        if acc == nil then
            acc = { c, cn }
        elseif acc[2] + cn <= max_chars then
            acc[1] = acc[1] .. c
            acc[2] = acc[2] + cn
        else
            out[#out + 1] = acc[1]
            acc = { c, cn }
        end
    end
    if acc then out[#out + 1] = acc[1] end
    return out
end
M.pack = pack

-- ── 时长估算 ───────────────────────────────────────────────
-- rate: 克隆音色在 speed=1.0 时的字/秒（实测话术音色约 3.5）
local function est_sec(t, rate)
    local n = #chars(clean_punct(t))
    if n == 0 then return 0 end
    local clauses = math.max(1, #cut_by(t, CLAUSE_END))
    return n / rate + (clauses - 1) * 0.15 + 0.08
end
M.est_sec = est_sec

-- ── 按时间轴均匀挑一个子集（Bresenham 式，保首末句）────────
-- durs[i] 单位秒；返回 keep[i]=true/false
local function choose(durs, avail, gap_min, gap_max)
    local n = #durs
    local keep = {}
    for i = 1, n do keep[i] = true end
    local cnt = n
    local function used(k)
        local t, c = 0, 0
        for i = 1, n do if k[i] then t = t + durs[i]; c = c + 1 end end
        return t, c
    end
    local guard = 0
    while cnt > 2 do
        local t = used(keep)
        if t + gap_min * (cnt - 1) <= avail then break end
        guard = guard + 1
        if guard > 64 then break end
        -- 目标保留数：按时间比例算，略偏保守
        local room = math.max(0.5, avail - gap_min * (cnt - 1))
        local m = math.floor(cnt * room / math.max(0.01, t) * 0.97)
        if m >= cnt then m = cnt - 1 end
        if m < 2 then m = 2 end

        local idx = {}
        for i = 1, n do if keep[i] then idx[#idx + 1] = i end end
        local pick, cum, p = {}, 0, 1
        for j = 1, m do pick[j] = (j - 0.5) * t / m end
        local out = {}
        for k, i in ipairs(idx) do
            local b = cum + durs[i]
            cum = b
            local hit = false
            while p <= m and pick[p] < b do hit = true; p = p + 1 end
            if hit or k == 1 or k == #idx then out[#out + 1] = i end
        end
        if #out >= cnt then -- 没减少，强制去掉中间一条最长的
            local worst, wi = -1, nil
            for k = 2, #idx - 1 do
                if durs[idx[k]] > worst then worst, wi = durs[idx[k]], idx[k] end
            end
            if not wi then break end
            out = {}
            for _, i in ipairs(idx) do if i ~= wi then out[#out + 1] = i end end
        end
        for i = 1, n do keep[i] = false end
        cnt = 0
        for _, i in ipairs(out) do keep[i] = true; cnt = cnt + 1 end
    end
    return keep
end
M.choose = choose

-- ── 主入口：文案 + 可用时长 → 分段方案 ─────────────────────
-- 返回 { segs, speed, gap, dropped, slack, est }
function M.plan(text, avail, opt)
    opt = opt or {}
    local rate = opt.rate or 3.5
    local gap_min = opt.gap_min or 0.25
    local gap_max = opt.gap_max or 1.2
    local smin = opt.speed_min or 0.88
    local smax = opt.speed_max or 1.15

    local max_chars = opt.max_chars or 20
    -- 先切句，再把超长句按子句合并成一条字幕的长度
    local sents = {}
    for _, s in ipairs(sentences(text)) do
        if #chars(clean_punct(s)) > max_chars then
            for _, part in ipairs(pack(s, max_chars)) do sents[#sents + 1] = part end
        else
            sents[#sents + 1] = s
        end
    end
    if #sents == 0 then return nil, "文案为空" end
    local durs = {}
    local need = 0
    for i, s in ipairs(sents) do durs[i] = est_sec(s, rate); need = need + durs[i] end

    -- 1) 变速（先不删内容）
    local speech_budget = math.max(0.5, avail - gap_min * (#sents - 1))
    local speed = need / speech_budget
    if speed < smin then speed = smin elseif speed > smax then speed = smax end
    local scaled = {}
    for i = 1, #durs do scaled[i] = durs[i] / speed end

    -- 2) 变速到上限仍超时：
    --    agent 工作流里「文案长度只能接近视频长度」，砍哪句必须由人/agent 决定，
    --    所以默认只报告超了多少，不静默删句；显式 allow_drop 才均匀抽稀。
    local function sum(t, k) local x = 0 for i = 1, #t do if not k or k[i] then x = x + t[i] end end return x end
    local over = sum(scaled) + gap_min * (#sents - 1) - avail
    local keep = {}
    local dropped = 0
    if over > 0.05 and opt.allow_drop then
        keep = choose(scaled, avail, gap_min, gap_max)
        for i = 1, #sents do if not keep[i] then dropped = dropped + 1 end end
    else
        for i = 1, #sents do keep[i] = true end
    end
    local segs, est, total = {}, {}, 0
    for i, s in ipairs(sents) do
        if keep[i] then
            segs[#segs + 1] = s
            est[#est + 1] = scaled[i]
            total = total + scaled[i]
        end
    end
    local n_all = #segs
    local fit_left = avail - (total + gap_min * math.max(0, n_all - 1))
    -- 3) 剩余时间怎么花：先拉句间停顿，再整体后移起播点，仍剩就留给片尾
    local n = #segs
    local gap = gap_min
    if n > 1 then gap = gap_min + math.max(0, math.min(gap_max - gap_min, (avail - total - gap_min * (n - 1)) / n)) end
    local used = total + gap * (n - 1)
    local slack = math.max(0, avail - used)
    local lead = 0
    if slack > 0 then
        lead = math.min(slack * 0.5, 1.5)   -- 起播点后移，最多 1.5s
        slack = slack - lead
    end

    -- overflow：变速到上限仍塞不下（秒）；cut_chars：约等于要砍掉多少字
    local overflow = math.max(0, -fit_left)
    local cut_chars = overflow > 0 and math.ceil(overflow * speed * rate) or 0
    -- 建议字数区间：给 agent 改文案用（rate 按 speed=1.0）
    local lo = math.floor((avail - gap_min * 2) * rate * 0.95)
    local hi = math.floor((avail - gap_min * 2) * rate * smax)
    return {
        segs = segs, est = est, total = total, speed = speed, gap = gap,
        slack = slack, lead = lead, dropped = dropped, rate = rate, avail = avail,
        overflow = overflow, cut_chars = cut_chars, chars = #chars(clean_punct(table.concat(segs))),
        suggest_lo = math.max(1, lo), suggest_hi = math.max(1, hi),
    }
end

-- 生成逐条时间轴（估算用，给出片前预览／写 srt）
function M.timeline(p, lead)
    local out = {}
    local cur = lead or 0
    for i, s in ipairs(p.segs) do
        local d = p.est[i] or 0
        out[i] = { text = s, start = cur, stop = cur + d, dur = d }
        cur = cur + d + p.gap
    end
    return out
end

function M.srt(list)
    local function tc(t)
        local ms = math.floor(t * 1000 + 0.5)
        return string.format("%02d:%02d:%02d,%03d", math.floor(ms / 3600000), math.floor(ms / 60000) % 60,
            math.floor(ms / 1000) % 60, ms % 1000)
    end
    local buf = {}
    for i, s in ipairs(list) do
        buf[#buf + 1] = tostring(i) .. "\n" .. tc(s.start) .. " --> " .. tc(s.stop) .. "\n" .. s.text .. "\n"
    end
    return table.concat(buf, "\n")
end

return M
