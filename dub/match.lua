-- dub/match.lua — 文案行 → 阶段窗口 → EDL（画面剪辑表）
--
--   lua dub/match.lua --text 文案.txt --shot shot/IMG_0195.json --out edl.json
--
-- 思路：
--   每句文案先按关键词定一个「意图 tag」，再从阶段表里挑同 tag 的窗口；
--   多份文案轮流生成时，用全局使用计数（edl 目录下 .usage.json）优先挑没被用过的窗口，
--   这样一条素材出 N 条片子也不会撞脸。
--   本文件只产出 EDL（in/out/字幕），不碰 ffmpeg；执行交给 render.sh。
local M = {}

local RATE = 3.6        -- 中文口播字/秒（估算字幕停留时长）
local MIN_DUR = 1.3
local MAX_DUR = 5.0

-- ── 小工具 ────────────────────────────────────────────────
local function chars(s)
    local n = 0
    for _ in tostring(s):gmatch("[%z\1-\127\194-\244][\128-\191]*") do n = n + 1 end
    return n
end

local function readfile(p)
    local f = assert(io.open(p, "r"), "打不开: " .. p)
    local s = f:read("*a"); f:close()
    return s
end

local function trim(s) return (s:gsub("^%s+", ""):gsub("%s+$", "")) end

local function json_esc(s)
    return (s:gsub('[%c"\\]', function(c)
        local m = { ['"'] = '\\"', ['\\'] = '\\\\', ['\n'] = '\\n', ['\t'] = '\\t', ['\r'] = '\\r' }
        return m[c] or string.format("\\u%04x", c:byte())
    end))
end

-- ── 意图识别：文案 → tag ───────────────────────────────────
-- 顺序敏感：先匹配到的赢（品牌词放最前，避免「音响」吞掉别的）
local INTENT = {
    { "wireless",{ "无线充" } },
    { "port",   { "充电", "续航", "快充", "Ｃ口", "接口" } },
    { "takeout",{ "单独佩戴", "单独", "单只", "一只", "取出", "左右耳" } },
    { "both",   { "双主机", "双设备", "两台", "同时连接", "双耳" } },
    { "side",   { "防水", "格栅", "防汗", "ｉｐ", "IP" } },
    { "brand",  { "十大", "品牌", "JBL", "jbl", "哈曼", "旗下", "音响", "家庭影院", "天花板", "低频", "重低音", "调教" } },
    { "final",  { "直播间", "想要的", "下单", "主页", "关注", "小黄车" } },
    { "guide",  { "降噪", "小巧", "舒服", "佩戴", "好看", "颜色" } },
}
M.INTENT = INTENT

-- tag 缺失时的兜底顺序
local FALLBACK = {
    wireless={ "closed", "brand", "side" },
    port   = { "port", "side", "closed", "both" },
    takeout= { "takeout", "both", "hero" },
    both   = { "both", "hero", "final" },
    side   = { "side", "closed", "port" },
    brand  = { "brand", "hero", "final" },
    final  = { "final", "hero", "both" },
    guide  = { "guide", "both", "hero" },
}

function M.intent(line)
    local low = line:lower()
    for _, rule in ipairs(INTENT) do
        for _, kw in ipairs(rule[2]) do
            if low:find(kw:lower(), 1, true) then return rule[1] end
        end
    end
    return "guide"
end

-- ── 阶段表加载（手写 mini json 解析，只认本项目模板） ────────
local function parse_shot(s)
    local ph = {}
    for blk in s:gmatch("{([^{}]-)}") do
        local t = {}
        -- Lua 模式没有 | 交替，引号串和裸数字分两遍取
        for k, v in blk:gmatch('"([%w_]+)"%s*:%s*"([^"]*)"') do t[k] = v end
        for k, v in blk:gmatch('"([%w_]+)"%s*:%s*([^"%s][^,}]*)') do
            if t[k] == nil then t[k] = (v:gsub("%s+$", "")) end
        end
        if t["in"] and tonumber(t["in"]) then
            ph[#ph + 1] = {
                into = tonumber(t["in"]), out = tonumber(t.out),
                tag = t.tag or "guide", subject = t.subject or "",
                quality = tonumber(t.quality) or 3,
            }
        end
    end
    table.sort(ph, function(a, b) return a.into < b.into end)
    M.meta = {
        src = s:match('"src"%s*:%s*"([^"]+)"'),
        duration = tonumber(s:match('"duration"%s*:%s*([%d%.]+)')) or (ph[#ph] and ph[#ph].out) or 0,
        continuous = s:match('"continuous"%s*:%s*true') ~= nil,
    }
    return ph
end
M.parse_shot = parse_shot

-- ── 使用计数（多份文案轮产出时避免重复取同一段） ────────────
local function load_usage(p)
    local u = {}
    local ok, f = pcall(io.open, p, "r")
    if ok and f then
        for k, v in f:read("*a"):gmatch('"([^"]+)"%s*:%s*(%d+)') do u[k] = tonumber(v) end
        f:close()
    end
    return u
end

local function save_usage(p, u)
    local keys = {}
    for k in pairs(u) do keys[#keys + 1] = k end
    table.sort(keys)
    local buf = { "{" }
    for i, k in ipairs(keys) do
        buf[#buf + 1] = string.format('%s"%s":%d', i > 1 and "," or "", json_esc(k), u[k])
    end
    buf[#buf + 1] = "}\n"
    local f = assert(io.open(p, "w")); f:write(table.concat(buf)); f:close()
end

local function wid(p) return string.format("%.1f-%.1f", p.into, p.out) end

-- ── 选窗：带游标的切段 ────────────────────────────────────
-- 每个窗口记一个已用长度 cursor；一句只从窗口里往后切一段，
-- 同一窗口被多句分走是连续的（不会重复播同一段画面）。
-- 单镜头素材没有真切点，窗口尾巴可以外借，免得 2s 的小格永远凑不出 3s 字幕
local EXTEND = 1.8
local function out_of(p, dur_total)
    if not (M.meta and M.meta.continuous) then return p.out end
    local e = p.out + EXTEND
    if dur_total and dur_total > 0 then e = math.min(e, dur_total) end
    return e
end
M.out_of = out_of

local function pick(pool, tag, need, usage, cursor, dur_total)
    local chain = FALLBACK[tag] or { tag, "both", "hero" }
    for _, want in ipairs(chain) do
        local cand = {}
        for _, p in ipairs(pool) do
            local room = (out_of(p, dur_total) - p.into) - (cursor[wid(p)] or 0)
            if p.tag == want and room >= need then cand[#cand + 1] = p end
        end
        if #cand == 0 then
            for _, p in ipairs(pool) do
                local room = (out_of(p, dur_total) - p.into) - (cursor[wid(p)] or 0)
                if p.tag == want and room >= need * 0.78 then cand[#cand + 1] = p end
            end
        end
        if #cand > 0 then
            table.sort(cand, function(a, b)
                local ua, ub = usage[wid(a)] or 0, usage[wid(b)] or 0
                if ua ~= ub then return ua < ub end
                if a.quality ~= b.quality then return a.quality > b.quality end
                return ((out_of(a, dur_total) - a.into) - (cursor[wid(a)] or 0)) >
                       ((out_of(b, dur_total) - b.into) - (cursor[wid(b)] or 0))
            end)
            return cand[1], want
        end
    end
    -- 兜底：链上都没空间了，就从还有空位的窗口里挑（按 未用→质量→余量）
    local any = {}
    for _, p in ipairs(pool) do
        local room = (out_of(p, dur_total) - p.into) - (cursor[wid(p)] or 0)
        if room >= need then any[#any + 1] = p end
    end
    if #any > 0 then
        table.sort(any, function(a, b)
            local ua, ub = usage[wid(a)] or 0, usage[wid(b)] or 0
            if ua ~= ub then return ua < ub end
            local ra = (out_of(a, dur_total) - a.into) - (cursor[wid(a)] or 0)
            local rb = (out_of(b, dur_total) - b.into) - (cursor[wid(b)] or 0)
            if math.abs(ra - rb) > 0.05 then return ra < rb end   -- best-fit：少留碎片
            return a.quality > b.quality
        end)
        return any[1], "any"
    end
    return nil
end

-- ── 主流程 ────────────────────────────────────────────────
function M.build(text, shot, opt)
    opt = opt or {}
    local usage = load_usage(opt.usage or "/tmp/dub-usage.json")
    local cursor = {}
    local lines = {}
    for l in (text .. "\n"):gmatch("([^\n]*)\n") do
        l = trim(l)
        if l ~= "" and l:sub(1, 1) ~= "#" then lines[#lines + 1] = l end
    end
    if #lines == 0 then return { src = opt.src, items = {}, total = 0, miss = 0 } end

    -- 1) 每句按字数估时长
    local needs, tags = {}, {}
    local sum = 0
    for i, line in ipairs(lines) do
        tags[i] = M.intent(line)
        needs[i] = math.max(MIN_DUR, math.min(MAX_DUR, chars(line) / RATE))
        sum = sum + needs[i]
    end
    -- 2) 要铺满目标时长就整体放慢/加快（幅度受限，别让字幕闪得太快）
    local scale = 1.0
    if opt.target and opt.target > 0 then
        scale = math.max(0.72, math.min(1.4, opt.target / sum))
        for i = 1, #needs do needs[i] = math.min(MAX_DUR * 1.4, needs[i] * scale) end
    end

    -- 3) 逐句切窗
    local items, miss = {}, 0
    for i, line in ipairs(lines) do
        local p, want = pick(shot, tags[i], needs[i], usage, cursor, M.meta and M.meta.duration)
        if not p then miss = #lines - i + 1; break end
        local w_ = wid(p)
        local used = cursor[w_] or 0
        local room = (out_of(p, M.meta and M.meta.duration) - p.into) - used
        local dur = math.min(needs[i], room)
        items[#items + 1] = {
            idx = i, text = line, intent = tags[i], matched = want,
            src = opt.src, into = p.into + used, out = p.into + used + dur,
            dur = dur, speed = needs[i] > room and (needs[i] / room) or 1.0,
            subject = p.subject, window = w_,
        }
        cursor[w_] = used + dur
        usage[w_] = (usage[w_] or 0) + 1
    end
    if opt.usage then save_usage(opt.usage, usage) end
    local total = 0
    for _, it in ipairs(items) do total = total + it.dur end
    return { src = opt.src, items = items, total = total, miss = miss, scale = scale }
end

-- ── 输出 EDL json / ASS 预览 / 文本清单 ────────────────────
function M.to_json(e)
    local b = { '{\n  "src": "' .. json_esc(e.src) .. '",\n  "total": ' .. string.format("%.2f", e.total) .. ',\n  "items": [' }
    for i, it in ipairs(e.items) do
        b[#b + 1] = string.format(
            '\n    {"idx":%d,"in":%.2f,"out":%.2f,"dur":%.2f,"speed":%.3f,"tag":"%s","matched":"%s","window":"%s","subject":"%s","text":"%s"}%s',
            it.idx, it.into, it.out, it.dur, it.speed, it.intent, it.matched,
            it.window, json_esc(it.subject), json_esc(it.text), i < #e.items and "," or "")
    end
    b[#b + 1] = "\n  ]\n}\n"
    return table.concat(b)
end

function M.to_list(e)
    local b = {}
    for _, it in ipairs(e.items) do
        b[#b + 1] = string.format("%2d  %5.2f-%5.2f (%4.2fs x%.2f) [%-7s→%-7s] %s\n    画面: %s",
            it.idx, it.into, it.out, it.dur, it.speed, it.intent, it.matched, it.text, it.subject)
    end
    return table.concat(b, "\n")
end


-- ── CLI ───────────────────────────────────────────────────
if arg and arg[0] and arg[0]:match("match%.lua$") then
    local o = { src = nil, text = nil, shot = nil, out = nil, usage = nil }
    local i = 1
    while i <= #arg do
        local a, v = arg[i], arg[i + 1]
        if a == "--text" then o.text = v; i = i + 2
        elseif a == "--shot" then o.shot = v; i = i + 2
        elseif a == "--out" then o.out = v; i = i + 2
        elseif a == "--usage" then o.usage = v; i = i + 2
        elseif a == "--no-usage" then o.usage = "/dev/null"; i = i + 1
        elseif a == "-h" or a == "--help" then
            print("用法: lua dub/match.lua --text 文案.txt --shot shot/X.json [--out edl.json] [--no-usage]")
            os.exit(0)
        else i = i + 1 end
    end
    if not o.text or not o.shot then
        io.stderr:write("缺 --text / --shot\n"); os.exit(1)
    end
    local shotraw = readfile(o.shot)
    local src = shotraw:match('"src"%s*:%s*"([^"]+)"')
    local ph = parse_shot(shotraw)
    if #ph == 0 then io.stderr:write("阶段表为空: " .. o.shot .. "\n"); os.exit(1) end
    o.src = src
    local e = M.build(readfile(o.text), ph, o)
    if e.miss > 0 then
        io.stderr:write(string.format("警告: 第 %d 句起没有可用窗口（阶段表不够长）\n", #e.items + 1))
    end
    print(("素材 %s  阶段 %d  文案 %d 句  成片 %.2fs"):format(src or "?", #ph, #e.items, e.total))
    print(M.to_list(e))
    if o.out then
        local f = assert(io.open(o.out, "w")); f:write(M.to_json(e)); f:close()
        print("\nEDL -> " .. o.out)
    end
    os.exit(0)
end

return M
