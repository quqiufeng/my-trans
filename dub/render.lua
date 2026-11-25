-- dub/render.lua — EDL → ASS 字幕 + ffmpeg filter_complex_script + 命令行
--
--   luajit dub/render.lua edl.json --ass /tmp/s.ass --fc /tmp/fc.txt --cmd
--
-- 只做生成，不跑 ffmpeg；由 render.sh 执行，排错好看。
local M = {}
local ok_cjson, cjson = pcall(require, "cjson")

-- 默认竖屏 1080x1920（手机口播素材多为 portrait，ffprobe 报 1920x1080 是带 rotation 的
-- 存储尺寸，作者要的是竖屏成片）。可用 --w/--h 覆盖。
local W, H = 1080, 1920
local FPS = "30000/1001"
local FONT = "Noto Sans CJK SC"

local function readfile(p)
    local f = assert(io.open(p, "r"), "打不开: " .. p)
    local s = f:read("*a"); f:close(); return s
end

-- 只认本项目 EDL 模板：一行一个 item
local function parse_edl(s)
    local src = s:match('"src"%s*:%s*"([^"]+)"')
    local items = {}
    for line in s:gmatch('({[^{}]-})') do
        local it = {}
        for k, v in line:gmatch('"(%w+)"%s*:%s*"([^"]*)"') do it[k] = v end
        for k, v in line:gmatch('"(%w+)"%s*:%s*([%d%.%-]+)') do it[k] = tonumber(v) end
        if it["in"] and it.out then items[#items + 1] = it end
    end
    table.sort(items, function(a, b) return (a.idx or 0) < (b.idx or 0) end)
    return src, items
end

-- ── 文案工具 ────────────────────────────────────────────
local function chars(s)
    local n = 0
    for _ in tostring(s):gmatch("[%z\1-\127\194-\244][\128-\191]*") do n = n + 1 end
    return n
end

local PUNCT = {}
for _, c in ipairs({ "，", "。", "、", "；", "：", "！", "？", "…", "—", "」", "）",
    ",", ".", "!", "?", ";", ":", " ", ")" }) do PUNCT[c] = true end

-- 字幕断行：先按标点/空格切短语，再贪心装箱成每行 ≤ n 字
-- （不再从词中间劈开：JBL / DJ / IPX5 这类英文词整块保留）
local function phrases(text, n)
    local segs, cur = {}, {}
    local function push() if #cur > 0 then segs[#segs + 1] = table.concat(cur); cur = {} end end
    for ch in tostring(text):gmatch("[%z\1-\127\194-\244][\128-\191]*") do
        cur[#cur + 1] = ch
        if PUNCT[ch] then push()
        elseif #cur >= n + 6 and not ch:match("%w") then push() end
    end
    push()
    return segs   -- 长短语整块保留：交给 ass() 按行宽缩字号，不从词中间劈
end

function M.wrap(text, n, maxLines)
    maxLines = maxLines or 3
    local lines, cur = {}, ""
    for _, sg in ipairs(phrases(text, n)) do
        if cur == "" then
            cur = sg
        elseif chars(cur) + chars(sg) <= n then
            cur = cur .. sg
        else
            lines[#lines + 1] = cur; cur = sg
        end
    end
    if cur ~= "" then lines[#lines + 1] = cur end
    if #lines == 0 then lines[1] = tostring(text) end
    while #lines > maxLines do                    -- 超行数：并进末行，别糊满屏
        local last = table.remove(lines)
        lines[#lines] = lines[#lines] .. last
    end
    local mx = 0
    for _, l in ipairs(lines) do mx = math.max(mx, chars(l)) end
    return table.concat(lines, "\\N"), #lines, mx
end

local function tc(t)
    local cs = math.floor(t * 100 + 0.5)
    local h = math.floor(cs / 360000); cs = cs - h * 360000
    local m = math.floor(cs / 6000);   cs = cs - m * 6000
    local sec = math.floor(cs / 100);  cs = cs - sec * 100
    return string.format("%d:%02d:%02d.%02d", h, m, sec, cs)
end

-- ASS：横屏底部居中，粗体+黑描边，屏幕对齐 2（底部居中）
function M.ass(items, w, h, fsize, wrap)
    w, h = w or W, h or H
    local fs = fsize or math.floor(w * 0.062)          -- 竖屏 1080 → 67px
    local mv = math.floor(h * 0.075)                   -- 底部留白 144px
    local per = math.max(6, math.floor((w - 200) / fs)) -- 一行能放几个字
    local b = { "\239\187\191" .. "[Script Info]",
        "ScriptType: v4.00+", "PlayResX: " .. w, "PlayResY: " .. h,
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, " ..
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, " ..
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        string.format("Style: D,%s,%d,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,1,0,1,4,2,2,100,100,%d,1",
            FONT, fs, mv),
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text" }
    local t = 0
    for _, it in ipairs(items) do
        local txt = it.text
        local scale = 1.0
        if wrap ~= false and chars(txt) > per then
            local _, _, mx = M.wrap(txt, per)
            txt = (M.wrap(txt, per))
            if mx and mx > per then scale = per / mx end
            if mx and mx > per * 1.6 then
                io.stderr:write(string.format("⚠ 单行 %d 字，字号已缩到 %d%%，建议拆句：%s\n", mx, math.floor(scale * 100), it.text))
            end
        end
        if scale < 1.0 then
            txt = string.format("{\\fs%d}%s", math.floor(fs * scale), txt)
        end
        b[#b + 1] = string.format("Dialogue: 0,%s,%s,D,,0,0,0,,%s", tc(t), tc(t + it.dur), txt)
        t = t + it.dur
    end
    return table.concat(b, "\n") .. "\n", t
end


-- ── 文案驱动排期（话术加长也不会没画面）────────────────────
-- 一句文案 → 时长 = 字数/rate，夹在 minDur..maxDur；画面从窗口池取，
-- 同 tag 优先、用过的不优先；窗口比这句短就循环该窗口，池子用完整池轮转。
function M.fit(items, windows, o)
    o = o or {}
    local rate = o.rate or 3.6
    local minDur = o.minDur or 1.0
    local maxDur = o.maxDur or 6.0
    local pool = {}
    for _, w in ipairs(windows or {}) do
        local a, b = tonumber(w["in"]), tonumber(w.out)
        if a and b and b > a then
            pool[#pool + 1] = { ["in"] = a, out = b, tag = w.tag, subj = w.subject }
        end
    end
    assert(#pool > 0, "窗口池为空：--shot 的 phases 里没有 in/out")
    local used, cursor, out, times = {}, 1, {}, {}

    local function pick(tag)
        local cand = {}
        for j = 1, #pool do if not used[j] then cand[#cand + 1] = j end end
        if #cand == 0 then
            used = {}
            for j = 1, #pool do cand[#cand + 1] = j end
        end
        local c
        for _, j in ipairs(cand) do
            if tag and pool[j].tag == tag then c = j break end
        end
        if not c then
            for k = 0, #pool - 1 do
                local j = (cursor - 1 + k) % #pool + 1
                if not used[j] then c = j break end
            end
        end
        c = c or cand[1]
        used[c] = (used[c] or 0) + 1
        times[c] = (times[c] or 0) + 1
        cursor = c % #pool + 1
        return c
    end

    for _, it in ipairs(items) do
        local n = chars(it.text or "")
        local dur = math.max(minDur, math.min(maxDur, n / rate))
        local j = pick(it.tag)
        local w = pool[j]
        local wlen = w.out - w["in"]
        local rep = math.max(1, math.ceil(dur / wlen - 1e-6))
        local each = dur / rep
        local segs, span = {}, math.max(0.001, wlen - each)
        local seen = (times[j] or 1) - 1                        -- 这个窗口第几次被用
        for r = 1, rep do
            -- 起点漂移：同一个窗口被反复用也换帧，别让观众看见同一格
            local off = (seen * 0.61 * wlen + (r - 1) * each * 0.37) % span
            segs[#segs + 1] = { ["in"] = w["in"] + off, out = w["in"] + off + each }
        end
        local ni = {}
        for k, v in pairs(it) do ni[k] = v end
        ni.dur, ni.segs, ni.want = dur, segs, n / rate
        ni.window = string.format("%.1f-%.1f", w["in"], w.out)
        ni.rep = rep
        out[#out + 1] = ni
    end
    return out
end

-- 展平成「一段一行」，供 plan/fc/ffmpeg 用
function M.flatten(items)
    local segs, t = {}, 0
    for _, it in ipairs(items) do
        local s0 = t
        local ss = it.segs or { { ["in"] = it["in"], out = it.out } }
        for k, sg in ipairs(ss) do
            segs[#segs + 1] = { ["in"] = sg["in"], out = sg.out,
                s0 = (k == 1) and s0 or nil, s1 = (k == #ss) and (t + it.dur) or nil }
        end
        t = t + it.dur
    end
    return segs, t
end

-- filter_complex_script：每段 trim → setpts → 统一 scale/fps → concat → ass
function M.fc(items, assPath, keepAudio, w, h)
    local p, chains, pads, apads = {}, {}, {}, {}
    items = M.flatten(items)
    for i, it in ipairs(items) do
        p[#p + 1] = string.format(
            "[0:v]trim=start=%.3f:end=%.3f,setpts=PTS-STARTPTS,fps=%d,scale=%d:%d,setsar=1[v%d]",
            it["in"], it.out, FPS, w or W, h or H, i)
        pads[#pads + 1] = string.format("[v%d]", i)
        if keepAudio then
            p[#p + 1] = string.format("[0:a]atrim=start=%.3f:end=%.3f,asetpts=PTS-STARTPTS,aresample=48000[a%d]",
                it["in"], it.out, i)
            apads[#apads + 1] = string.format("[a%d]", i)
        end
    end
    p[#p + 1] = table.concat(pads) .. string.format("concat=n=%d:v=1:a=0[vcat]", #items)
    if keepAudio then
        p[#p + 1] = table.concat(apads) .. string.format("concat=n=%d:v=0:a=1[acat]", #items)
    end
    p[#p + 1] = string.format("[vcat]ass=%s[vout]", assPath)
    return table.concat(p, ";\n") .. "\n"
end

-- 切段计划（tab 分隔）：in / out / 段文件 / 字幕起 / 字幕止
function M.plan(items, segdir, keepAudio)
    local segs = M.flatten(items)
    local b = {}
    for i, sg in ipairs(segs) do
        b[#b + 1] = string.format("%.3f\t%.3f\t%s/seg_%03d.mp4\t%.3f\t%.3f",
            sg["in"], sg.out, segdir, i, sg.s0 or 0, sg.s1 or 0)
    end
    return table.concat(b, "\n") .. "\n"
end

-- 输出 shell 命令：一条命令完成切段+拼接+烧字幕
function M.cmd(src, fcPath, out, keepAudio, crf, preset)
    local a = keepAudio and '-map "[acat]" -c:a aac -b:a 160k' or "-an"
    return string.format(
        'ffmpeg -y -v error -stats -i %q -filter_complex_script %q -map "[vout]" %s ' ..
        "-c:v libx264 -preset %s -crf %d -pix_fmt yuv420p -movflags +faststart %q",
        src, fcPath, a, preset or "medium", crf or 20, out)
end

if arg and arg[0] and arg[0]:match("render%.lua$") then
    local o = { w = nil, h = nil, ass = nil, fc = nil, plan = nil, segdir = nil, cmd = false, keep = false,
        crf = 20, preset = "medium", out = nil, fit = false, shot = nil, rate = 3.6, minDur = 1.0, maxDur = 6.0 }
    local edlPath, i = nil, 1
    while i <= #arg do
        local a, v = arg[i], arg[i + 1]
        if a == "--ass" then o.ass = v; i = i + 2
        elseif a == "--fc" then o.fc = v; i = i + 2
        elseif a == "--plan" then o.plan = v; i = i + 2
        elseif a == "--segdir" then o.segdir = v; i = i + 2
        elseif a == "--keep-audio" then o.keep = true; i = i + 1
        elseif a == "--w" then o.w = tonumber(v); i = i + 2
        elseif a == "--h" then o.h = tonumber(v); i = i + 2
        elseif a == "--crf" then o.crf = tonumber(v); i = i + 2
        elseif a == "--fit" then o.fit = true; i = i + 1
        elseif a == "--shot" then o.shot = v; i = i + 2
        elseif a == "--rate" then o.rate = tonumber(v); i = i + 2
        elseif a == "--min-dur" then o.minDur = tonumber(v); i = i + 2
        elseif a == "--max-dur" then o.maxDur = tonumber(v); i = i + 2
        elseif a == "--preset" then o.preset = v; i = i + 2
        elseif a == "--cmd" then o.cmd = true; i = i + 1
        elseif a == "--out" then o.out = v; i = i + 2
        elseif a:sub(1, 1) ~= "-" and not edlPath then edlPath = a; i = i + 1
        else i = i + 1 end
    end
    assert(edlPath, "用法: luajit dub/render.lua edl.json --ass a.ass --fc fc.txt --cmd 输出.mp4 [--keep-audio]")
    local src, items = parse_edl(readfile(edlPath))
    assert(#items > 0, "EDL 里没有 item")
    if o.fit then
        assert(o.shot, "--fit 需要 --shot 窗口表（match.lua 用同一个）")
        assert(ok_cjson, "--fit 需要 cjson")
        local f = assert(io.open(o.shot, "r")); local sj = cjson.decode(f:read("*a")); f:close()
        local before = #items
        items = M.fit(items, sj.phases or {}, o)
        local segs = M.flatten(items)
        io.stderr:write(string.format("文案驱动：%d 句 → %d 个画面段（rate %.1f 字/秒）\n", before, #segs, o.rate))
    end
    local ass, total = M.ass(items, o.w, o.h)
    if o.fit then
        local segs = M.flatten(items)
        local over = {}
        for _, it in ipairs(items) do
            if (it.want or 0) > it.dur + 0.01 then
                over[#over + 1] = string.format("%s  (.%.2fs 需 %.2fs)", it.text, it.dur, it.want)
            end
        end
        if #over > 0 then
            io.stderr:write(string.format("⚠ %d 句超过 max-dur %.1fs，建议拆短句：\n", #over, o.maxDur))
            for _, l in ipairs(over) do io.stderr:write("   · " .. l .. "\n") end
        end
    end
    if o.ass then local f = assert(io.open(o.ass, "wb")); f:write(ass); f:close() end
    if o.plan then
        local f = assert(io.open(o.plan, "w")); f:write(M.plan(items, o.segdir or ".", o.keep)); f:close()
    end
    if o.fc then
        local f = assert(io.open(o.fc, "w")); f:write(M.fc(items, o.ass or "subs.ass", o.keep, o.w, o.h)); f:close()
    end
    io.stderr:write(string.format("EDL %d 段  成片 %.2fs  %dx%d\n", #items, total, o.w or W, o.h or H))
    if o.cmd then
        assert(o.fc, "--cmd 需要配合 --fc")
        assert(o.out, "--cmd 需要 --out 输出路径")
        print(M.cmd(src, o.fc, o.out, o.keep, o.crf, o.preset))
    end
    os.exit(0)
end

return M
