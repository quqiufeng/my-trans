-- dub/dub.lua — 视频配音 + 烧字幕（ffmpeg + 克隆音色 / Kokoro / espeak-ng）
-- 自包含实现：只依赖 luajit、ffmpeg/ffprobe、cosyvoice（克隆）或 sherpa-onnx（kokoro）
local M = {}

local HOME = os.getenv("HOME") or "/root"
local HERE = (arg and arg[0] or ""):match("^(.*)/[^/]*$") or "."
local TMP = os.getenv("TMPDIR") or "/tmp"
-- 文案适配引擎（按视频时长整理字幕文案）
local PLAN = dofile(HERE .. "/plan.lua")

local function env(k, d)
    local v = os.getenv(k)
    if v == nil or v == "" then return d end
    return v
end

-- ── 基础工具 ────────────────────────────────────────────────
local function shq(s) return "'" .. tostring(s):gsub("'", "'\\''") .. "'" end

local function read_file(p)
    local f = io.open(p, "rb")
    if not f then return nil end
    local d = f:read("*a")
    f:close()
    return d
end

local function write_file(p, d)
    local f = io.open(p, "wb")
    if not f then return false end
    f:write(d)
    f:close()
    return true
end

local function append_file(p, d)
    local f = io.open(p, "a")
    if not f then return false end
    f:write(d)
    f:close()
    return true
end

local function exists(p)
    if not p or p == "" then return false end
    local f = io.open(p, "rb")
    if f then f:close() return true end
    return false
end

local function size(p)
    local f = io.open(p, "rb")
    if not f then return 0 end
    local n = f:seek("end") or 0
    f:close()
    return n
end

local function mkdirp(p) os.execute("mkdir -p " .. shq(p)) end
local function rm_rf(p) if p and p ~= "" and p ~= "/" then os.execute("rm -rf " .. shq(p)) end end

-- 跑命令（env 前缀 + 合并输出 + 真实退出码）
local function run(argv, opts)
    opts = opts or {}
    local po = os.tmpname()
    local pc = os.tmpname()
    local parts = {}
    if opts.env then
        local ks = {}
        for k in pairs(opts.env) do ks[#ks + 1] = k end
        table.sort(ks)
        for _, k in ipairs(ks) do parts[#parts + 1] = k .. "=" .. shq(opts.env[k]) end
    end
    for _, a in ipairs(argv) do parts[#parts + 1] = shq(a) end
    os.execute(table.concat(parts, " ") .. " >" .. shq(po) .. " 2>&1; printf '%s' $? >" .. shq(pc))
    local rcs = read_file(pc) or "1"
    local code = tonumber((rcs:gsub("%s", ""))) or 1
    local out = read_file(po) or ""
    os.remove(po)
    os.remove(pc)
    if opts.log and opts.log ~= "" then append_file(opts.log, out) end
    return { code = code, out = out }
end

local function ffmpeg(args, log)
    local av = { "ffmpeg", "-hide_banner", "-loglevel", "error", "-y" }
    for _, a in ipairs(args or {}) do av[#av + 1] = a end
    return run(av, { log = log })
end

local function duration(path)
    local r = run({ "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path })
    return tonumber((r.out or ""):match("[%d%.]+")) or 0
end

local function has_audio(path)
    local r = run({ "ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", path })
    return (r.out or ""):find("audio", 1, true) ~= nil
end

-- ── 文本切分 / 字幕排版 ─────────────────────────────────────
local function chars(s)
    local t = {}
    for c in tostring(s):gmatch("[%z\1-\127\194-\244][\128-\191]*") do t[#t + 1] = c end
    return t
end

-- 按 UTF-8 码点解码，避免把多字节标点写进字符集类里（会被当成非法 range）
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

local function timecode(t)
    t = math.max(0, t)
    return string.format("0:%02d:%05.2f", math.floor(t / 60), t % 60)
end

-- 把一串字均分成 n 段（避免末行只剩一个字）
local function split_even(c, n)
    local out = {}
    local base, extra, k = math.floor(#c / n), #c % n, 1
    for i = 1, n do
        local take = base + (i <= extra and 1 or 0)
        if take > 0 then out[#out + 1] = table.concat(c, "", k, k + take - 1) end
        k = k + take
    end
    return out
end

-- 空格在字符流里用 \1 占位（clean_punct 会吃掉空白/标点，断行要靠它）
local SP = "\1"
local function is_space(ch) return ch == SP or ch == " " or ch == "\t" end
local function strip_sp(t) return (tostring(t):gsub(SP, "")) end

-- 均分 n 段，但断点优先落在空格上（±win 字内），不切开词
local function split_near(c, n, win)
    local total = #c
    if n <= 1 or total <= n then
        local t = table.concat(c, "", 1, total):gsub("^%s+", ""):gsub("%s+$", "")
        return { t }
    end
    win = win or 3
    local cuts, k = {}, 1
    for i = 1, n - 1 do
        local ideal = math.floor(total * i / n + 0.5)
        local best
        for d = 0, win do
            for _, p in ipairs({ ideal - d, ideal + d }) do
                if p > 0 and p < total and p >= k and is_space(c[p]) then best = p break end
            end
            if best then break end
        end
        cuts[#cuts + 1] = best or ideal
    end
    local out = {}
    for _, p in ipairs(cuts) do
        if p >= k then
            out[#out + 1] = table.concat(c, "", k, p)
            k = p + 1
        end
    end
    out[#out + 1] = table.concat(c, "", k, total)
    local res = {}
    for _, t in ipairs(out) do
        t = t:gsub("^" .. SP .. "+", ""):gsub(SP .. "+$", "")
        t = t:gsub("^%s+", ""):gsub("%s+$", "")
        if #t > 0 then res[#res + 1] = t end
    end
    if #res == 0 then return { "" } end
    return res
end

local function emit_subs(f, start, dur, text)
    -- 空白先换成 \1 占位，断行时按它切；输出前再抹掉
    local c = chars((tostring(text):gsub("%s+", SP)))
    c = chars(clean_punct(table.concat(c)))
    local vis = strip_sp(table.concat(c))
    if #chars(vis) == 0 then return end
    local cards = math.max(1, math.ceil(#chars(vis) / 24))
    local chunks = split_near(c, cards, 5)
    local dt = dur / #chunks
    for k, part in ipairs(chunks) do
        local pc = chars(part)
        local nvis = #chars(strip_sp(part))
        local lines = math.max(1, math.ceil(nvis / 12))
        local ls = split_near(pc, lines, 5)
        for j, t in ipairs(ls) do ls[j] = strip_sp(t) end
        f:write("Dialogue: 0,", timecode(start + (k - 1) * dt), ",", timecode(start + k * dt),
            ",Default,,0,0,0,,", table.concat(ls, "\\N"), "\n")
    end
end

-- ── TTS ────────────────────────────────────────────────────
local TTS = {
    SHERPA_BIN = env("SHERPA_BIN", "/opt/sherpa-onnx/bin/sherpa-onnx-offline-tts"),
    SHERPA_LIB = env("SHERPA_LIB", "/opt/sherpa-onnx/lib"),
    KOKORO_DIR = env("KOKORO_DIR", "/data/models/kokoro-multi-lang-v1_0"),
    KOKORO_SID = env("KOKORO_SID", "47"),
    TTS_THREADS = tonumber(env("TTS_THREADS", "4")) or 4,
    COSY_LIB = env("COSYVOICE_LIB_DIR", "/opt/cosyvoice.cpp/build/lib"),
    COSY_BIN = env("COSYVOICE_BIN_DIR", "/opt/cosyvoice.cpp/build/bin"),
    COSY_CLI = env("COSYVOICE_CLI", "/opt/cosyvoice.cpp/build/bin/cosyvoice-cli"),
    COSY_TOKENIZER = env("COSYVOICE_TOKENIZER", "/data/models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx"),
    COSY_CAMPPLUS = env("COSYVOICE_CAMPPLUS", "/data/models/Fun-CosyVoice3-0.5B/campplus.onnx"),
    VOICES_DIR = env("VOICES_DIR", HOME .. "/myagent_voices"),
    VOICES_DIR_OLD = HOME .. "/.myagent_voices",
    ASR_BIN = env("SENSEVOICE_BIN", "/opt/SenseVoice.cpp/bin/sense-voice-main"),
    ASR_MODEL = env("SENSEVOICE_MODEL", "/data/models/sense-voice-small-q4_k.gguf"),
    ASR_LANG = env("SENSEVOICE_LANG", "zh"),
}

local function cosy_env()
    local ld = TTS.COSY_LIB .. ":" .. TTS.COSY_BIN .. ":"
    local old = os.getenv("LD_LIBRARY_PATH")
    if old and old ~= "" then ld = ld .. old end
    return { LD_LIBRARY_PATH = ld }
end

local function tts_kokoro(text, out)
    if not exists(TTS.SHERPA_BIN) or not exists(TTS.KOKORO_DIR .. "/model.onnx") then return false end
    local d = TTS.KOKORO_DIR
    local r = run({
        TTS.SHERPA_BIN,
        "--kokoro-model=" .. d .. "/model.onnx",
        "--kokoro-voices=" .. d .. "/voices.bin",
        "--kokoro-tokens=" .. d .. "/tokens.txt",
        "--kokoro-data-dir=" .. d .. "/espeak-ng-data",
        "--kokoro-lexicon=" .. d .. "/lexicon-us-en.txt," .. d .. "/lexicon-zh.txt",
        "--tts-rule-fsts=" .. d .. "/date-zh.fst," .. d .. "/number-zh.fst",
        "--num-threads=" .. tostring(TTS.TTS_THREADS),
        "--sid=" .. TTS.KOKORO_SID,
        "--output-filename=" .. out, text,
    }, { env = { LD_LIBRARY_PATH = TTS.SHERPA_LIB .. ":" .. (os.getenv("LD_LIBRARY_PATH") or "") } })
    return r.code == 0 and size(out) > 0
end

local function tts_espeak(text, out)
    local r = run({ "espeak-ng", "-v", "cmn", "-w", out, text })
    return r.code == 0 and size(out) > 0
end

local function asr(ref)
    local wav = TMP .. "/dub_asr_" .. tostring(os.time()) .. ".wav"
    local c = run({ "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", ref, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav })
    local src = (c.code == 0) and wav or ref
    -- 在 TMP 里跑：sense-voice 会往 cwd 丢 fbank_lfr_cmvn_feature.json 调试文件
    local r = run({ "sh", "-c", "cd " .. shq(TMP) .. " && exec " .. shq(TTS.ASR_BIN) .. " -m " .. shq(TTS.ASR_MODEL)
        .. " -ng -t " .. tostring(TTS.TTS_THREADS) .. " -l " .. shq(TTS.ASR_LANG) .. " " .. shq(src) })
    if src ~= ref then os.remove(wav) end
    local out = (r.out or ""):gsub("<|.-|>", ""):gsub("%[[%d%.]+%-[%d%.]+%]", "")
    out = out:gsub("%s+", " "):gsub("^%s+", ""):gsub("%s+$", "")
    return (out ~= "") and out or nil
end

-- 参考音频 → 克隆音色：只留原始 wav + 参考文本，不再预抽 gguf
-- （cosyvoice_tts.lua 会自己抽特征并缓存到 ~/.cache/cosyvoice-ps）
local function voice_add(name, ref, txt)
    if not exists(ref) then return nil, "参考音频不存在: " .. tostring(ref) end
    if txt and txt ~= "" then -- --ref-text 也可以直接给文本文件
        local t = read_file(txt)
        if t and t:gsub("%s", "") ~= "" then txt = t:gsub("%s*\n%s*", " "):gsub("^%s+", ""):gsub("%s+$", "") end
    end
    if not txt or txt == "" then
        txt = asr(ref)
        if not txt then return nil, "参考音频转写失败，请手动给出参考文本" end
        write_file(TTS.VOICES_DIR .. "/" .. name .. ".ref.txt", txt)
    end
    mkdirp(TTS.VOICES_DIR)
    local wav = TTS.VOICES_DIR .. "/" .. name .. ".wav"
    os.remove(TTS.VOICES_DIR .. "/" .. name .. ".gguf") -- 清掉同名旧 gguf，免得盖过新音频
    if not (ref == wav) then
        local r = run({ "cp", "-f", ref, wav })
        if r.code ~= 0 then return nil, "复制参考音频失败: " .. tostring(r.out) end
    end
    write_file(TTS.VOICES_DIR .. "/" .. name .. ".txt", txt)
    return txt
end

-- 音色定位：绝对路径 > 同名(cwd) > 音色目录/<名>.wav|.gguf > 旧隐藏目录 > $HOME/<名>.wav
local function voice_resolve(name)
    if not name or name == "" then return nil end
    if name:sub(1, 1) == "/" then return exists(name) and name or nil end
    -- 顺序：cwd > 音色目录（wav 优先，.txt 参考文案同名并排）> 旧隐藏目录 > $HOME
    for _, p in ipairs({
        name,
        TTS.VOICES_DIR .. "/" .. name .. ".wav",
        TTS.VOICES_DIR .. "/" .. name .. ".gguf",
        TTS.VOICES_DIR_OLD .. "/" .. name .. ".wav",
        TTS.VOICES_DIR_OLD .. "/" .. name .. ".gguf",
        HOME .. "/" .. name .. ".wav",
        HOME .. "/" .. name .. ".gguf",
    }) do
        if exists(p) then return p end
    end
    return nil
end

local function tts_clone(name, text, out, speed)
    local ps = voice_resolve(name)
    if not ps then return false end
    local extra = nil
    if not ps:lower():match("%.gguf$") then
        -- 原始参考音频：参考文本只认同名 .ref.txt（.txt 可能是配音文案，不能混用）
        local side = ps:gsub("%.[^%.]*$", "") .. ".ref.txt"
        if not exists(side) then
            local t = asr(ps)
            if t then write_file(side, t) end
        end
        local pt = read_file(side)
        if pt then extra = pt:gsub("^%s+", ""):gsub("%s+$", "") end
    end
    local argv = { "luajit", HERE .. "/cosyvoice_tts.lua", ps, out, text, tostring(speed or 1.0) }
    if extra then argv[#argv + 1] = extra end
    local r = run(argv, { env = cosy_env() })
    return r.code == 0 and size(out) > 0
end

local function tts(text, out, voice, clone_speed)
    if voice and voice ~= "" and voice ~= "__kokoro__" then
        if tts_clone(voice, text, out, clone_speed) then return "clone" end
    end
    if tts_kokoro(text, out) then return "kokoro" end
    if tts_espeak(text, out) then return "espeak" end
    return nil
end

-- ── 主流程 ─────────────────────────────────────────────────
function M.run(opts)
    local VIDEO = opts.video
    local TEXT = opts.text
    if not VIDEO or VIDEO == "" or size(VIDEO) == 0 then return nil, "视频文件不存在或为空: " .. tostring(VIDEO) end
    if not TEXT or TEXT == "" then return nil, "文案不能为空" end
    local from_file = read_file(TEXT)
    if from_file then TEXT = from_file:gsub("\n", "|"):gsub("|$", "") end

    local OUT = opts.out or (VIDEO:gsub("%.[^%.]*$", "") .. "_dubbed.mp4")
    local WORK = TMP .. "/vdub_" .. tostring(os.time()) .. "_" .. tostring(math.random(1000, 9999))
    local LOG = WORK .. ".log"
    local ASS = WORK .. "/sub.ass"
    local VOICE = opts.voice or env("VDUB_VOICE", "话术")
    local CLONE_SPEED = opts.clone_speed or 1.0
    local SS = opts.ss or 0
    local ET = opts.trim_end or 0
    local FONT = opts.font or env("VDUB_FONT", "Noto Sans CJK SC Bold")
    local FONT_STYLE = "Default," .. FONT .. ",14,&H00FFCC66,&H00FFCC66,&H00000000,&H00000000,"
        .. "-1,0,0,0,100,100,0,0,1,1,0,2,10,10,24,1"
    local PAUSE, INTRO = 0.3, 0.3

    local function cleanup()
        if os.getenv("VDUB_KEEP") == "1" then print("工作目录保留: " .. WORK) else rm_rf(WORK) end
    end
    -- --ref 用的是临时音色，跑完就清掉，别把 _vdub_tmp.* 留在音色目录
    local function cleanup_ref()
        if VOICE == "_vdub_tmp" then
            os.remove(TTS.VOICES_DIR .. "/_vdub_tmp.wav")
            os.remove(TTS.VOICES_DIR .. "/_vdub_tmp.txt")
        end
    end

    mkdirp(WORK .. "/aud")
    write_file(LOG, "[dub] start " .. os.date("%T") .. "\n")
    local vd = duration(VIDEO) - SS - ET
    if vd < 1 then vd = 1 end

    -- ── 文案适配：按视频播放时间整理字幕文案 ──────────────────
    -- 整句取舍（绝不掐字）→ 轻微变速 → 句间停顿拉伸铺满
    local AVAIL = vd - INTRO - 0.2
    local RATE = tonumber(env("VDUB_RATE", "3.5"))
    local MAX_GAP = tonumber(env("VDUB_MAX_GAP", "1.2"))
    local p, perr = PLAN.plan(TEXT, AVAIL, { rate = opts.rate or RATE, gap_min = PAUSE,
        gap_max = opts.max_gap or MAX_GAP, speed_min = 0.88, speed_max = 1.15,
        allow_drop = opts.allow_drop })
    if not p then
        cleanup()
        return nil, perr
    end
    if p.overflow > 0.05 and not opts.allow_drop and not opts.plan then
        cleanup()
        return nil, string.format(
            "文案超时 %.2fs（约多 %d 字）：视频 %.1fs，可用 %.1fs，建议 %d~%d 字。\n" ..
            "先改文案（这是 agent 的活），或加 --allow-drop 让它自动均匀抽句。\n" ..
            "跑 --plan 可看逐条落点。", p.overflow, p.cut_chars, vd, AVAIL, p.suggest_lo, p.suggest_hi)
    end
    local SEG = p.segs
    CLONE_SPEED = CLONE_SPEED * p.speed

    -- --plan：只出文案（<视频名>_copy.txt + _copy.srt），不合成
    if opts.plan then
        local tl = PLAN.timeline(p, INTRO + (p.lead or 0))
        local base = VIDEO:gsub("%.[^%.]*$", "")
        write_file(base .. "_copy.txt", table.concat(p.segs, "\n") .. "\n")
        write_file(base .. "_copy.srt", PLAN.srt(tl))
        print(string.format("视频 %.2fs ｜ 可用 %.2fs ｜ 文案 %d 字 %d 句", vd, AVAIL, p.chars, #SEG))
        print(string.format("建议字数 %d~%d（rate %.1f 字/秒）", p.suggest_lo, p.suggest_hi, p.rate))
        print(string.format("配音 %.2fs ｜ 语速 x%.2f ｜ 句间 %.2fs ｜ 片尾余 %.2fs",
            p.total, p.speed, p.gap, p.slack))
        if p.overflow > 0.05 then
            print(string.format("⚠ 超时 %.2fs，约需砍 %d 字（或加 --allow-drop 让它自动均匀抽句）",
                p.overflow, p.cut_chars))
        elseif p.slack > 1 then
            print(string.format("⚠ 偏短，片尾会空 %.1fs，建议补 %d~%d 字内容",
                p.slack, math.ceil(p.slack * 0.8 * p.rate), math.ceil(p.slack * p.rate)))
        else
            print("✓ 长度贴合")
        end
        for k, s in ipairs(tl) do
            print(string.format("  %s → %s  %s", timecode(s.start), timecode(s.stop), s.text))
        end
        print("已写出: " .. base .. "_copy.txt / _copy.srt")
        cleanup()
        return { plan = true, video = vd, segments = #SEG }
    end

    if opts.ref and opts.ref ~= "" and opts.ref ~= "none" and size(opts.ref) > 0 then
        local _, err = voice_add("_vdub_tmp", opts.ref, opts.ref_text)
        if err then
            cleanup()
            return nil, err
        end
        VOICE = "_vdub_tmp"
    end

    local SPEED = opts.speed or 0.85
    local function wav_of(i) return string.format("%s/aud/%d.wav", WORK, i) end

    local function tts_all()
        for i, tx in ipairs(SEG) do
            local wav = wav_of(i)
            os.remove(wav)
            local kind = tts(tx, wav, VOICE, CLONE_SPEED)
            if kind and kind ~= "clone" and SPEED ~= 1 and size(wav) > 0 then
                local sp = string.format("%s/aud/sp_%d.wav", WORK, i)
                if ffmpeg({ "-i", wav, "-filter:a", "atempo=" .. SPEED, sp }, LOG).code == 0 then
                    os.remove(wav)
                    os.rename(sp, wav)
                end
            end
        end
    end

    tts_all()

    -- 按实测时长复核：仍超时 → 整句均匀抽稀；有余 → 拉停顿铺满
    local durs, total = {}, 0
    for i = 1, #SEG do
        durs[i] = (size(wav_of(i)) > 0) and duration(wav_of(i)) or 0
        total = total + durs[i]
    end
    if total <= 0 then
        cleanup()
        return nil, "语音合成全部失败，检查 TTS（克隆音色 / kokoro / espeak-ng），日志: " .. LOG
    end

    local keep = PLAN.choose(durs, AVAIL, PAUSE, MAX_GAP)
    local idx, dropped, speech = {}, 0, 0
    for i = 1, #SEG do
        if keep[i] and durs[i] > 0 then
            idx[#idx + 1] = i
            speech = speech + durs[i]
        else
            dropped = dropped + 1
        end
    end
    local GAP = PAUSE
    if #idx > 1 then
        GAP = PAUSE + math.max(0, math.min(MAX_GAP - PAUSE, (AVAIL - speech - PAUSE * (#idx - 1)) / #idx))
    end
    local LEAD0 = INTRO + (p.lead or 0)
    local final_total = LEAD0 + speech + GAP * math.max(0, #idx - 1)
    append_file(LOG, string.format("  · 文案 %d 句 → 保留 %d 句（抽掉 %d），实测配音 %.2fs + 句间 %.2fs×%d = %.2fs / 视频 %.2fs\n",
        #SEG + dropped, #idx, dropped, speech, GAP, math.max(0, #idx - 1), final_total, vd))

    if opts.probe then
        print(string.format("视频 %.2fs ｜ 保留 %d/%d 句 ｜ 配音 %.2fs ｜ 句间 %.2fs ｜ 落点 %.2fs",
            vd, #idx, #SEG, speech, GAP, final_total))
        local cur = LEAD0
        for k, i in ipairs(idx) do
            print(string.format("  段%d(%s→%s, %.2fs): %s", k, timecode(cur), timecode(cur + durs[i]), durs[i], SEG[i]))
            cur = cur + durs[i] + GAP
        end
        cleanup()
        return { probe = true, video = vd, audio = final_total, segments = #idx }
    end

    local af = assert(io.open(ASS, "w"))
    af:write("[Script Info]\nTitle: Video Dub\nScriptType: v4.00+\nWrapStyle: 1\n"
        .. "ScaledBorderAndShadow: yes\nYCbCr Matrix: None\n\n[V4+ Styles]\n"
        .. "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        .. "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        .. "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        .. "Style: " .. FONT_STYLE .. "\n\n[Events]\n"
        .. "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    local cur = LEAD0
    for _, i in ipairs(idx) do
        emit_subs(af, cur, durs[i], SEG[i])
        cur = cur + durs[i] + GAP
    end
    af:close()

    local al = assert(io.open(WORK .. "/alist.txt", "w"))
    for _, i in ipairs(idx) do
        local wav = wav_of(i)
        local pad = string.format("%s/aud/pad_%d.wav", WORK, i)
        ffmpeg({ "-i", wav, "-af", "apad=pad_dur=" .. GAP, pad }, LOG)
        al:write("file '" .. pad .. "'\n")
    end
    al:close()
    ffmpeg({ "-f", "concat", "-safe", "0", "-i", WORK .. "/alist.txt", "-c:a", "aac", WORK .. "/voice.aac" }, LOG)
    ffmpeg({ "-f", "lavfi", "-t", string.format("%.3f", LEAD0), "-i", "anullsrc=r=24000:cl=mono", WORK .. "/sil.m4a" }, LOG)
    local ff = assert(io.open(WORK .. "/final.txt", "w"))
    ff:write("file '" .. WORK .. "/sil.m4a'\nfile '" .. WORK .. "/voice.aac'\n")
    ff:close()
    ffmpeg({ "-f", "concat", "-safe", "0", "-i", WORK .. "/final.txt", "-c:a", "aac", WORK .. "/narration.aac" }, LOG)

    local r
    if opts.keep_audio and has_audio(VIDEO) then
        local fc = "[0:a]volume=0.35[a0];[1:a]volume=1.5[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        r = ffmpeg({ "-ss", tostring(SS), "-t", string.format("%.3f", vd),
            "-i", VIDEO, "-i", WORK .. "/narration.aac",
            "-filter_complex", fc, "-map", "0:v", "-map", "[aout]",
            "-vf", "ass=" .. ASS, "-c:v", "libx264", "-c:a", "aac", OUT }, LOG)
    else
        r = ffmpeg({ "-ss", tostring(SS), "-t", string.format("%.3f", vd),
            "-i", VIDEO, "-i", WORK .. "/narration.aac",
            "-vf", "ass=" .. ASS, "-af", "volume=1.5",
            "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-c:a", "aac", OUT }, LOG)
    end

    if r.code ~= 0 or size(OUT) == 0 then
        local tail = (r.out or ""):sub(-800)
        cleanup()
        return nil, "合成失败，日志: " .. LOG .. "\n" .. tail
    end
    cleanup()
    cleanup_ref()
    return { out = OUT, video = vd, audio = final_total, segments = #idx, dropped = dropped }
end

local function usage()
    print([[
用法: dub/dub.sh <视频> "文案1|文案2" [选项]
      dub/dub.sh <视频> 文案.txt  [选项]
选项:
  --voice 名|路径    克隆音色：名字（$HOME/<名>.wav|.gguf，也认 ~/myagent_voices/）
                   或直接给参考音频/音色文件路径（默认 话术 / $VDUB_VOICE）
                   参考文案 = 同名 .txt，跟 wav 并排（如 ~/话术.wav + ~/话术.txt）
  --ref 参考.wav   临时克隆音色（可用 --ref-text 给参考文本或文本文件）
  --speed 0.85     非克隆引擎的语速（克隆用 --clone-speed）
  --clone-speed    克隆引擎语速
  --plan           体检：量视频时长、报建议字数/超出量，写 <视频>_copy.txt/_copy.srt（不合成）
  --allow-drop     文案超时允许程序自动均匀抽句（默认拒绝，改文案是 agent 的活）
  --probe          试合成一次，打印每条落点/时长，不出片
  --rate 3.5       估算用字速（中文口播 3.0~4.5 字/秒）
  --max-gap 1.2    句间停顿上限（文案偏短时自动拉长到它）
  --ss 秒          从视频第几秒开始
  --trim-end 秒    掐掉结尾多少秒
  --keep-audio     保留原声做背景（音量 0.35）
  --out 文件       输出路径（默认 原名_dubbed.mp4）
  --font 字体名    字幕字体（默认 Noto Sans CJK SC Bold）
]])
end

function M.main(argv)
    local opts = { voice = env("VDUB_VOICE", "话术") }
    local i = 1
    while i <= #argv do
        local a = argv[i]
        if a == "--voice" then opts.voice = argv[i + 1]; i = i + 2
        elseif a == "--ref" then opts.ref = argv[i + 1]; i = i + 2
        elseif a == "--ref-text" then opts.ref_text = argv[i + 1]; i = i + 2
        elseif a == "--speed" then opts.speed = tonumber(argv[i + 1]); i = i + 2
        elseif a == "--clone-speed" then opts.clone_speed = tonumber(argv[i + 1]); i = i + 2
        elseif a == "--ss" then opts.ss = tonumber(argv[i + 1]); i = i + 2
        elseif a == "--trim-end" then opts.trim_end = tonumber(argv[i + 1]); i = i + 2
        elseif a == "--font" then opts.font = argv[i + 1]; i = i + 2
        elseif a == "--plan" then opts.plan = true; i = i + 1
        elseif a == "--rate" then opts.rate = tonumber(argv[i + 1]); i = i + 2
        elseif a == "--max-gap" then opts.max_gap = tonumber(argv[i + 1]); i = i + 2
        elseif a == "--probe" then opts.probe = true; i = i + 1
        elseif a == "--fit" then opts.fit = true; i = i + 1
        elseif a == "--allow-drop" then opts.allow_drop = true; i = i + 1
        elseif a == "--keep-audio" then opts.keep_audio = true; i = i + 1
        elseif a == "--out" then opts.out = argv[i + 1]; i = i + 2
        elseif a == "--help" or a == "-h" then usage(); return 0
        else
            if not opts.video then opts.video = a elseif not opts.text then opts.text = a end
            i = i + 1
        end
    end
    if not opts.video or not opts.text then usage(); return 2 end

    local r, err = M.run(opts)
    if not r then
        io.stderr:write(tostring(err) .. "\n")
        return 1
    end
    if r.probe or r.plan then return 0 end
    print("已生成: " .. r.out)
    return 0
end

if arg and arg[0] and arg[0]:match("dub%.lua$") then
    os.exit(M.main(arg))
end

return M
