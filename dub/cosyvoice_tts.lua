-- lib/voice/cosyvoice_tts.lua — LuaJIT FFI 调 libcosyvoice.so 做零样本克隆合成
-- 用法: luajit cosyvoice_tts.lua <prompt_speech.gguf | 参考音频.wav> <out.wav> <text> [speed] [prompt_text]
--   第一参数给原始参考音频（wav/mp3/flac/m4a/ogg…）也可以：内联抽音色特征，
--   不需要事先转成 gguf；结果按 (文件, mtime, 大小, 参考文本) 缓存在 ~/.cache/cosyvoice-ps。
--   参考文本取 第五个参数 → COSYVOICE_PROMPT_TEXT → 音频同名 .txt。
local ffi = require("ffi")

ffi.cdef[[
typedef struct cosyvoice_context* cosyvoice_context_t;
typedef struct cosyvoice_prompt_speech* cosyvoice_prompt_speech_t;
typedef struct cosyvoice_prompt* cosyvoice_prompt_t;
typedef struct cosyvoice_tts_context* cosyvoice_tts_context_t;
typedef struct cosyvoice_generated_speech { float* data; uint32_t length; } *cosyvoice_generated_speech_ptr;
typedef struct cosyvoice_frontend_context* cosyvoice_frontend_context_t;

void cosyvoice_init_backend_from_path(const char* dir_path);
cosyvoice_context_t cosyvoice_load_from_file(const char* filename);
uint32_t cosyvoice_get_sample_rate(cosyvoice_context_t);
cosyvoice_prompt_speech_t cosyvoice_prompt_speech_load_from_file(const char* filename);
bool cosyvoice_prompt_speech_save_to_file(cosyvoice_prompt_speech_t, const char* filename);
cosyvoice_prompt_t cosyvoice_prompt_init_from_prompt_speech(cosyvoice_context_t, cosyvoice_prompt_speech_t);
cosyvoice_tts_context_t cosyvoice_tts_context_new(cosyvoice_context_t, cosyvoice_prompt_t);
bool cosyvoice_tts_zero_shot(cosyvoice_tts_context_t, const char* text, float speed, cosyvoice_generated_speech_ptr result);
bool cosyvoice_save_wav(const char* filename, const float* data, uint32_t len, uint32_t sr);
bool cosyvoice_set_sampler_seed(cosyvoice_context_t, uint32_t seed);
void cosyvoice_free(cosyvoice_context_t);
void cosyvoice_prompt_speech_free(cosyvoice_prompt_speech_t);
void cosyvoice_prompt_free(cosyvoice_prompt_t);
void cosyvoice_tts_context_free(cosyvoice_tts_context_t);

cosyvoice_frontend_context_t cosyvoice_frontend_load_from_files(const char* speech_tokenizer, const char* campplus);
cosyvoice_prompt_speech_t cosyvoice_frontend_prompt_speech(cosyvoice_frontend_context_t, float* speech, uint32_t speech_len, uint32_t sample_rate, const char* prompt_text);
void cosyvoice_frontend_free(cosyvoice_frontend_context_t);
bool cosyvoice_audio_load_from_file(const char* filename, float** data, uint32_t* length, uint32_t* sample_rate);
void cosyvoice_audio_free(float* data);

int setenv(const char* name, const char* value, int overwrite);
]]

local ps_path, out_path, text = arg[1], arg[2], arg[3]
local speed = tonumber(arg[4] or "1.0") or 1.0
if not (ps_path and out_path and text) then
    io.stderr:write("用法: luajit cosyvoice_tts.lua <prompt_speech.gguf|参考音频.wav> <out.wav> \"文本\" [speed] [prompt_text]\n")
    os.exit(2)
end

local LIB_DIR = os.getenv("COSYVOICE_LIB_DIR") or "/opt/cosyvoice.cpp/build/lib"
local BIN_DIR = os.getenv("COSYVOICE_BIN_DIR") or "/opt/cosyvoice.cpp/build/bin"
local MODEL = os.getenv("COSYVOICE_MODEL") or "/data/models/cosyvoice3-gguf/CosyVoice3-2512_F16.gguf"
local TOKENIZER = os.getenv("COSYVOICE_TOKENIZER") or "/data/models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx"
local CAMPPLUS = os.getenv("COSYVOICE_CAMPPLUS") or "/data/models/Fun-CosyVoice3-0.5B/campplus.onnx"
local HOME = os.getenv("HOME") or "/tmp"
local CACHE = os.getenv("COSYVOICE_PS_CACHE") or (HOME .. "/.cache/cosyvoice-ps")

local ld = os.getenv("LD_LIBRARY_PATH") or ""
if not ld:find(LIB_DIR, 1, true) then
    pcall(function() ffi.C.setenv("LD_LIBRARY_PATH", LIB_DIR .. ":" .. BIN_DIR .. ":" .. ld, 1) end)
end

local ok_load, lib = pcall(ffi.load, "cosyvoice")
if not ok_load then lib = ffi.load(LIB_DIR .. "/libcosyvoice.so") end

-- ── 小工具 ────────────────────────────────────────────────
local function exists(p)
    local f = io.open(p, "rb")
    if f then f:close() return true end
    return false
end

local function read_all(p)
    local f = io.open(p, "rb")
    if not f then return nil end
    local s = f:read("*a")
    f:close()
    return s
end

local function trim(s) return (s:gsub("^%s+", ""):gsub("%s+$", "")) end

local function djb2(s)
    local h = 5381
    for i = 1, #s do h = (h * 33 + s:byte(i)) % 4294967296 end
    return string.format("%08x", h)
end

local function q(s) return '"' .. s .. '"' end

local function isAudio(p)
    local e = p:lower():match("%.([%w]+)$")
    return e ~= nil and e ~= "gguf" and e ~= "bin" and e ~= "safetensors"
end

-- ── 参考音频 → prompt speech（无 gguf 时内联抽特征，带缓存）──
local function prompt_speech_from_audio(audio_path, prompt_text)
    if not prompt_text or prompt_text == "" then
        prompt_text = trim(read_all((audio_path:gsub("%.[^%.]*$", "")) .. ".txt") or "")
    end
    if prompt_text == "" then
        io.stderr:write("[cosyvoice] 参考音频需要参考文本：给第 5 个参数、设 COSYVOICE_PROMPT_TEXT，" ..
            "或用 SenseVoice 转写后写成同名 .txt\n")
        return nil
    end

    local s = io.popen("stat -c %s:%Y -- " .. q(audio_path) .. " 2>/dev/null")
    local meta = trim((s and s:read("*a")) or "")
    if s then s:close() end
    local key = djb2(audio_path .. "|" .. meta .. "|" .. prompt_text)
    os.execute("mkdir -p " .. q(CACHE))
    local cached = CACHE .. "/" .. key .. ".gguf"
    if exists(cached) then return cached end

    if not exists(TOKENIZER) or not exists(CAMPPLUS) then
        io.stderr:write("[cosyvoice] 缺音频前端模型: " .. TOKENIZER .. " / " .. CAMPPLUS .. "\n")
        return nil
    end

    local data, len, sr = ffi.new("float*[1]"), ffi.new("uint32_t[1]"), ffi.new("uint32_t[1]")
    if not lib.cosyvoice_audio_load_from_file(audio_path, data, len, sr) then
        io.stderr:write("[cosyvoice] 读音频失败: " .. audio_path .. "\n")
        return nil
    end
    local fctx = lib.cosyvoice_frontend_load_from_files(TOKENIZER, CAMPPLUS)
    if fctx == nil then
        lib.cosyvoice_audio_free(data[0])
        io.stderr:write("[cosyvoice] 音频前端载入失败\n")
        return nil
    end
    io.stderr:write(string.format("[cosyvoice] 抽音色特征 %s (%.1fs @%dHz)…\n",
        audio_path, tonumber(len[0]) / tonumber(sr[0]), tonumber(sr[0])))
    local psp = lib.cosyvoice_frontend_prompt_speech(fctx, data[0], len[0], sr[0], prompt_text)
    lib.cosyvoice_audio_free(data[0])
    lib.cosyvoice_frontend_free(fctx)
    if psp == nil then
        io.stderr:write("[cosyvoice] 抽音色特征失败\n")
        return nil
    end
    if not lib.cosyvoice_prompt_speech_save_to_file(psp, cached) then
        lib.cosyvoice_prompt_speech_free(psp)
        io.stderr:write("[cosyvoice] 写缓存失败: " .. cached .. "\n")
        return nil
    end
    io.stderr:write("[cosyvoice] 音色特征已缓存 " .. cached .. "\n")
    return cached
end

local use_ps = ps_path
if isAudio(ps_path) then
    local prompt_text = trim(arg[5] or os.getenv("COSYVOICE_PROMPT_TEXT") or "")
    local c, err = prompt_speech_from_audio(ps_path, prompt_text)
    if not c then os.exit(1) end
    use_ps = c
elseif not exists(ps_path) then
    io.stderr:write("[cosyvoice] 音色文件不存在: " .. ps_path .. "\n")
    os.exit(1)
end

-- ── 合成 ──────────────────────────────────────────────────
lib.cosyvoice_init_backend_from_path(BIN_DIR)
local ctx = lib.cosyvoice_load_from_file(MODEL)
if ctx == nil then io.stderr:write("[cosyvoice] 载入模型失败: " .. MODEL .. "\n"); os.exit(1) end

local seed = tonumber(os.getenv("COSYVOICE_SEED") or "20261008") or 0
if seed > 0 then pcall(lib.cosyvoice_set_sampler_seed, ctx, seed) end

local psp = lib.cosyvoice_prompt_speech_load_from_file(use_ps)
if psp == nil then
    io.stderr:write("[cosyvoice] 载入音色失败: " .. use_ps .. "\n")
    lib.cosyvoice_free(ctx)
    os.exit(1)
end

local prompt = lib.cosyvoice_prompt_init_from_prompt_speech(ctx, psp)
local ttsctx = lib.cosyvoice_tts_context_new(ctx, prompt)
local res = ffi.new("struct cosyvoice_generated_speech[1]")
local ok = lib.cosyvoice_tts_zero_shot(ttsctx, text, speed, res)
if not ok then
    io.stderr:write("[cosyvoice] 合成失败\n")
else
    local sr = lib.cosyvoice_get_sample_rate(ctx)
    lib.cosyvoice_save_wav(out_path, res[0].data, res[0].length, sr)
    print(string.format("[cosyvoice] 已生成 %s (%.2fs @%dHz)", out_path, tonumber(res[0].length) / sr, sr))
end

lib.cosyvoice_tts_context_free(ttsctx)
lib.cosyvoice_prompt_free(prompt)
lib.cosyvoice_prompt_speech_free(psp)
lib.cosyvoice_free(ctx)
os.exit(ok and 0 or 1)
