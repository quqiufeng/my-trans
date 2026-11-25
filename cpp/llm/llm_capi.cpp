// llm_capi.cpp — Hunyuan-MT (llama.cpp 服务) 的 C ABI 封装
//
// 自写最小 HTTP/1.1 POST 客户端（POSIX socket），无 libcurl 依赖；
// JSON 用 nlohmann/json 构造与解析。
//
// 翻译策略（借鉴 BaoCut 字幕管线）：
//   1) 整句翻译：把连续字幕条按「句末标点 / 停顿 / 说话人切换」聚合成完整句子，
//      整句送模型，避免逐条碎片翻译导致的断句错误与上下文缺失。
//   2) 确定性回切：按各条原文字数比例，把整句译文用「目标语切分器」切回同样的
//      条数，时间轴不变；只在合法缝（不在词内、不悬标点）切。
//   3) 顺句驱动：可选（SUB_ORDER_HINT=1）提示模型尽量贴近源语小句顺序，
//      使每条译文与当时说的话对得上。
//
// 整句聚合默认按源语文字自动选择：拉丁字母等「以空格分词」的语言（英/法/德…）
// Whisper 常把一句切成多条，聚合收益大；中日韩等「不分词」语言的 Whisper 分段
// 往往已是短语级，强行聚合再回切会拆散专名、错位时间轴，因此默认逐条翻译。
//
// 可用环境变量（都有默认值）：
//   SUB_GROUP      -1|0|1   整句翻译：-1 自动（默认），0 关，1 强开
//   SUB_PAUSE      秒        句间停顿阈值，默认 0.7
//   SUB_MAX_LINES  条        单句最多聚合几条字幕，默认 8
//   SUB_MAX_CHARS  字        单句最多聚合多少字，默认 400
//   SUB_ORDER_HINT 1|0       是否加「贴近源语顺序」提示，默认 0
//   SUB_MARK       1|0       组内加 [n] 行号标记，整组一次翻译并逐行对齐，默认 1
#include "llm.h"

#include <nlohmann/json.hpp>

#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include <algorithm>

using json = nlohmann::json;

namespace {

std::mutex g_mutex;

struct Url {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string base = "/v1";
};

Url ParseUrl(const std::string& url) {
  Url u;
  std::string s = url;
  if (s.rfind("http://", 0) == 0) s = s.substr(7);
  else if (s.rfind("https://", 0) == 0) s = s.substr(8);
  size_t slash = s.find('/');
  std::string hostport = (slash == std::string::npos) ? s : s.substr(0, slash);
  u.base = (slash == std::string::npos) ? "" : s.substr(slash);
  size_t colon = hostport.find(':');
  if (colon == std::string::npos) {
    u.host = hostport;
  } else {
    u.host = hostport.substr(0, colon);
    u.port = std::atoi(hostport.c_str() + colon + 1);
  }
  if (u.base.empty()) u.base = "/v1";
  return u;
}

Url BaseUrl() {
  const char* e = std::getenv("LLM_HOST");
  return ParseUrl(e ? e : "http://127.0.0.1:8080/v1");
}

std::string Model() {
  const char* e = std::getenv("LLM_MODEL");
  return e ? e : "hunyuan-mt";
}

int EnvInt(const char* name, int def) {
  const char* e = std::getenv(name);
  if (!e || !e[0]) return def;
  return std::atoi(e);
}

double EnvDouble(const char* name, double def) {
  const char* e = std::getenv(name);
  if (!e || !e[0]) return def;
  return std::atof(e);
}

bool SendAll(int fd, const char* data, size_t len) {
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = ::send(fd, data + sent, len - sent, 0);
    if (n <= 0) return false;
    sent += (size_t)n;
  }
  return true;
}

// 解析 chunked 传输编码
std::string Dechunk(const std::string& body) {
  std::string out;
  size_t i = 0;
  while (i < body.size()) {
    size_t eol = body.find("\r\n", i);
    if (eol == std::string::npos) break;
    std::string szline = body.substr(i, eol - i);
    size_t sc = szline.find(';');
    if (sc != std::string::npos) szline = szline.substr(0, sc);
    long sz = std::strtol(szline.c_str(), nullptr, 16);
    if (sz <= 0) break;
    size_t start = eol + 2;
    out.append(body, start, (size_t)sz);
    i = start + (size_t)sz + 2;  // skip data + CRLF
  }
  return out;
}

std::string Lower(std::string s) {
  for (char& c : s) c = (char)tolower((unsigned char)c);
  return s;
}

bool HttpPost(const std::string& path, const std::string& body, std::string& out) {
  Url u = BaseUrl();
  struct addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(u.host.c_str(), std::to_string(u.port).c_str(), &hints, &res) != 0)
    return false;
  int fd = -1;
  for (auto* p = res; p; p = p->ai_next) {
    fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0) continue;
    if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
    ::close(fd);
    fd = -1;
  }
  freeaddrinfo(res);
  if (fd < 0) return false;

  std::string req = "POST " + path + " HTTP/1.1\r\n";
  req += "Host: " + u.host + ":" + std::to_string(u.port) + "\r\n";
  req += "Content-Type: application/json\r\n";
  req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  req += "Connection: close\r\n\r\n";
  req += body;
  bool ok = SendAll(fd, req.data(), req.size());

  std::string resp;
  if (ok) {
    char buf[16384];
    ssize_t n;
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) resp.append(buf, (size_t)n);
  }
  ::close(fd);
  if (!ok || resp.empty()) return false;

  size_t hdr_end = resp.find("\r\n\r\n");
  if (hdr_end == std::string::npos) return false;
  std::string headers = Lower(resp.substr(0, hdr_end));
  std::string raw = resp.substr(hdr_end + 4);

  if (headers.find("transfer-encoding: chunked") != std::string::npos) {
    out = Dechunk(raw);
  } else {
    out = raw;
  }
  return true;
}

std::string Trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

std::string StripTags(const std::string& s) {
  std::string o;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '{') {
      while (i < s.size() && s[i] != '}') ++i;
      continue;
    }
    o += s[i];
  }
  return o;
}

std::string CleanAssText(std::string t) {
  std::string o;
  for (size_t i = 0; i < t.size(); ++i) {
    if (t[i] == '\\' && i + 1 < t.size() && (t[i + 1] == 'N' || t[i + 1] == 'n')) {
      o += ' ';
      ++i;
    } else {
      o += t[i];
    }
  }
  return Trim(StripTags(o));
}

// 把译文放回 ASS 文本域：去掉换行/大括号覆盖标记，保留字符本身。
std::string SanitizeAssText(std::string t) {
  std::string o;
  for (size_t i = 0; i < t.size(); ++i) {
    char c = t[i];
    if (c == '\r' || c == '\n') { o += ' '; continue; }
    if (c == '{') { while (i < t.size() && t[i] != '}') ++i; continue; }
    if (c == '\\' && i + 1 < t.size() && (t[i + 1] == 'N' || t[i + 1] == 'n')) { o += ' '; ++i; continue; }
    o += c;
  }
  return Trim(o);
}

std::vector<std::string> SplitN(const std::string& s, char sep, size_t n) {
  std::vector<std::string> out;
  size_t start = 0;
  while (out.size() + 1 < n) {
    size_t p = s.find(sep, start);
    if (p == std::string::npos) break;
    out.push_back(s.substr(start, p - start));
    start = p + 1;
  }
  out.push_back(s.substr(start));
  return out;
}

std::string LangName(const std::string& code) {
  static const std::vector<std::pair<std::string, std::string>> m = {
      {"zh", "Chinese"},   {"en", "English"},    {"ja", "Japanese"},
      {"ko", "Korean"},    {"fr", "French"},     {"de", "German"},
      {"es", "Spanish"},   {"ru", "Russian"},    {"it", "Italian"},
      {"pt", "Portuguese"},{"ar", "Arabic"},     {"th", "Thai"},
      {"vi", "Vietnamese"},{"id", "Indonesian"}};
  for (auto& p : m)
    if (p.first == code) return p.second;
  return code;
}

// ---------------------- UTF-8 与目标语切分器 ----------------------

std::vector<std::string> Utf8Split(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0; i < s.size();) {
    unsigned char c = (unsigned char)s[i];
    size_t len = 1;
    if (c >= 0xF0) len = 4;
    else if (c >= 0xE0) len = 3;
    else if (c >= 0xC0) len = 2;
    if (i + len > s.size()) len = 1;
    out.push_back(s.substr(i, len));
    i += len;
  }
  return out;
}

unsigned int Utf8Decode(const std::string& ch) {
  unsigned char c = (unsigned char)ch[0];
  if (c < 0x80) return c;
  if ((c >> 5) == 0x6 && ch.size() >= 2)
    return ((unsigned int)(c & 0x1F) << 6) | ((unsigned char)ch[1] & 0x3F);
  if ((c >> 4) == 0xE && ch.size() >= 3)
    return ((unsigned int)(c & 0x0F) << 12) | (((unsigned char)ch[1] & 0x3F) << 6) |
           ((unsigned char)ch[2] & 0x3F);
  if ((c >> 3) == 0x1E && ch.size() >= 4)
    return ((unsigned int)(c & 0x07) << 18) | (((unsigned char)ch[1] & 0x3F) << 12) |
           (((unsigned char)ch[2] & 0x3F) << 6) | ((unsigned char)ch[3] & 0x3F);
  return c;
}

bool IsSpace(unsigned int c) { return c == ' ' || c == '\t' || c == 0x3000; }

bool IsCjk(unsigned int c) {
  return (c >= 0x3040 && c <= 0x30FF) || (c >= 0x3400 && c <= 0x4DBF) ||
         (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0xF900 && c <= 0xFAFF) ||
         (c >= 0xFF00 && c <= 0xFFEF) || (c >= 0xAC00 && c <= 0xD7AF) ||
         (c >= 0x3000 && c <= 0x303F);
}

// 「字」：汉字/假名/谚文（全角标点、※、… 之类不算）
bool IsCjkWord(unsigned int c) {
  return (c >= 0x3040 && c <= 0x30FF) || (c >= 0x3400 && c <= 0x4DBF) ||
         (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0xF900 && c <= 0xFAFF) ||
         (c >= 0xAC00 && c <= 0xD7AF);
}

bool IsOpening(unsigned int c) {
  switch (c) {
    case '(': case '[': case '{': case '<': case '"': case '\'':
    case 0x2018: case 0x201C: case 0x300C: case 0x300E: case 0x3010:
    case 0xFF08: case 0x300A: case 0x3008: case 0xFF5B: case 0x3014:
      return true;
  }
  return false;
}

bool IsClosing(unsigned int c) {
  switch (c) {
    case ')': case ']': case '}': case '>': case '"': case '\'':
    case '.': case ',': case '!': case '?': case ';': case ':':
    case 0x2019: case 0x201D: case 0x300D: case 0x300F: case 0x3011:
    case 0xFF09: case 0x300B: case 0x3009: case 0xFF5D: case 0x3015:
    case 0x3002: case 0xFF0C: case 0xFF01: case 0xFF1F: case 0xFF1B:
    case 0xFF1A: case 0x3001: case 0x2026:
      return true;
  }
  return false;
}

bool IsWordChar(unsigned int c) {
  return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
         (c >= 'a' && c <= 'z') || c == '_' || c == '\'';
}

bool IsTrailCloser(unsigned int c) {
  switch (c) {
    case ')': case ']': case '}': case '>': case '"': case '\'':
    case 0x2019: case 0x201D: case 0x300D: case 0x300F: case 0x3011:
    case 0xFF09: case 0x300B: case 0x3009: case 0xFF5D: case 0x3015:
      return true;
  }
  return false;
}

bool IsTerminal(unsigned int c) {
  return c == '.' || c == '!' || c == '?' || c == 0x3002 || c == 0xFF01 ||
         c == 0xFF1F || c == 0x2026;
}

bool IsClausePunct(unsigned int c) {
  switch (c) {
    case ',': case ';': case ':':
    case 0xFF0C: case 0x3001: case 0xFF1B: case 0xFF1A:
      return true;
  }
  return false;
}

// 两个码点之间能否断行（不切断词、不悬标点）。
bool AllowedBreak(unsigned int a, unsigned int b) {
  if (IsSpace(a) || IsSpace(b)) return true;         // 空白处总能切
  if (IsOpening(a)) return false;                    // 开括号/引号后不切
  if (IsClosing(b)) return false;                    // 闭标点/句读前不切
  if (IsWordChar(a) && IsWordChar(b)) return false;  // 词内、数字内不切
  if ((a >= '0' && a <= '9') && (b == '%' || b == 0xFF05)) return false;
  return true;
}

// 判定一行原文是否以句末标点收尾（忽略尾随引号/括号）。
bool EndsSentence(const std::string& s) {
  std::vector<std::string> ch = Utf8Split(s);
  for (int i = (int)ch.size() - 1; i >= 0; --i) {
    unsigned int c = Utf8Decode(ch[i]);
    if (IsSpace(c) || IsTrailCloser(c)) continue;
    return IsTerminal(c);
  }
  return false;
}

// "0:00:04.19" -> 秒
double ParseAssTime(const std::string& t) {
  int h = 0, m = 0;
  double s = 0;
  if (std::sscanf(t.c_str(), "%d:%d:%lf", &h, &m, &s) != 3) return 0;
  return h * 3600.0 + m * 60.0 + s;
}

// 把整句译文按 weights 比例切成等量片；只在合法缝切，尽量贴标点。
std::vector<std::string> SplitByWeights(const std::string& text,
                                        const std::vector<size_t>& weights) {
  std::vector<std::string> chars = Utf8Split(text);
  int L = (int)chars.size();
  int k = (int)weights.size();
  std::vector<std::string> out;
  if (k <= 1 || L == 0) { out.push_back(Trim(text)); return out; }

  if (L <= k) {  // 字太少：逐字分配，其余为空
    for (int i = 0; i < k; ++i) out.push_back(i < L ? chars[i] : std::string());
    return out;
  }

  std::vector<int> cand;
  for (int p = 1; p < L; ++p) {
    if (AllowedBreak(Utf8Decode(chars[p - 1]), Utf8Decode(chars[p]))) cand.push_back(p);
  }

  long W = 0;
  for (size_t w : weights) W += (long)std::max<size_t>(w, 1);

  std::vector<int> cuts;
  long cum = 0;
  for (int i = 0; i < k - 1; ++i) {
    cum += (long)std::max<size_t>(weights[i], 1);
    double target = (double)L * (double)cum / (double)W;
    int lo = cuts.empty() ? 1 : cuts.back() + 1;
    int hi = L - (k - 1 - i);
    if (hi < lo) hi = lo;
    int best = -1;
    double best_cost = 1e18;
    for (int c : cand) {
      if (c < lo) continue;
      if (c > hi) break;
      unsigned int prev = Utf8Decode(chars[c - 1]);
      double cost = std::fabs((double)c - target);
      if (IsTerminal(prev)) cost -= 6.0;       // 优先在句末标点后切
      else if (IsClausePunct(prev)) cost -= 3.0;  // 其次从句标点
      if (cost < best_cost) { best_cost = cost; best = c; }
    }
    if (best < 0) {
      best = (int)std::lround(target);
      if (best < lo) best = lo;
      if (best > hi) best = hi;
    }
    cuts.push_back(best);
  }

  int prev = 0;
  for (int i = 0; i < k; ++i) {
    int end = (i < k - 1) ? cuts[i] : L;
    std::string piece;
    for (int j = prev; j < end; ++j) piece += chars[j];
    out.push_back(Trim(piece));
    prev = end;
  }
  return out;
}

std::string ReadFile(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return "";
  std::string s;
  char buf[8192];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
  std::fclose(f);
  if (s.size() >= 3 && (unsigned char)s[0] == 0xEF && (unsigned char)s[1] == 0xBB &&
      (unsigned char)s[2] == 0xBF)
    s = s.substr(3);
  return s;
}

std::vector<std::string> SplitLines(const std::string& s) {
  std::vector<std::string> out;
  size_t start = 0;
  while (true) {
    size_t p = s.find('\n', start);
    if (p == std::string::npos) {
      out.push_back(s.substr(start));
      break;
    }
    out.push_back(s.substr(start, p - start));
    start = p + 1;
  }
  return out;
}

std::string BuildPrompt(const std::string& target_name, const std::string& text) {
  std::string p = "Translate the following segment into " + target_name +
                  ", without additional explanation.";
  if (EnvInt("SUB_ORDER_HINT", 0) != 0)
    p += " Keep the clause order close to the source.";
  p += "\n\n" + text;
  return p;
}

std::string TranslateOne(const std::string& text, const std::string& target_name,
                           double temperature = 0.0) {
  json payload;
  payload["model"] = Model();
  payload["messages"] = json::array(
      {{{"role", "user"}, {"content", BuildPrompt(target_name, text)}}});
  payload["max_tokens"] = 2048;
  payload["temperature"] = temperature;

  std::string resp;
  if (!HttpPost(BaseUrl().base + "/chat/completions", payload.dump(), resp)) return "";
  try {
    json j = json::parse(resp);
    if (!j.contains("choices") || j["choices"].empty()) return "";
    return SanitizeAssText(j["choices"][0]["message"]["content"].get<std::string>());
  } catch (const std::exception&) {
    return "";
  }
}

struct DialogueRow {
  std::string prefix;    // 字段 0..8 + ","
  std::string original;  // 原始文本域（含 \N 等）
  std::string src;       // CleanAssText 后的纯文本
  std::string actor;     // Name 字段
  double start = 0;
  double end = 0;
};

void EmitRow(const DialogueRow& r, const std::string& trans, std::string& events) {
  if (trans.empty() || trans == r.src)
    events += r.prefix + r.original + "\n";
  else
    events += r.prefix + r.original + "\\N" + trans + "\n";
}

// 标记组的提示词：这些行是**同一个句子被切开的片段**，
// 必须整句连贯地译，再按片段边界切成同样条数，
// 否则模型会给每行各译出一个完整句子（意思重复、逐行不对应）。
// 归一化（只留字母数字与 CJK、转小写）用于判断「译文其实还是原文」或「两行译重了」
std::string NormKey(const std::string& s) {
  std::string o;
  for (size_t i = 0; i < s.size();) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) {
      if (isalnum(c)) o += (char)tolower(c);
      ++i;
    } else {
      size_t n = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : 2;
      // 只保留真正的字：全角逗号/句号/问号不参与比较，
      // 否则 "1,070." → "1,070。" 会被当成「已翻译」而漏过回显检查。
      if (IsCjkWord(Utf8Decode(s.substr(i, n)))) o += s.substr(i, n);
      i += n;
    }
  }
  return o;
}

bool HasAsciiAlpha(const std::string& s) {
  for (unsigned char c : s)
    if (c < 0x80 && isalpha(c)) return true;
  return false;
}

// 真正的「字」：汉字/假名/谚文。不含 0x3000 段与 0xFF00 段的全角标点，
// 否则 "Akron Capital？" 这种只把问号换成全角的也会被当成已译成中文。
bool HasCjk(const std::string& s) {
  for (const auto& ch : Utf8Split(s))
    if (IsCjkWord(Utf8Decode(ch))) return true;
  return false;
}

// 目标语言是不是中文（决定「译文里必须出现汉字」这条硬检查）
bool WantsCjkTarget(const std::string& target) {
  std::string t;
  for (char c : target) t += (char)tolower((unsigned char)c);
  return t.find("chinese") != std::string::npos ||
         t.find("zh") != std::string::npos || t.find("中") != std::string::npos;
}

// 译文质量兜底：空、只有标点、原文回显、源文有英文却译出零汉字 —— 都算没译好。
// 这些行以前会原样带译文（如 "now!" → "！"、"Repeat that in English." →
// "Please repeat that in English."），现在退回重译，仍不行就只保留原文。
bool BadTranslation(const std::string& src, const std::string& trans,
                    const std::string& target) {
  std::string k = NormKey(trans);
  if (k.empty()) return true;                     // 译空 / 只剩标点
  if (k == NormKey(src)) return true;             // 原文回显
  if (WantsCjkTarget(target) && HasAsciiAlpha(src) && !HasCjk(trans))
    return true;                                  // 英文原文却没译出中文
  return false;
}

std::string TranslateOneChecked(const std::string& text, const std::string& target_name);

std::string BuildMarkPrompt(const std::string& target_name, int n,
                            const std::string& block) {
  std::string p =
      "下面每行开头的编号是字幕行号（共 " + std::to_string(n) +
      " 行，有的行是同一句被切开的片段）。请结合整段上下文译成 " + target_name +
      "，逐行对应输出：每行一个编号 [i] 加该行译文，行数与编号必须与原文完全一致，"
      "不要合并或拆分行，不要重复整句意思。";
  p += "\n\n" + block;
  return p;
}

// 带行号标记的整组翻译：整组原文一次性送模型（上下文完整），
// 每行前缀 "[n] " 让模型逐行输出同号译文，从而天然逐行对齐；
// 模型若丢标记/合并行，则返回空，由上层判断是否退回原路径。
std::string TranslateBlock(const std::string& block, const std::string& target_name,
                           int n, double temperature = 0) {
  json payload;
  payload["model"] = Model();
  payload["messages"] = json::array(
      {{{"role", "user"}, {"content", BuildMarkPrompt(target_name, n, block)}}});
  payload["max_tokens"] = 2048;
  payload["temperature"] = temperature;

  std::string resp;
  if (!HttpPost(BaseUrl().base + "/chat/completions", payload.dump(), resp)) return "";
  try {
    json j = json::parse(resp);
    if (!j.contains("choices") || j["choices"].empty()) return "";
    return j["choices"][0]["message"]["content"].get<std::string>();
  } catch (const std::exception&) {
    return "";
  }
}

// 从带 "[n]" 标记的输出里取出第 n 行译文（n 从 1 开始）。
// 模型可能把多行并到某一行、或丢标记，因此按「标记出现顺序」切段，
// 缺号的那几行留空，由上层判断是否采用。
std::vector<std::string> SplitByMarkers(const std::string& raw, int n,
                                        const std::vector<std::string>& src) {
  std::vector<std::string> out(n);
  // 模型常先把原文整段回显一遍再翻译，标记会出现两轮：
  // 先收集所有标记出现位置，取「最后一轮按 1..n 递增、且段数够」的那一轮。
  struct Hit { size_t pos; int idx; };
  std::vector<Hit> hits;
  for (int i = 1; i <= n; ++i) {
    std::string tag = "[" + std::to_string(i) + "]";
    for (size_t p = raw.find(tag); p != std::string::npos; p = raw.find(tag, p + 1))
      hits.push_back({p, i});
  }
  std::stable_sort(hits.begin(), hits.end(),
                   [](const Hit& a, const Hit& b) { return a.pos < b.pos; });
  // 找最后一轮从 1 开始、严格递增的序列
  size_t best = 0;
  bool found = false;
  for (size_t i = 0; i < hits.size(); ++i) {
    if (hits[i].idx != 1) continue;
    size_t j = i, want = 1;
    while (j < hits.size() && hits[j].idx == want) { ++j; ++want; }
    if ((int)(want - 1) >= n) { best = i; found = true; }  // 取最后一轮完整的
  }
  if (!found) return out;
  for (int i = 0; i < n; ++i) {
    size_t b = hits[best + i].pos + std::string("[" + std::to_string(i + 1) + "]").size();
    size_t e = raw.size();
    for (size_t h = best + i + 1; h < hits.size(); ++h) {
      std::string t2 = "[" + std::to_string(hits[h].idx) + "]";
      size_t q = hits[h].pos;
      if (hits[h].idx <= n && q > b) { e = q; break; }
      (void)t2;
    }
    std::string seg = Trim(SanitizeAssText(raw.substr(b, e - b)));
    seg = Trim(std::string(seg.c_str()));  // 去掉行首换行/冒号残留
    while (!seg.empty() && (seg[0] == '\n' || seg[0] == ':' || seg[0] == '\r' ||
                            seg[0] == ' '))
      seg.erase(0, 1);
    // 段内可能还带「翻译：」之类的引导词与换行残留
    size_t nl = seg.find('\n');
    if (nl != std::string::npos && seg.find('\n', nl + 1) != std::string::npos)
      seg = Trim(seg.substr(0, nl));
    if (i < (int)src.size() && seg == Trim(src[i])) seg.clear();  // 原文回显不算译文
    out[i] = seg;
  }
  return out;
}

// 单行翻译 + 质量兜底：不合格就换温度重试一次，仍不合格返回空
// （EmitRow 对空译文只输出原文，不会留下半截鬼话）。
std::string TranslateOneChecked(const std::string& text, const std::string& target_name) {
  if (text.empty()) return "";
  std::string t = TranslateOne(text, target_name, 0.0);
  if (!BadTranslation(text, t, target_name)) return t;
  std::string t2 = TranslateOne(text, target_name, 0.3);
  if (!BadTranslation(text, t2, target_name)) return t2;
  return "";
}

// 把一组连续字幕条当作一个句子翻译，再按原文字数比例回切。
void ProcessGroup(const std::vector<DialogueRow>& g, const std::string& target_name,
                  std::string& events, int& count, int depth = 0);

// 逐行对应地译一段连续字幕：整段带 [i] 一次性送模型（上下文完整、逐行不错位）。
// 返回 false 表示这批没译好，交给上层兜底。
bool TranslateMarked(const std::vector<DialogueRow>& g, const std::string& target_name,
                     std::vector<std::string>& got) {
  std::vector<std::string> srcs;
  std::string block;
  for (const auto& r : g) {
    if (r.src.empty()) continue;
    srcs.push_back(r.src);
    block += "[" + std::to_string(srcs.size()) + "] " + r.src + "\n";
  }
  int n = (int)srcs.size();
  if (n == 0) return false;
  for (int attempt = 0; attempt < 2; ++attempt) {
    got = SplitByMarkers(TranslateBlock(block, target_name, n, attempt ? 0.3 : 0.0), n, srcs);
    bool bad = false;
    std::set<std::string> seen;
    for (int i = 0; i < n; ++i) {
      std::string k = NormKey(got[i]);
      if (BadTranslation(srcs[i], got[i], target_name)) { bad = true; break; }
      if (k.empty() || k == NormKey(srcs[i])) { bad = true; break; }  // 漏行/回显原文
      if (!seen.insert(k).second) { bad = true; break; }              // 两行译成同一句
    }
    if (!bad) return true;
  }
  return false;
}

void EmitGroup(const std::vector<DialogueRow>& g, const std::string& target_name,
               std::string& events, int& count, int depth) {
  size_t idx = 0;
  // 只处理有原文的行；空行原样输出
  while (idx < g.size()) {
    while (idx < g.size() && g[idx].src.empty()) {
      EmitRow(g[idx], "", events);
      ++count;
      ++idx;
    }
    size_t end = idx;
    while (end < g.size() && !g[end].src.empty()) ++end;
    if (idx >= end) break;
    std::vector<DialogueRow> run(g.begin() + idx, g.begin() + end);
    ProcessGroup(run, target_name, events, count, depth);
    idx = end;
  }
}

void ProcessGroup(const std::vector<DialogueRow>& g, const std::string& target_name,
                  std::string& events, int& count, int depth) {
  if (g.size() == 1) {
    const DialogueRow& r = g[0];
    EmitRow(r, r.src.empty() ? "" : TranslateOneChecked(r.src, target_name), events);
    ++count;
    return;
  }

  // ① 上下文对齐路径
  if (EnvInt("SUB_MARK", 1) != 0 && depth < 3) {
    std::vector<std::string> got;
    if (TranslateMarked(g, target_name, got)) {
      size_t k = 0;
      for (const auto& r : g) {
        std::string t;
        if (!r.src.empty()) t = got[k++];
        EmitRow(r, t, events);
        ++count;
      }
      return;
    }
    // 部分行对不上：能译的先落盘，对不上的相邻行缩成小组再走一遍（更短更不容易串行）
    std::vector<std::string> srcs;
    std::string block;
    for (const auto& r : g) {
      if (r.src.empty()) continue;
      srcs.push_back(r.src);
      block += "[" + std::to_string(srcs.size()) + "] " + r.src + "\n";
    }
    int n = (int)srcs.size();
    std::vector<std::string> raw = SplitByMarkers(TranslateBlock(block, target_name, n), n, srcs);
    std::vector<bool> okv(n, false);
    std::set<std::string> seen;
    int ok = 0;
    for (int i = 0; i < n; ++i) {
      std::string k = NormKey(raw[i]);
      if (BadTranslation(srcs[i], raw[i], target_name)) continue;
      if (k.empty() || k == NormKey(srcs[i]) || !seen.insert(k).second) continue;
      okv[i] = true;
      ++ok;
    }
    if (ok * 2 >= n) {
      size_t k = 0;
      size_t i = 0;
      while (i < g.size()) {
        if (g[i].src.empty()) {
          EmitRow(g[i], "", events);
          ++count;
          ++i;
          continue;
        }
        if (okv[k]) {
          EmitRow(g[i], raw[k], events);
          ++count;
          ++k;
          ++i;
          continue;
        }
        size_t j = i;                       // 收集连续对不上的行
        std::vector<DialogueRow> bad;
        while (j < g.size() && !g[j].src.empty() && !okv[k]) {
          bad.push_back(g[j]);
          ++k;
          ++j;
        }
        ProcessGroup(bad, target_name, events, count, depth + 1);
        i = j;
      }
      return;
    }
  }

  // 拼接整句（CJK 之间不加空格，其余加空格）
  std::string joined;
  for (const auto& r : g) {
    if (r.src.empty()) continue;
    if (joined.empty()) { joined = r.src; continue; }
    std::vector<std::string> a = Utf8Split(joined);
    std::vector<std::string> b = Utf8Split(r.src);
    bool tight = !a.empty() && !b.empty() && IsCjk(Utf8Decode(a.back())) &&
                 IsCjk(Utf8Decode(b.front()));
    joined += tight ? r.src : (" " + r.src);
  }

  std::string trans = joined.empty() ? "" : TranslateOne(joined, target_name);
  if (trans.empty()) {  // 整句失败：退回逐条
    for (const auto& r : g) {
      EmitRow(r, r.src.empty() ? "" : TranslateOneChecked(r.src, target_name), events);
      ++count;
    }
    return;
  }

  std::vector<size_t> weights;
  for (const auto& r : g) weights.push_back(Utf8Split(r.src).size());
  std::vector<std::string> pieces = SplitByWeights(trans, weights);
  for (size_t i = 0; i < g.size(); ++i) {
    std::string piece = i < pieces.size() ? pieces[i] : std::string();
    // 回切出来的碎片同样要过质检：空的、只剩标点/括号的、原文回显的，
    // 单独重译这一条（回切是按字数比例切的，偶尔会把括号尾巴切到别行）。
    if (BadTranslation(g[i].src, piece, target_name))
      piece = TranslateOneChecked(g[i].src, target_name);
    EmitRow(g[i], piece, events);
    ++count;
  }
}

}  // namespace

// =============================== C ABI ======================================

const char* llm_host(void) {
  static std::string h = [] { return BaseUrl().host + ":" + std::to_string(BaseUrl().port); }();
  return h.c_str();
}

const char* llm_translate(const char* text, const char* target_name) {
  std::lock_guard<std::mutex> lk(g_mutex);
  static thread_local std::string out;
  out = TranslateOne(text ? text : "", target_name ? target_name : "Chinese");
  return out.c_str();
}

int llm_translate_ass(const char* in_ass, const char* out_ass, const char* target_lang) {
  std::lock_guard<std::mutex> lk(g_mutex);
  std::string content = ReadFile(in_ass);
  if (content.empty()) return -1;

  std::vector<std::string> lines = SplitLines(content);
  size_t events_start = std::string::npos;
  for (size_t i = 0; i < lines.size(); ++i) {
    if (lines[i].rfind("[Events]", 0) == 0) { events_start = i; break; }
  }
  if (events_start == std::string::npos) return -1;

  std::string header;
  for (size_t i = 0; i <= events_start + 1 && i < lines.size(); ++i)
    header += lines[i] + "\n";

  const std::string target_name = LangName(target_lang ? target_lang : "zh");

  // 预扫描源语用字：中日韩等不分词语言默认关闭整句聚合
  int script_cjk = 0, script_latin = 0;
  for (size_t i = events_start + 2; i < lines.size(); ++i) {
    if (lines[i].rfind("Dialogue:", 0) != 0) continue;
    std::vector<std::string> parts = SplitN(lines[i], ',', 10);
    if (parts.size() < 10) continue;
    for (const auto& ch : Utf8Split(CleanAssText(parts[9]))) {
      unsigned int c = Utf8Decode(ch);
      if (IsCjk(c)) ++script_cjk;
      else if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) ++script_latin;
    }
  }
  int grp_env = EnvInt("SUB_GROUP", -1);
  bool grouping = (grp_env < 0) ? (script_cjk <= script_latin) : (grp_env != 0);

  const double pause_sec = EnvDouble("SUB_PAUSE", 0.7);
  const int max_lines = grouping ? EnvInt("SUB_MAX_LINES", 8) : 1;
  const int max_chars = EnvInt("SUB_MAX_CHARS", 400);

  std::string events;
  int count = 0;
  std::vector<DialogueRow> group;

  auto flush = [&]() {
    if (group.empty()) return;
    ProcessGroup(group, target_name, events, count, 0);
    group.clear();
  };

  for (size_t i = events_start + 2; i < lines.size(); ++i) {
    const std::string& line = lines[i];
    if (line.rfind("Dialogue:", 0) != 0) {
      flush();
      events += line + "\n";
      continue;
    }
    std::vector<std::string> parts = SplitN(line, ',', 10);
    if (parts.size() < 10) {
      flush();
      events += line + "\n";
      continue;
    }

    DialogueRow r;
    for (int k = 0; k < 9; ++k) r.prefix += parts[k] + ",";
    r.original = parts[9];
    r.src = CleanAssText(r.original);
    r.actor = parts[4];
    r.start = ParseAssTime(parts[1]);
    r.end = ParseAssTime(parts[2]);

    if (r.src.empty()) {  // 无文本：原样输出，不与相邻聚合
      flush();
      events += r.prefix + r.original + "\n";
      ++count;
      continue;
    }

    if (!group.empty()) {
      const DialogueRow& p = group.back();
      bool boundary = p.actor != r.actor || (r.start - p.end) >= pause_sec ||
                      EndsSentence(p.src) || (int)group.size() >= max_lines;
      if (boundary) flush();
    }
    group.push_back(r);

    size_t gl = 0;
    for (const auto& x : group) gl += Utf8Split(x.src).size();
    if ((int)gl >= max_chars) flush();
  }
  flush();

  std::string result = header + events;
  FILE* f = std::fopen(out_ass, "wb");
  if (!f) return -1;
  const unsigned char bom[3] = {0xEF, 0xBB, 0xBF};
  std::fwrite(bom, 1, 3, f);
  std::fwrite(result.data(), 1, result.size(), f);
  std::fclose(f);
  return count;
}
