/* asswav.c — ass_retime 用的最小 WAV 读取 C ABI
 *
 * 只管「把 PCM 变成可算的采样」这一件 StaticPy 做不了的事（二进制 float/bytes），
 * 重定时算法本身全在 ass_retime.static.py 里。
 *
 * 句柄用 mmap 只读映射，177MB 的音轨不复制，采样按需取。
 *
 *   void*  aw_open(const char* path)   成功返回句柄，失败返回 0
 *   long   aw_len(void* h)             总帧数（每声道采样数）
 *   int    aw_rate(void* h)            采样率
 *   int    aw_ch(void* h)              声道数
 *   double aw_rms(void* h, int start, int n)   多声道混成单声道后 [start, start+n) 的 RMS（0..1）
 *   int    aw_close(void* h)           总是返回 0
 *   int    aw_bom(const char* path)    文件是否以 UTF-8 BOM 开头（1/0/-1）
 *
 * aw_bom 单独存在的原因：StaticPy 的 read_text_file 会吃掉 BOM，
 * 字幕改写后必须把 BOM 原样补回去，否则部分播放器/工具链识别不了。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>

typedef struct {
    int16_t *pcm;   /* 指向 mmap 里的数据区 */
    long     n;     /* 每声道帧数 */
    int      sr;
    int      ch;
    int      fd;
    void    *base;
    size_t   maplen;
} aw_t;

static uint32_t rd32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint16_t rd16(const unsigned char *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

void *aw_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 44) { close(fd); return 0; }
    size_t len = (size_t)st.st_size;
    void *base = mmap(0, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { close(fd); return 0; }
    const unsigned char *b = (const unsigned char *)base;

    if (memcmp(b, "RIFF", 4) != 0 || memcmp(b + 8, "WAVE", 4) != 0) {
        munmap(base, len); close(fd); return 0;
    }

    int sr = 0, ch = 0, bits = 0, fmt = 0;
    long data_off = -1, data_len = 0;
    size_t p = 12;
    while (p + 8 <= len) {
        uint32_t sz = rd32(b + p + 4);
        if (memcmp(b + p, "fmt ", 4) == 0 && p + 8 + 16 <= len) {
            fmt  = rd16(b + p + 8);
            ch   = rd16(b + p + 10);
            sr   = (int)rd32(b + p + 12);
            bits = rd16(b + p + 22);
        } else if (memcmp(b + p, "data", 4) == 0) {
            data_off = (long)(p + 8);
            data_len = (long)sz;
            if (data_off + data_len > (long)len) data_len = (long)len - data_off;
        }
        p += 8 + (size_t)sz + (sz & 1u);
    }
    if (data_off < 0 || ch <= 0 || sr <= 0) { munmap(base, len); close(fd); return 0; }
    if (bits != 16 || fmt == 3) {   /* 只支持 16-bit PCM；float WAV 直接拒 */
        munmap(base, len); close(fd); return 0;
    }

    aw_t *h = (aw_t *)calloc(1, sizeof(aw_t));
    if (!h) { munmap(base, len); close(fd); return 0; }
    h->pcm = (int16_t *)(b + data_off);
    h->n = data_len / (2 * ch);
    h->sr = sr;
    h->ch = ch;
    h->fd = fd;
    h->base = base;
    h->maplen = len;
    return h;
}

long aw_len(void *v)  { aw_t *h = (aw_t *)v; return h ? h->n : 0; }
int  aw_rate(void *v) { aw_t *h = (aw_t *)v; return h ? h->sr : 0; }
int  aw_ch(void *v)   { aw_t *h = (aw_t *)v; return h ? h->ch : 0; }

double aw_rms(void *v, int start, int n) {
    aw_t *h = (aw_t *)v;
    if (!h || n <= 0) return 0.0;
    if (start < 0) start = 0;
    if ((long)start >= h->n) return 0.0;
    long end = (long)start + n;
    if (end > h->n) end = h->n;
    double s = 0.0;
    if (h->ch == 1) {
        const int16_t *p = h->pcm + start;
        for (long i = 0; i < end - start; i++) {
            double x = (double)p[i];
            s += x * x;
        }
        return sqrt(s / (double)(end - start)) / 32768.0;
    }
    const int16_t *p = h->pcm + (long)start * h->ch;
    long cnt = end - start;
    for (long i = 0; i < cnt; i++) {
        double acc = 0.0;
        for (int c = 0; c < h->ch; c++) acc += (double)p[i * h->ch + c];
        acc /= (double)h->ch;
        s += acc * acc;
    }
    return sqrt(s / (double)cnt) / 32768.0;
}

int aw_bom(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    unsigned char b[3];
    ssize_t n = read(fd, b, 3);
    close(fd);
    if (n < 3) return 0;
    return (b[0] == 0xEF && b[1] == 0xBB && b[2] == 0xBF) ? 1 : 0;
}

int aw_close(void *v) {
    aw_t *h = (aw_t *)v;
    if (!h) return 0;
    if (h->base) munmap(h->base, h->maplen);
    if (h->fd >= 0) close(h->fd);
    free(h);
    return 0;
}
