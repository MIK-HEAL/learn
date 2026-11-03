#include "console.hpp"

#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace console {
namespace {

/// 用 vsnprintf 的两次调用法把可变参数格式化成 UTF-8 字符串。
std::string formatArgs(const char* format, std::va_list args) {
    std::va_list counter;
    va_copy(counter, args);
    const int length = std::vsnprintf(nullptr, 0, format, counter);
    va_end(counter);
    if (length <= 0) {
        return {};
    }

    std::vector<char> buffer(static_cast<std::size_t>(length) + 1, '\0');
    std::vsnprintf(buffer.data(), buffer.size(), format, args);
    return std::string(buffer.data(), static_cast<std::size_t>(length));
}

#ifdef _WIN32
/// 目标句柄是控制台时按宽字符写（避开代码页），否则原样写 UTF-8 字节。
void writeUtf8(std::FILE* stream, const std::string& text) {
    HANDLE handle = INVALID_HANDLE_VALUE;
    if (stream == stdout) {
        handle = GetStdHandle(STD_OUTPUT_HANDLE);
    } else if (stream == stderr) {
        handle = GetStdHandle(STD_ERROR_HANDLE);
    }

    DWORD mode = 0;
    const bool isConsole = handle != INVALID_HANDLE_VALUE && handle != nullptr &&
                           GetConsoleMode(handle, &mode) != 0;
    if (isConsole) {
        const int byteCount = static_cast<int>(text.size());
        const int wideLength =
            MultiByteToWideChar(CP_UTF8, 0, text.data(), byteCount, nullptr, 0);
        if (wideLength > 0) {
            std::wstring wide(static_cast<std::size_t>(wideLength), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, text.data(), byteCount, wide.data(),
                                wideLength);
            DWORD written = 0;
            ::WriteConsoleW(handle, wide.data(), static_cast<DWORD>(wideLength),
                            &written, nullptr);
            return;
        }
    }

    std::fwrite(text.data(), 1, text.size(), stream);
    std::fflush(stream);
}
#else
void writeUtf8(std::FILE* stream, const std::string& text) {
    std::fwrite(text.data(), 1, text.size(), stream);
    std::fflush(stream);
}
#endif

} // namespace

void init() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
}

void print(const char* format, ...) {
    std::va_list args;
    va_start(args, format);
    const std::string text = formatArgs(format, args);
    va_end(args);
    writeUtf8(stdout, text);
}

void eprint(const char* format, ...) {
    std::va_list args;
    va_start(args, format);
    const std::string text = formatArgs(format, args);
    va_end(args);
    writeUtf8(stderr, text);
}

} // namespace console
