#pragma once

/// 控制台中文输出。
///
/// Windows 控制台默认代码页是 936(GBK)，而本项目的源码是 UTF-8 并且用 /utf-8 编译，
/// 字符串字面量的字节流就是 UTF-8；直接 printf 出去会被终端按 GBK 解码，
/// 于是「撞击部件 #1」显示成「鎾炲嚮閮ㄤ欢 #1」。
///
/// 这里的做法是：检测到输出目标是真正的控制台时改用 WriteConsoleW 写宽字符
/// （任何代码页下都能正确显示中文），被重定向到文件/管道时则原样输出 UTF-8 字节
/// （日志文件保持 UTF-8，Python 侧读起来也没问题）。
namespace console {

/// 关闭 stdout/stderr 的缓冲，让 WriteConsoleW 的输出与其它库的输出不互相插队。
/// 在 main() 最开始调用一次即可，重复调用无副作用。
void init();

/// 等价于 printf，但保证中文在 Windows 控制台上不乱码。
void print(const char* format, ...);

/// 等价于 fprintf(stderr, ...)，但保证中文在 Windows 控制台上不乱码。
void eprint(const char* format, ...);

} // namespace console
