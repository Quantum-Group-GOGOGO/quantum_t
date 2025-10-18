# -*- coding: utf-8 -*-
import threading
import queue
import time
import sys
import os
import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output import DummyOutput

from IB_realsync_main import main_Program

IN_COMMAND_MODE = threading.Event()
log_queue1 = queue.Queue()
log_queue2 = queue.Queue()


class QueueWriter:
    """
    进程级 stdout 代理：
      - Worker 线程写入 -> 放入 log_queue（异步打印）
      - 其他线程写入 -> 直接写回原 stdout
    兼容 TTY 探测：实现 isatty()/fileno()/writable()/encoding 等属性
    """
    def __init__(self, q, old_stream):
        self.q = q
        self.old = old_stream
        # prompt_toolkit 有时会访问 .encoding
        self.encoding = getattr(old_stream, "encoding", "utf-8")

    def write(self, s):
        if not s:
            return
        if threading.current_thread().name == "Worker":
            self.q.put(s)
        else:
            self.old.write(s)

    def writelines(self, lines):
        for s in lines:
            self.write(s)

    def flush(self):
        try:
            self.old.flush()
        except Exception:
            pass

    def isatty(self):
        # 让 prompt_toolkit 能正常判断；在大多数情况下沿用原 stdout 的特性
        try:
            if hasattr(self.old, "isatty"):
                return self.old.isatty()
        except Exception:
            pass
        return False

    def fileno(self):
        # 有些库会调用 fileno；若不可用则抛出合理异常
        if hasattr(self.old, "fileno"):
            try:
                return self.old.fileno()
            except Exception:
                pass
        raise OSError("QueueWriter has no valid fileno")

    def readable(self):
        return False

    def writable(self):
        return True

    def close(self):
        # 不主动关闭底层 old stdout
        pass

    def __getattr__(self, name):
        # 兜底把其他属性委托给 old_stream（如 errors/newlines 等）
        return getattr(self.old, name)


def _has_tty():
    try:
        return (
            hasattr(sys.stdin, "isatty") and sys.stdin.isatty() and
            hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        )
    except Exception:
        return False


def worker_wrapper(program):
    # 把这个线程叫成 "Worker"
    threading.current_thread().name = "Worker"

    # 独立事件循环（某些异步库在子线程需要）
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # 全局重定向 stdout 为可分流的 QueueWriter（注意：这是进程级别的）
    old_stdout = sys.stdout
    old_stderr = sys.stderr  # 目前不重定向 stderr，如需可同样包装

    sys.stdout = QueueWriter(log_queue1, old_stdout)

    try:
        program.run()
    finally:
        # 恢复
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def log_printer():
    buffer = []
    while True:
        msg = log_queue1.get()
        if msg is None:
            break
        if IN_COMMAND_MODE.is_set():
            buffer.append(msg)
        else:
            if buffer:
                for m in buffer:
                    sys.stdout.write(m)
                sys.stdout.flush()
                buffer.clear()
            sys.stdout.write(msg)
            sys.stdout.flush()


def main():
    # 允许通过环境变量强制无交互模式
    HEADLESS = os.environ.get("HEADLESS", "0") == "1"
    TTY_OK = _has_tty()

    main_program = main_Program()

    # 启动日志打印线程
    threading.Thread(target=log_printer, daemon=True).start()

    # 启动 IB 线程
    worker = threading.Thread(
        target=worker_wrapper,
        args=(main_program,),
        daemon=True
    )
    worker.start()

    # ---- Prompt/REPL 区域 ----
    kb = KeyBindings()

    @kb.add('c-x')
    def _(event):
        if IN_COMMAND_MODE.is_set():
            IN_COMMAND_MODE.clear()
        else:
            IN_COMMAND_MODE.set()
        event.app.exit()

    # 在无 TTY 或 HEADLESS 环境，走“无交互模式”
    if HEADLESS or not TTY_OK:
        # 无交互：保持主线程活着，支持 Ctrl+C 安全退出
        print("[test4] 运行在无交互模式（headless or no TTY）。按 Ctrl+C 触发保存并退出。")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            try:
                main_program.save()
                print("\n检测到中断，已保存。")
            except Exception as e:
                print(f"\n保存出错: {e}")
            finally:
                log_queue1.put(None)
                sys.exit(0)
        return

    # 交互模式（TTY 存在）
    # 如果你的 stdout 被我们换成了 QueueWriter，也具备 isatty/fileno，不会再触发 prompt_toolkit 的报错
    session = PromptSession(key_bindings=kb)

    while True:
        if IN_COMMAND_MODE.is_set():
            try:
                cmd = session.prompt("命令模式 > ")
            except (EOFError, KeyboardInterrupt):
                IN_COMMAND_MODE.clear()
                continue

            c = cmd.strip()
            if c == "exit":
                IN_COMMAND_MODE.clear()
                print("回到主界面记录")
            elif c == "save":
                try:
                    main_program.save()
                    print("已执行 main_program.save()，保存完成。")
                except Exception as e:
                    print(f"保存过程中出错: {e}")
            else:
                try:
                    res = eval(cmd, globals(), locals())
                    print(f"结果: {res}")
                except Exception as e:
                    print(f"命令错误: {e}")

        else:
            # 主界面：空提示行，仅用于捕获 Ctrl+C
            try:
                session.prompt("", prompt_continuation=lambda w, l: "")
            except KeyboardInterrupt:
                try:
                    main_program.save()
                    print("\n检测到中断，退出。")
                except Exception as e:
                    print(f"\n退出保存时出错: {e}")
                finally:
                    log_queue1.put(None)
                    sys.exit(0)


if __name__ == "__main__":
    main()
