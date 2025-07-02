import threading
import queue
import time
import sys
import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from IB_realsync_main import main_Program

IN_COMMAND_MODE = threading.Event()
log_queue1 = queue.Queue()
log_queue2 = queue.Queue()

class QueueWriter:
    """
    一个“分流”用的 stdout/stderr：
     - Worker 线程的输出，放到 log_queue；
     - 其它线程的输出，直接写回原来的 old_stream。
    """
    def __init__(self, q, old_stream):
        self.q = q
        self.old = old_stream

    def write(self, s):
        # 空串、控制字符就略过
        if not s:
            return
        # 根据线程名决定走哪条路
        if threading.current_thread().name == "Worker":
            # Worker 线程 → 丢给 log_queue
            self.q.put(s)
        else:
            # 其它线程 → 直接写到原来的 stdout/stderr
            self.old.write(s)

    def flush(self):
        # 保证 old_stream 得以 flush
        self.old.flush()

def worker_wrapper(program):
    # 把这个线程叫成 "Worker"
    threading.current_thread().name = "Worker"
    

    # 1. 给线程绑一个新的 asyncio loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # 2. 全局重定向 stdout/stderr，分流到 QueueWriter
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    sys.stdout = QueueWriter(log_queue1, old_stdout)
    
    try:
        program.run()
    finally:
        # 3. 运行完（或出错）后恢复
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
    main_program = main_Program()

    # 启动日志打印线程
    threading.Thread(target=log_printer, daemon=True).start()

    # 启动 IB 线程，但用 wrapper 先设置好事件循环
    worker = threading.Thread(
        target=worker_wrapper,
        args=(main_program,),
        daemon=True
    )
    worker.start()

    # prompt_toolkit 部分保持不变……
    kb = KeyBindings()
    session = PromptSession(key_bindings=kb)

    @kb.add('c-x')
    def _(event):
        IN_COMMAND_MODE.set() if not IN_COMMAND_MODE.is_set() else IN_COMMAND_MODE.clear()
        event.app.exit()

    while True:
        if IN_COMMAND_MODE.is_set():
            try:
                cmd = session.prompt("命令模式 > ")
            except (EOFError, KeyboardInterrupt):
                IN_COMMAND_MODE.clear()
                continue

            if cmd.strip() == "exit":
                IN_COMMAND_MODE.clear()
                print("回到主界面记录")

            elif cmd.strip() == "save":
                # 调用你的保存方法
                try:
                    main_program.save()
                    print("已执行 main_program.save()，保存完成。")
                except Exception as e:
                    print(f"保存过程中出错: {e}")

            else:
                # 其余命令仍旧走 eval（或你自己的命令解析）
                try:
                    res = eval(cmd, globals(), locals())
                    print(f"结果: {res}")
                except Exception as e:
                    print(f"命令错误: {e}")

        else:
            try:
                session.prompt("", prompt_continuation=lambda w, l: "")
            except KeyboardInterrupt:
                main_program.save()
                print("\n检测到中断，退出。")
                log_queue1.put(None)
                sys.exit(0)

if __name__ == "__main__":
    main()