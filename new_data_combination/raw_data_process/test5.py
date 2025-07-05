#!/usr/bin/env python3
import subprocess
import sys
import time
import signal


# 用来区分「父进程」和「子进程」，参数 --child 传给子进程
IS_CHILD = '--child' in sys.argv

def run_child():
    """
    这里放你原来的 IB 逻辑：connect、subscribe、ib.run() ...
    如果断线了，直接 sys.exit(1) 来告诉父进程「重启」。
    """
    from ib_insync import IB, Future
    import threading
    import test4

    # 如果想自动重启，断线时就 exit(1)
    

    test4.main()



def main():
    if IS_CHILD:
        # 子进程分支：运行 IB 逻辑
        run_child()
        return

    # 父进程分支：做一个重启循环
    print("启动父进程，看护子进程… (按 Ctrl+C 可停止)")
    try:
        while True:
            # 启动子进程
            args = [sys.executable] + sys.argv + ['--child']
            p = subprocess.Popen(args)
            # 等它结束
            ret = p.wait()
            if ret == 0:
                # 子进程正常退出（如 Ctrl+C），则父进程也退出
                print("子进程正常退出，父进程也退出。")
                break
            else:
                # 子进程非 0 退出（如断线触发 sys.exit(1)），重启它
                print(f"子进程退出码 {ret}，5 秒后重启…")
                time.sleep(5)
    except KeyboardInterrupt:
        print("父进程收到 Ctrl+C，退出。")

if __name__ == "__main__":
    main()