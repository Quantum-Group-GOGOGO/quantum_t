import sys
import os
import platform
import importlib
import shutil
from textwrap import indent

def safe_import(name):
    try:
        mod = importlib.import_module(name)
        return mod, None
    except Exception as e:
        return None, e

def version_of(mod):
    try:
        return mod.__version__
    except Exception:
        return "unknown"

def path_of(mod):
    try:
        return getattr(mod, "__file__", "unknown")
    except Exception:
        return "unknown"

def is_venv():
    # 最可靠：比较 sys.prefix 与 sys.base_prefix
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix

def detect_mixed_sites():
    suspicious = []
    for p in sys.path:
        if not isinstance(p, str):
            continue
        if "site-packages" in p and not p.startswith(sys.prefix):
            suspicious.append(p)
    # 特别检查常见的冲突路径
    user_site = os.path.expanduser("~/Library/Python")
    brew_site = "/usr/local/lib/python"
    for p in sys.path:
        if isinstance(p, str) and (p.startswith(user_site) or p.startswith(brew_site)):
            if not p.startswith(sys.prefix):
                if p not in suspicious:
                    suspicious.append(p)
    return suspicious

def recommend_versions(numpy_ver, scipy_ver, torch_ver):
    """
    根据常用稳定组合给出建议：
    - 推荐组合： NumPy 2.2.x + SciPy 1.14.* + Torch 2.3.*（CPU）
    兼容性要点：
      * SciPy 1.14.* 兼容 NumPy 2.2.*
      * 许多项目目前不建议 NumPy 2.3.*（尤其你遇到 SciPy 的 <2.3.0 限制）
    """
    target = {
        "numpy": "2.2.2",
        "scipy": "1.14.*",
        "pandas": "2.2.*",
        "torch": "2.3.*",
    }

    notes = []
    # 检查 SciPy 的常见报错条件
    if scipy_ver and numpy_ver:
        try:
            from packaging.version import Version, InvalidVersion
            v_np = Version(numpy_ver)
            # SciPy 1.14 要求 NumPy >= 1.22.4；且许多轮子对 2.3.* 支持滞后
            if v_np >= Version("2.3.0"):
                notes.append("检测到 NumPy >= 2.3；很多 SciPy 轮子尚未完全兼容，建议降到 2.2.*")
        except Exception:
            pass

    return target, notes

def pip_exe():
    # 与当前解释器绑定的 pip
    return f"{shutil.which(sys.executable)} -m pip" if shutil.which(sys.executable) else "python -m pip"

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    print_header("环境体检报告")

    print(f"Python: {sys.version.splitlines()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Interpreter: {sys.executable}")
    print(f"Virtualenv: {'YES' if is_venv() else 'NO'}")

    print_header("sys.path（用于判定是否混入其它 site-packages）")
    for p in sys.path:
        print(" -", p)

    mixed = detect_mixed_sites()
    if mixed:
        print("\n[警告] 发现可能的混用路径（不属于当前解释器的 site-packages）：")
        for p in mixed:
            print(" *", p)
        print("建议：优先使用 venv，并确保仅 .venv/... 出现在 sys.path 中。")

    print_header("包版本与安装路径")
    modules = ["numpy", "scipy", "pandas", "torch"]
    results = {}
    for name in modules:
        mod, err = safe_import(name)
        if mod:
            results[name] = {
                "version": version_of(mod),
                "path": path_of(mod),
                "error": None,
            }
        else:
            results[name] = {
                "version": None,
                "path": None,
                "error": err,
            }

    for name in modules:
        info = results[name]
        if info["error"] is None:
            print(f"{name:<8} version: {info['version']:<10} path: {info['path']}")
        else:
            print(f"{name:<8} 未安装或导入失败：{repr(info['error'])}")

    # 兼容性提示
    print_header("兼容性检查与建议")
    np_ver = results["numpy"]["version"]
    sp_ver = results["scipy"]["version"]
    th_ver = results["torch"]["version"]

    target, notes = recommend_versions(np_ver, sp_ver, th_ver)

    # 规则：如果 SciPy 报错 “requires NumPy >=1.22.4 and <2.3.0”，建议 NumPy 降到 2.2.*
    # 这里无法读你的运行时警告文本，改为按版本逻辑给建议。
    problems = []
    try:
        from packaging.version import Version, InvalidVersion
        if np_ver:
            v_np = Version(np_ver)
            if v_np >= Version("2.3.0"):
                problems.append("当前 NumPy >= 2.3.0；建议降到 2.2.* 以匹配 SciPy 1.14.* 及多数现有轮子。")
        if th_ver and np_ver:
            # 针对 PyTorch 的 numpy 依赖不做严格范围判断，仅给出常见建议
            pass
    except Exception:
        pass

    if problems:
        print("[问题]")
        for p in problems:
            print(" -", p)

    if notes:
        print("\n[说明]")
        for n in notes:
            print(" -", n)

    print("\n[推荐稳定组合]")
    for k, v in target.items():
        print(f" - {k}: {v}")

    # 生成修复命令（两套：venv 方案 + 当前解释器直装方案）
    print_header("一键修复命令（方案 A：使用 venv，推荐）")
    project_root = os.getcwd()
    cmds = [
        f"cd {project_root}",
        f"{sys.executable} -m venv .venv",
        "source .venv/bin/activate",
        "python -m pip install -U pip setuptools wheel",
        f'pip install "numpy=={target["numpy"]}" "scipy=={target["scipy"]}" "pandas=={target["pandas"]}"',
        # PyTorch CPU 轮子（更稳）：如需 GPU/Metal 自行调整官方指引
        f'pip install "torch=={target["torch"]}" --index-url https://download.pytorch.org/whl/cpu',
        "python - <<'PY'\nimport sys, numpy, scipy, torch, pandas as pd\n"
        "print('OK Python:', sys.version)\n"
        "print('OK NumPy :', numpy.__version__, numpy.__file__)\n"
        "print('OK SciPy :', scipy.__version__)\n"
        "print('OK Pandas:', pd.__version__)\n"
        "print('OK Torch :', torch.__version__)\nPY"
    ]
    print(indent("\n".join(cmds), "  "))

    print_header("一键修复命令（方案 B：不建 venv，当前解释器直装，不推荐）")
    px = f"{sys.executable} -m pip"
    cmds_b = [
        f"{px} install -U pip setuptools wheel",
        f'{px} install --upgrade --force-reinstall "numpy=={target["numpy"]}"',
        f'{px} install --upgrade --force-reinstall "scipy=={target["scipy"]}" "pandas=={target["pandas"]}"',
        f'{px} install --upgrade --force-reinstall "torch=={target["torch"]}" --index-url https://download.pytorch.org/whl/cpu',
        "python - <<'PY'\nimport sys, numpy, scipy, torch, pandas as pd\n"
        "print('OK Python:', sys.version)\n"
        "print('OK NumPy :', numpy.__version__, numpy.__file__)\n"
        "print('OK SciPy :', scipy.__version__)\n"
        "print('OK Pandas:', pd.__version__)\n"
        "print('OK Torch :', torch.__version__)\nPY"
    ]
    print(indent("\n".join(cmds_b), "  "))

    print_header("附加建议")
    print("- 若你有用到旧的 pickle（例如依赖 numpy._core），保持 NumPy 2.2.* 可读，读出后建议改存 Parquet：")
    print(indent("df.to_parquet('data.parquet'); pd.read_parquet('data.parquet')", "  "))
    print("- 运行训练/推理脚本前务必先 `source .venv/bin/activate`，避免混入全局包。")
    print("- 如果你在 Apple Silicon 上需要 Metal 加速的 PyTorch，请参考官方安装指引替换上面的 torch 安装命令。")

if __name__ == "__main__":
    main()