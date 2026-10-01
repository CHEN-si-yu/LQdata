"""只按真实 Python 入口和工作目录识别上游进程，不把命令参数中的引用误当成写入任务。"""
from pathlib import Path

def _entry(argv):
    if not argv:return None
    exe=Path(argv[0]).name.lower()
    if exe.endswith(".py"):return ("file",argv[0])
    if not exe.startswith(("python","pypy")):return None
    i=1
    while i<len(argv):
        arg=argv[i]
        if arg=="-c":return None
        if arg=="-m":return ("module",argv[i+1]) if i+1<len(argv) else None
        if arg in ("-W","-X"):i+=2;continue
        if arg=="--":return ("file",argv[i+1]) if i+1<len(argv) else None
        if arg.startswith("-"):i+=1;continue
        return ("file",arg)
    return None

def upstream_writer_pids(proc_root=None,upstream_root=None):
    proc_root=Path("/proc") if proc_root is None else Path(proc_root)
    root=(Path(__file__).resolve().parents[2]/"everyday_tasks" if upstream_root is None else Path(upstream_root)).resolve()
    target=root/"main.py";busy=[]
    for process in proc_root.iterdir():
        if not process.name.isdigit():continue
        try:
            argv=[a.decode("utf-8",errors="replace") for a in (process/"cmdline").read_bytes().split(b"\0") if a]
            entry=_entry(argv)
            if entry is None:continue
            cwd=(process/"cwd").resolve()
            kind,value=entry
            same=(cwd/value).resolve()==target if kind=="file" else (
                value=="everyday_tasks.main" and cwd==root.parent or value=="main" and cwd==root)
            if same:busy.append(int(process.name))
        except OSError:continue
    return sorted(busy)
