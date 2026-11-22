"""Build a self-contained Kaggle script: embeds lta/*.py (base64) + a runner body + baked-in config.
usage: python tools/build_kernel.py <runner.py> <out_dir> <kernel_slug> <owner> key=value ...
"""
import base64, json, os, sys, glob

runner, out_dir, slug, owner = sys.argv[1:5]
cfg = dict(a.split("=", 1) for a in sys.argv[5:])
files = {os.path.basename(p): base64.b64encode(open(p, "rb").read()).decode() for p in sorted(glob.glob("lta/*.py"))}
head = (
    "import os, sys, base64, json\n"
    f"CFG = json.loads({json.dumps(json.dumps(cfg))})\n"
    f"_FILES = json.loads({json.dumps(json.dumps(files))})\n"
    "_W = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('./kout')\n"
    "os.makedirs(os.path.join(_W, 'lta'), exist_ok=True)\n"
    "for _n, _b in _FILES.items():\n"
    "    open(os.path.join(_W, 'lta', _n), 'wb').write(base64.b64decode(_b))\n"
    "sys.path.insert(0, _W)\n"
)
os.makedirs(out_dir, exist_ok=True)
open(os.path.join(out_dir, "script.py"), "w").write(head + open(runner).read())
meta = {"id": f"{owner}/{slug}", "title": slug, "code_file": "script.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
json.dump(meta, open(os.path.join(out_dir, "kernel-metadata.json"), "w"))
print("built", out_dir, "cfg", cfg, "files", list(files))
