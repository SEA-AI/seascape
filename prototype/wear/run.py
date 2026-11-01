"""usage: python run.py scenario.toml outdir [worn] [name=amount ...]"""
import json, resource, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import seascape.render as R
import importlib, os
weather = importlib.import_module(os.environ.get("WEAR", "weather"))
from seascape.config import load

scenario_path, out = sys.argv[1], Path(sys.argv[2])
worn = len(sys.argv) > 3 and sys.argv[3] == "worn"
amounts = {k: float(v) for k, v in (a.split("=") for a in sys.argv[4:])}
scenario = load(scenario_path)
if worn:
    weather.install(scenario, amounts)

renders = []
original = R._render_still
def timed(path, what):
    t0 = time.perf_counter(); original(path, what); renders.append((path.name, round(time.perf_counter() - t0, 1)))
R._render_still = timed

t0 = time.perf_counter()
R.render(scenario, out)
report = {"worn": worn, "total_s": round(time.perf_counter() - t0, 1), "renders": renders,
          "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
          **{k: round(v, 2) for k, v in weather.timings.items()}}
print(json.dumps(report))
(out / "timing.json").write_text(json.dumps(report, indent=1))
