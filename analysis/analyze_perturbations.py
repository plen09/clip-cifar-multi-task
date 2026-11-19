"""Round-2 perturbation experiment analysis.

Discovers runs/perturbations/<scenario>/<method>/seed_<n>/ automatically (so optional
single-task baselines are picked up too) and prints:
  1. per-scenario test tables + paired delta vs equal (per seed, signs)
  2. perturbation cost vs the clean round-1 runs/ (same method, same seed)
  3. weight trajectories per scenario, with the clean round as reference
  4. GradNorm relative-rate / tracking diagnostics under label noise
  5. best-epoch audit per scenario
Saves results/perturbations/analysis.json.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "runs" / "clean"
PERT = ROOT / "runs/perturbations"
PROBE_EPS = [1, 5, 10, 15, 20, 25, 30]


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def signs(xs):
    return "".join("+" if x > 1e-9 else ("-" if x < -1e-9 else "0") for x in xs)


def load_run(d):
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    with (d / "metrics.csv").open(encoding="utf-8") as f:
        rows = [{k: (float(v) if v not in ("", None) else None)
                 for k, v in r.items()} for r in csv.DictReader(f)]
    return res, rows


def discover(base):
    out = {}
    if not base.is_dir():
        return out
    for scen in sorted(p for p in base.iterdir() if p.is_dir()):
        for meth in sorted(p for p in scen.iterdir() if p.is_dir()):
            for seed_dir in sorted(meth.glob("seed_*")):
                if (seed_dir / "results.json").is_file():
                    s = int(seed_dir.name.split("_")[1])
                    out[(scen.name, meth.name, s)] = load_run(seed_dir)
    return out


def discover_clean(base):
    """Clean layout below runs/clean is <method>/seed_<n>."""
    out = {}
    for meth in sorted(p for p in base.iterdir() if p.is_dir()):
        for seed_dir in sorted(meth.glob("seed_*")):
            if (seed_dir / "results.json").is_file():
                s = int(seed_dir.name.split("_")[1])
                out[(meth.name, s)] = load_run(seed_dir)
    return out


def val_key(m):
    return {"fine_only": "val_fine_acc",
            "coarse_only": "val_coarse_acc"}.get(m, "val_mean_acc")


def main():
    pert = discover(PERT)
    clean = discover_clean(CLEAN)
    scenarios = sorted({k[0] for k in pert})
    out = {"scenarios": {}}

    for scen in scenarios:
        methods = sorted({k[1] for k in pert if k[0] == scen})
        seeds = sorted({k[2] for k in pert if k[0] == scen})
        print("=" * 104)
        print(f"SCENARIO {scen}   methods={methods}  seeds={seeds}")
        print(f"{'method':<12}" + "".join(f"{'fine':>8}{'coarse':>9}{'mean':>8}" for _ in [0])
              + "   (test mean +/- std over seeds)")
        table = {}
        for m in methods:
            rs = [pert[(scen, m, s)][0] for s in seeds if (scen, m, s) in pert]
            row = {}
            for key, lab in [("test_fine_acc_pct", "fine"),
                             ("test_coarse_acc_pct", "coarse"),
                             ("test_mean_acc_pct", "mean")]:
                vals = [r[key] for r in rs if r.get(key) is not None]
                row[lab] = dict(mean=mean(vals) if vals else None,
                                std=std(vals) if vals else None,
                                per_seed=[r.get(key) for r in rs])
            table[m] = row
            cells = []
            for lab in ["fine", "coarse", "mean"]:
                v = row[lab]
                cells.append("   --   " if v["mean"] is None
                             else f"{v['mean']:7.2f}±{v['std']:.2f}")
            print(f"{m:<12}" + "".join(f"{c:>17}" for c in cells))
        out["scenarios"][scen] = {"table": table}

        if "equal" in methods:
            print("\n  paired delta vs equal (pp, per seed):")
            deltas = {}
            for m in methods:
                if m == "equal":
                    continue
                for lab, key in [("fine", "test_fine_acc_pct"),
                                 ("coarse", "test_coarse_acc_pct"),
                                 ("mean", "test_mean_acc_pct")]:
                    pairs = [(pert[(scen, m, s)][0].get(key),
                              pert[(scen, "equal", s)][0].get(key))
                             for s in seeds
                             if (scen, m, s) in pert and (scen, "equal", s) in pert]
                    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                    if not pairs:
                        continue
                    d = [a - b for a, b in pairs]
                    deltas[f"{m}/{lab}"] = d
                    print(f"    {m:<12}{lab:<7} {['%+.2f' % x for x in d]}  "
                          f"mean {mean(d):+.3f} std {std(d):.3f} signs {signs(d)}")
            out["scenarios"][scen]["delta_vs_equal_pp"] = deltas

        print("\n  perturbation cost vs clean round-1 (same method/seed, pp):")
        costs = {}
        for m in methods:
            for lab, key in [("fine", "test_fine_acc_pct"),
                             ("coarse", "test_coarse_acc_pct")]:
                pairs = [(pert[(scen, m, s)][0].get(key), clean[(m, s)][0].get(key))
                         for s in seeds
                         if (scen, m, s) in pert and (m, s) in clean]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                if not pairs:
                    continue
                d = [a - b for a, b in pairs]
                costs[f"{m}/{lab}"] = d
                print(f"    {m:<12}{lab:<7} mean {mean(d):+.3f} std {std(d):.3f} signs {signs(d)}")
        out["scenarios"][scen]["cost_vs_clean_pp"] = costs

        print("\n  weight_fine trajectory (mean over seeds):")
        traj = {}
        header = f"{'method':<12}" + "".join(f"ep{e:<6}" for e in PROBE_EPS)
        print(header)
        for m in methods:
            if m in ("fine_only", "coarse_only"):
                continue
            vals = []
            for e in PROBE_EPS:
                ws = [pert[(scen, m, s)][1][e - 1]["weight_fine"]
                      for s in seeds if (scen, m, s) in pert
                      and len(pert[(scen, m, s)][1]) >= e]
                vals.append(mean(ws) if ws else float("nan"))
            traj[m] = dict(zip(map(str, PROBE_EPS), vals))
            print(f"{m:<12}" + "".join(f"{v:<8.3f}" for v in vals))
        if ("equal", 0) in clean:
            vals = [mean([clean[("equal", s)][1][e - 1]["weight_fine"] for s in range(3)])
                    for e in PROBE_EPS]
            print(f"{'clean ref':<12}" + "".join(f"{v:<8.3f}" for v in vals))
        out["scenarios"][scen]["weight_fine_trajectory"] = traj

        if "gradnorm" in methods:
            print("\n  gradnorm r_fine / r_coarse and rel. tracking error (mean over seeds):")
            gn = {}
            for e in PROBE_EPS:
                rf, rc, ef, ec = [], [], [], []
                for s in seeds:
                    if (scen, "gradnorm", s) not in pert:
                        continue
                    row = pert[(scen, "gradnorm", s)][1][e - 1]
                    if row.get("gradnorm_r_fine") is None:
                        continue
                    rf.append(row["gradnorm_r_fine"])
                    rc.append(row["gradnorm_r_coarse"])
                    ef.append(abs(row["gradnorm_G_fine"] - row["gradnorm_target_fine"])
                              / row["gradnorm_target_fine"])
                    ec.append(abs(row["gradnorm_G_coarse"] - row["gradnorm_target_coarse"])
                              / row["gradnorm_target_coarse"])
                if rf:
                    gn[str(e)] = dict(r_fine=mean(rf), r_coarse=mean(rc),
                                      rel_err_fine=mean(ef), rel_err_coarse=mean(ec))
                    print(f"    ep{e:<3} r_f={mean(rf):.3f} r_c={mean(rc):.3f}   "
                          f"err_f={mean(ef)*100:5.2f}% err_c={mean(ec)*100:5.2f}%")
            out["scenarios"][scen]["gradnorm_diagnostics"] = gn

        print("\n  best-epoch audit:")
        audit = {}
        for m in methods:
            for s in seeds:
                if (scen, m, s) not in pert:
                    continue
                res, rows = pert[(scen, m, s)]
                vk = val_key(m)
                vals = [(r["epoch"], r[vk]) for r in rows if r[vk] is not None]
                best = res["best_epoch"]
                bv = res["best_validation_score_pct"]
                second = max(v for e, v in vals if e != best)
                audit[f"{m}/s{s}"] = dict(best_epoch=best, gap_to_second_pp=bv - second)
                print(f"    {m:<12}s{s}  best_ep={best:>2}  gap_to_2nd={bv - second:.3f}pp")
        out["scenarios"][scen]["best_epoch_audit"] = audit
        print()

    dest = ROOT / "results/perturbations" / "analysis.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
