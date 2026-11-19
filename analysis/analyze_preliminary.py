"""Preliminary multi-task experiment analysis (5 methods x 3 seeds).

Reads runs/clean/<method>/seed_<n>/{results.json,metrics.csv} and prints:
  1. per-seed test results + best-epoch audit
  2. paired per-seed differences vs equal / vs single-task
  3. weight trajectories and GradNorm tracking diagnostics
Outputs results/clean/preliminary_analysis.json for downstream report use.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["fine_only", "coarse_only", "equal", "uncertainty", "gradnorm"]
SEEDS = [0, 1, 2]


def load_runs():
    runs = {}
    for m in METHODS:
        for s in SEEDS:
            d = ROOT / "runs" / "clean" / m / f"seed_{s}"
            res = json.loads((d / "results.json").read_text(encoding="utf-8"))
            with (d / "metrics.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            rows = [{k: (float(v) if v not in ("", None) else None)
                     for k, v in r.items()} for r in rows]
            runs[(m, s)] = {"res": res, "rows": rows}
    return runs


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def fmt(x, nd=2):
    return "  --  " if x is None else f"{x:.{nd}f}"


def main():
    runs = load_runs()
    out = {}

    print("=" * 100)
    print("1. PER-SEED TEST RESULTS  (test = 10000 images, evaluated at best-val epoch)")
    print(f"{'method':<12}{'seed':>4} | {'fine':>7}{'coarse':>8}{'mean':>8} | "
          f"{'best_ep':>7}{'val_best':>9}{'val_final':>10}{'ep_gap_pp':>10}{'n_within0.1':>12}")
    audit = {}
    for m in METHODS:
        for s in SEEDS:
            r = runs[(m, s)]["res"]
            rows = runs[(m, s)]["rows"]
            val_key = {"fine_only": "val_fine_acc",
                       "coarse_only": "val_coarse_acc"}.get(m, "val_mean_acc")
            vals = [(row["epoch"], row[val_key]) for row in rows
                    if row[val_key] is not None]
            best_ep = r["best_epoch"]
            best_val = r["best_validation_score_pct"]
            final_val = vals[-1][1]
            second = max(v for e, v in vals if e != best_ep)
            gap = best_val - second
            n_close = sum(1 for _, v in vals if best_val - v <= 0.1)
            audit[f"{m}/s{s}"] = dict(best_epoch=best_ep, best_val=best_val,
                                      final_val=final_val, gap_to_second_pp=gap,
                                      n_epochs_within_0p1pp=n_close)
            print(f"{m:<12}{s:>4} | {fmt(r.get('test_fine_acc_pct')):>7}"
                  f"{fmt(r.get('test_coarse_acc_pct')):>8}{fmt(r.get('test_mean_acc_pct')):>8} | "
                  f"{best_ep:>7}{best_val:>9.2f}{final_val:>10.2f}{gap:>10.3f}{n_close:>12}")
    out["best_epoch_audit"] = audit

    print()
    print("=" * 100)
    print("2. PAIRED PER-SEED DIFFERENCES (pp)")
    paired = {}
    for m in ["uncertainty", "gradnorm"]:
        print(f"\n  {m} - equal:")
        for key, label in [("test_fine_acc_pct", "fine "),
                           ("test_coarse_acc_pct", "coarse"),
                           ("test_mean_acc_pct", "mean ")]:
            diffs = [runs[(m, s)]["res"][key] - runs[("equal", s)]["res"][key] for s in SEEDS]
            paired[f"{m}-equal/{label.strip()}"] = diffs
            print(f"    {label}: per-seed {['%+.2f' % d for d in diffs]}  "
                  f"mean {mean(diffs):+.3f}  std {std(diffs):.3f}  "
                  f"signs {''.join('+' if d > 0 else '-' for d in diffs)}")
    print("\n  multi-task minus single-task (negative = multi-task hurts):")
    for m in ["equal", "uncertainty", "gradnorm"]:
        df = [runs[(m, s)]["res"]["test_fine_acc_pct"]
              - runs[("fine_only", s)]["res"]["test_fine_acc_pct"] for s in SEEDS]
        dc = [runs[(m, s)]["res"]["test_coarse_acc_pct"]
              - runs[("coarse_only", s)]["res"]["test_coarse_acc_pct"] for s in SEEDS]
        paired[f"{m}-single/fine"] = df
        paired[f"{m}-single/coarse"] = dc
        print(f"    {m:<12} fine {mean(df):+.3f} (std {std(df):.3f}, "
              f"{''.join('+' if d > 0 else '-' for d in df)})   "
              f"coarse {mean(dc):+.3f} (std {std(dc):.3f}, "
              f"{''.join('+' if d > 0 else '-' for d in dc)})")
    out["paired_diffs_pp"] = paired

    print()
    print("=" * 100)
    print("3. WEIGHT TRAJECTORIES (weight_fine, averaged over seeds; sum of weights = 2)")
    traj = {}
    probe_eps = [1, 5, 10, 15, 20, 25, 30]
    print(f"{'method':<12}" + "".join(f"ep{e:<6}" for e in probe_eps))
    for m in ["equal", "uncertainty", "gradnorm"]:
        vals = []
        for e in probe_eps:
            ws = [runs[(m, s)]["rows"][e - 1]["weight_fine"] for s in SEEDS]
            vals.append(mean(ws))
        traj[m] = dict(zip([str(e) for e in probe_eps], vals))
        print(f"{m:<12}" + "".join(f"{v:<8.3f}" for v in vals))
    out["weight_fine_trajectory"] = traj

    print("\n  uncertainty: implied s_fine = -log(w_fine) at ep30, per seed:")
    for s in SEEDS:
        w = runs[("uncertainty", s)]["rows"][-1]["weight_fine"]
        print(f"    seed {s}: w_fine={w:.4f}  s_fine={-math.log(w):+.4f}")

    print("\n  gradnorm tracking error |G-target|/target (mean over seeds):")
    track = {}
    for e in probe_eps:
        errs_f, errs_c, rf = [], [], []
        for s in SEEDS:
            row = runs[("gradnorm", s)]["rows"][e - 1]
            errs_f.append(abs(row["gradnorm_G_fine"] - row["gradnorm_target_fine"])
                          / row["gradnorm_target_fine"])
            errs_c.append(abs(row["gradnorm_G_coarse"] - row["gradnorm_target_coarse"])
                          / row["gradnorm_target_coarse"])
            rf.append(row["gradnorm_r_fine"])
        track[str(e)] = dict(rel_err_fine=mean(errs_f), rel_err_coarse=mean(errs_c),
                             r_fine=mean(rf))
        print(f"    ep{e:<3} fine {mean(errs_f)*100:6.2f}%   coarse {mean(errs_c)*100:6.2f}%   "
              f"r_fine {mean(rf):.3f}")
    out["gradnorm_tracking"] = track

    print("\n  late-training val plateau (epochs 25-30 val_mean_acc, per run):")
    plateau = {}
    for m in METHODS:
        for s in SEEDS:
            rows = runs[(m, s)]["rows"]
            val_key = {"fine_only": "val_fine_acc",
                       "coarse_only": "val_coarse_acc"}.get(m, "val_mean_acc")
            late = [r[val_key] for r in rows if r["epoch"] >= 25
                    and r[val_key] is not None]
            if late:
                plateau[f"{m}/s{s}"] = dict(min=min(late), max=max(late),
                                            spread_pp=max(late) - min(late))
    spreads = [v["spread_pp"] for v in plateau.values()]
    print(f"    spread max-min: mean {mean(spreads):.3f} pp, "
          f"range {min(spreads):.3f}-{max(spreads):.3f} pp")
    out["late_val_plateau"] = plateau

    print("\n  final weight_fine vs test fine acc (9 MTL runs):")
    pts = [(runs[(m, s)]["res"]["final_weight_fine"],
            runs[(m, s)]["res"]["test_fine_acc_pct"])
           for m in ["equal", "uncertainty", "gradnorm"] for s in SEEDS]
    xs, ys = zip(*pts)
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in pts)
    corr = cov / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    print(f"    Pearson r = {corr:+.3f}")
    out["weight_vs_fine_acc_pearson_r"] = corr

    dest = ROOT / "results" / "clean" / "preliminary_analysis.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
