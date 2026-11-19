"""Round-3 heterogeneous MTL (Fine+Rotation / Fine+Color) analysis.

Discovers runs/<scenario>/<method>/seed_<n>/ and prints:
  1. per-scenario test table + gradient-conflict summary per method
  2. paired per-seed delta vs equal and single-task gaps (within scenario)
  3. Equal's gradient cosine / conflict-rate trajectory + raw norm ratio
  4. weight trajectories and GradNorm rate/tracking diagnostics
  5. aux-task decodability vs chance and best-epoch audit
Saves results/<scenario>/analysis.json.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE_EPS = [1, 5, 10, 15, 20, 25, 30]
CHANCE = {"rotation": 25.0, "color": 25.0}


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def signs(xs):
    return "".join("+" if x > 1e-9 else ("-" if x < -1e-9 else "0") for x in xs)


def fnum(x):
    return None if x in ("", None) else float(x)


def load_run(d):
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    with (d / "metrics.csv").open(encoding="utf-8") as f:
        rows = [{k: fnum(v) for k, v in r.items()} for r in csv.DictReader(f)]
    return res, rows


def discover():
    out = {}
    for scen in ("color", "rotation"):
        base = ROOT / "runs" / scen
        if not base.is_dir():
            continue
        for meth in sorted(p for p in base.iterdir() if p.is_dir()):
            for sd in sorted(meth.glob("seed_*")):
                if (sd / "results.json").is_file():
                    out[(scen, meth.name, int(sd.name.split("_")[1]))] = load_run(sd)
    return out


def main():
    runs = discover()
    scenarios = sorted({k[0] for k in runs})
    for scen in scenarios:
        methods = sorted({k[1] for k in runs if k[0] == scen})
        seeds = sorted({k[2] for k in runs if k[0] == scen})
        print("=" * 108)
        print(f"SCENARIO {scen}  methods={methods} seeds={seeds}")
        print(f"{'method':<12}{'fine':>14}{'aux':>14}{'mean':>14}"
              f"{'cos_mean':>10}{'conflict':>10}{'w_fine_end':>12}")
        table = {}
        for m in methods:
            rs = [runs[(scen, m, s)][0] for s in seeds if (scen, m, s) in runs]
            row = {}
            for key, lab in [("test_fine_acc_pct", "fine"),
                             ("test_aux_acc_pct", "aux"),
                             ("test_mean_acc_pct", "mean")]:
                vals = [r[key] for r in rs if r.get(key) is not None]
                row[lab] = dict(mean=mean(vals) if vals else None,
                                std=std(vals) if vals else None)
            cos = [r["grad_cosine_all_epochs_mean"] for r in rs
                   if r.get("grad_cosine_all_epochs_mean") is not None]
            conf = [r["grad_conflict_rate_all_epochs"] for r in rs
                    if r.get("grad_conflict_rate_all_epochs") is not None]
            wf = [r["final_weight_fine"] for r in rs if r.get("final_weight_fine") is not None]
            row["cos_mean"] = mean(cos) if cos else None
            row["conflict"] = mean(conf) if conf else None
            row["w_fine_end"] = mean(wf) if wf else None
            table[m] = row

            def cell(v, nd=2):
                return "     --     " if v is None else f"{v:7.{nd}f}"
            print(f"{m:<12}"
                  + "".join(f"{cell(row[l]['mean']):>8}±{row[l]['std']:.2f}"
                            if row[l]["mean"] is not None else f"{'--':>14}"
                            for l in ["fine", "aux", "mean"])
                  + f"{cell(row['cos_mean'], 3):>10}{cell(row['conflict'], 3):>10}"
                  + f"{cell(row['w_fine_end'], 3):>12}")
        out = {"table": table}

        if "equal" in methods:
            print("\n  paired delta vs equal (pp, per seed):")
            deltas = {}
            for m in methods:
                if m == "equal":
                    continue
                for lab, key in [("fine", "test_fine_acc_pct"),
                                 ("aux", "test_aux_acc_pct"),
                                 ("mean", "test_mean_acc_pct")]:
                    d = [runs[(scen, m, s)][0][key] - runs[(scen, "equal", s)][0][key]
                         for s in seeds
                         if (scen, m, s) in runs and (scen, "equal", s) in runs
                         and runs[(scen, m, s)][0].get(key) is not None
                         and runs[(scen, "equal", s)][0].get(key) is not None]
                    if not d:
                        continue
                    deltas[f"{m}/{lab}"] = d
                    print(f"    {m:<12}{lab:<5} {['%+.2f' % x for x in d]}  "
                          f"mean {mean(d):+.3f} std {std(d):.3f} signs {signs(d)}")
            out["delta_vs_equal_pp"] = deltas

            print("\n  multi-task minus single-task (within scenario, pp):")
            gaps = {}
            for m in methods:
                if m in ("equal", "fine_only", "aux_only"):
                    continue
                for lab, key, ref in [("fine", "test_fine_acc_pct", "fine_only"),
                                      ("aux", "test_aux_acc_pct", "aux_only")]:
                    d = [runs[(scen, m, s)][0][key] - runs[(scen, ref, s)][0][key]
                         for s in seeds
                         if (scen, m, s) in runs and (scen, ref, s) in runs
                         and runs[(scen, m, s)][0].get(key) is not None]
                    if not d:
                        continue
                    gaps[f"{m}/{lab}"] = d
                    print(f"    {m:<12}{lab:<5} mean {mean(d):+.3f} std {std(d):.3f} signs {signs(d)}")
            out["single_task_gaps_pp"] = gaps

        print("\n  equal: grad cosine / conflict rate / raw norm ratio aux/fine:")
        traj = {}
        for e in PROBE_EPS:
            cs, cf, nr = [], [], []
            for s in seeds:
                if (scen, "equal", s) not in runs:
                    continue
                rows = runs[(scen, "equal", s)][1]
                if len(rows) < e:
                    continue
                r = rows[e - 1]
                if r["grad_cosine"] is not None:
                    cs.append(r["grad_cosine"])
                    cf.append(r["grad_conflict_rate"])
                if r["raw_grad_norm_fine"] and r["raw_grad_norm_aux"] is not None:
                    nr.append(r["raw_grad_norm_aux"] / r["raw_grad_norm_fine"])
            if cs:
                traj[str(e)] = dict(cos=mean(cs), conflict=mean(cf),
                                    norm_ratio=mean(nr) if nr else None)
                print(f"    ep{e:<3} cos {mean(cs):+.3f}  conflict {mean(cf)*100:5.1f}%  "
                      f"|g_aux|/|g_fine| {mean(nr) if nr else float('nan'):.3f}")
        neg = []
        for s in seeds:
            if (scen, "equal", s) in runs:
                cs = [r["grad_cosine"] for r in runs[(scen, "equal", s)][1]
                      if r["grad_cosine"] is not None]
                neg.append(sum(1 for c in cs if c < 0) / len(cs))
        if neg:
            print(f"    fraction of epochs with mean cos<0: {mean(neg)*100:.1f}%")
            traj["frac_epochs_neg_cos"] = mean(neg)
        out["equal_gradient_trajectory"] = traj

        print("\n  weight_fine trajectory (mean over seeds):")
        wt = {}
        print(f"{'method':<12}" + "".join(f"ep{e:<6}" for e in PROBE_EPS))
        for m in methods:
            if m in ("fine_only", "aux_only"):
                continue
            vals = []
            for e in PROBE_EPS:
                ws = [runs[(scen, m, s)][1][e - 1]["weight_fine"]
                      for s in seeds if (scen, m, s) in runs
                      and len(runs[(scen, m, s)][1]) >= e
                      and runs[(scen, m, s)][1][e - 1]["weight_fine"] is not None]
                vals.append(mean(ws) if ws else float("nan"))
            wt[m] = dict(zip(map(str, PROBE_EPS), vals))
            print(f"{m:<12}" + "".join(f"{v:<8.3f}" for v in vals))
        out["weight_fine_trajectory"] = wt

        if "gradnorm" in methods:
            print("\n  gradnorm r_fine / r_aux and rel. tracking error:")
            gn = {}
            for e in PROBE_EPS:
                rf, ra, ef, ea = [], [], [], []
                for s in seeds:
                    if (scen, "gradnorm", s) not in runs:
                        continue
                    rows = runs[(scen, "gradnorm", s)][1]
                    if len(rows) < e or rows[e - 1]["gradnorm_r_fine"] is None:
                        continue
                    r = rows[e - 1]
                    rf.append(r["gradnorm_r_fine"])
                    ra.append(r["gradnorm_r_aux"])
                    ef.append(abs(r["gradnorm_G_fine"] - r["gradnorm_target_fine"])
                              / r["gradnorm_target_fine"])
                    ea.append(abs(r["gradnorm_G_aux"] - r["gradnorm_target_aux"])
                              / r["gradnorm_target_aux"])
                if rf:
                    gn[str(e)] = dict(r_fine=mean(rf), r_aux=mean(ra),
                                      err_fine=mean(ef), err_aux=mean(ea))
                    print(f"    ep{e:<3} r_f={mean(rf):.3f} r_a={mean(ra):.3f}  "
                          f"err_f={mean(ef)*100:5.2f}% err_a={mean(ea)*100:5.2f}%")
            out["gradnorm_diagnostics"] = gn

        print("\n  aux decodability and best-epoch audit:")
        audit = {}
        if "aux_only" in methods:
            aux = [runs[(scen, "aux_only", s)][0]["test_aux_acc_pct"] for s in seeds
                   if (scen, "aux_only", s) in runs]
            print(f"    aux_only test aux = {mean(aux):.2f}%  (chance {CHANCE.get(scen, 25):.0f}%)")
            out["aux_only_vs_chance"] = dict(mean=mean(aux), chance=CHANCE.get(scen, 25))
        for m in methods:
            for s in seeds:
                if (scen, m, s) not in runs:
                    continue
                res, rows = runs[(scen, m, s)]
                vals = [(r["epoch"], r["val_checkpoint_score"]) for r in rows
                        if r["val_checkpoint_score"] is not None]
                best = res["best_epoch"]
                bv = res["best_validation_score_pct"]
                second = max(v for e, v in vals if e != best)
                audit[f"{m}/s{s}"] = dict(best_epoch=best, gap_to_second_pp=bv - second)
        late = [v["gap_to_second_pp"] for v in audit.values()]
        eps = [v["best_epoch"] for v in audit.values()]
        print(f"    best_epoch range {min(eps)}-{max(eps)}; gap-to-2nd mean {mean(late):.3f}pp "
              f"max {max(late):.3f}pp")
        out["best_epoch_audit"] = audit

        dest = ROOT / "results" / scen / "analysis.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved {dest}\n")


if __name__ == "__main__":
    main()
