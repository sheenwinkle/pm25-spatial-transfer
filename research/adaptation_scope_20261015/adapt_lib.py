"""Shared code for the adaptation-scope and label-coverage experiments (see PROTOCOL.md)."""
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
BENCH = Path(os.environ["BENCH_DIR"])           # benchmark_experiment folder (scripts)
sys.path.insert(0, str(BENCH))


def _torch_stub():
    """bench_harness imports torch for the LDF autoencoder, which these experiments never call. Where torch is
    not installed, a minimal stand-in lets the module load; any LDF call would fail loudly."""
    torch = types.ModuleType("torch")
    torch.set_num_threads = lambda n: None
    torch.zeros = lambda *a, **k: None
    optim = types.ModuleType("torch.optim"); optim.NAdam = lambda *a, **k: None
    nn = types.ModuleType("torch.nn"); nn.Module = object
    utils = types.ModuleType("torch.utils"); data = types.ModuleType("torch.utils.data")
    data.DataLoader = data.Dataset = object
    utils.data = data
    dynamo = types.ModuleType("torch._dynamo")
    torch.optim, torch.nn, torch.utils, torch._dynamo = optim, nn, utils, dynamo
    sys.modules.update({"torch": torch, "torch.optim": optim, "torch.nn": nn, "torch.utils": utils,
                        "torch.utils.data": data, "torch._dynamo": dynamo})


try:
    import torch  # noqa: F401
except ImportError:
    _torch_stub()

import bench_harness as bh                      # noqa: E402  (load_task, benchmark_splits, core, score)
from bench_harness import X, Y, core            # noqa: E402

PANEL = Path(os.environ["BENCH_PANEL"])
MEMBER = Path(os.environ["BENCH_MEMBERSHIP"])
REGIONS = json.loads((HERE / "regions.json").read_text())
LOCKED = REGIONS["locked_region"]
FUA_REGION = {int(k): v for k, v in REGIONS["fua_region"].items()}

COND = ["monthly_power_temperature_c", "temp_range", "monthly_power_humidity_pct", "monthly_power_wind_mps",
        "monthly_power_precip_mm_day", "log_fire_count_100km", "local_season_sin", "local_season_cos"]
SEED = 20261015


def panel():
    p = pd.read_csv(PANEL)
    p["temp_range"] = p.monthly_power_temperature_max_c - p.monthly_power_temperature_min_c
    return p


_P = panel()
_C = _P[_P.country_iso.eq("CHN")]
Z_MU, Z_SD = _C[COND].mean(), _C[COND].std()


def zmat(df):
    return ((df[COND] - Z_MU) / Z_SD).to_numpy(float)


def yrd_ids():
    return set(pd.read_csv(MEMBER).query("country_iso == 'CHN'").eFUA_ID)


def eligible_fuas(tgt, share):
    cnt = tgt.groupby("eFUA_ID").size()
    return np.array(sorted(cnt[cnt >= np.ceil(share * _P.t.nunique())].index))


def load(task):
    """Return (source rows, target rows of eligible FUAs, eligible FUA ids) with temp_range added."""
    p = _P
    if task in ("yrd", "aus"):
        src, tgt, eligible = bh.load_task(task)        # identical rows and eligibility to the thesis
        for d in (src, tgt):
            d["temp_range"] = d.monthly_power_temperature_max_c - d.monthly_power_temperature_min_c
        return src, tgt, eligible
    if task == "locked":
        reg = p.eFUA_ID.map(FUA_REGION)
        src = p[p.country_iso.eq("CHN") & ~reg.eq(LOCKED)].copy()
        tgt = p[p.country_iso.eq("CHN") & reg.eq(LOCKED)].copy()
        eligible = eligible_fuas(tgt, .8)
        tgt = tgt[tgt.eFUA_ID.isin(eligible)].copy()
        for d in (src, tgt):
            d["pm25_value"] = d[Y]
            d.reset_index(drop=True, inplace=True)
        return src, tgt, eligible
    raise ValueError(task)


def source_model(src):
    return core.tree(True).fit(src[X], src[Y])


# ------------------------------------------------------------------ estimators
def affine(raw_l, y_l, raw_t):
    """Source-affine exactly as in the thesis: ridge (alpha 10) on the standardised source prediction.
    Returns unclipped fitted values for labelled rows and test rows."""
    z = StandardScaler()
    rx = z.fit_transform(raw_l[:, None])
    m = Ridge(alpha=10).fit(rx, y_l)
    return m.predict(rx), m.predict(z.transform(raw_t[:, None]))


def kern(a, b, ls):
    a, b = a / ls, b / ls
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return np.exp(-0.5 * np.maximum(d, 0))


def gp_mean(zl, r, zt, ls, sf, sn):
    k = sf ** 2 * kern(zl, zl, ls) + (sn ** 2 + 1e-9) * np.eye(len(zl))
    c = np.linalg.cholesky(k)
    alpha = np.linalg.solve(c.T, np.linalg.solve(c, r))
    return sf ** 2 * kern(zt, zl, ls) @ alpha


def gp_logml(zl, r, ls, sf, sn):
    k = sf ** 2 * kern(zl, zl, ls) + (sn ** 2 + 1e-9) * np.eye(len(zl))
    try:
        c = np.linalg.cholesky(k)
    except np.linalg.LinAlgError:
        return -1e12
    a = np.linalg.solve(c, r)
    return -0.5 * a @ a - np.log(np.diag(c)).sum() - 0.5 * len(r) * np.log(2 * np.pi)


def map_hyper(zl, r, ybar, prior, fit_ls=False):
    """MAP of log(a_f), log(a_n) (and optionally log length-scales) under independent normal priors.
    prior: dict with 'ls' (array, log-scale means), 'af', 'an' (log-scale means) and 'sd' (log-scale SDs)."""
    d = zl.shape[1]
    x0 = np.r_[prior["ls"], prior["af"], prior["an"]] if fit_ls else np.r_[prior["af"], prior["an"]]
    mu = x0.copy()
    sd = np.r_[np.full(d, prior["sd_ls"]), prior["sd"], prior["sd"]] if fit_ls else np.r_[prior["sd"], prior["sd"]]

    def nlp(x):
        if fit_ls:
            ls, af, an = np.exp(x[:d]), np.exp(x[d]), np.exp(x[d + 1])
        else:
            ls, af, an = np.exp(prior["ls"]), np.exp(x[0]), np.exp(x[1])
        lml = gp_logml(zl, r, ls, af * ybar, an * ybar)
        return -(lml - 0.5 * np.sum(((x - mu) / sd) ** 2))

    bounds = [(-5, 5)] * len(x0)
    res = minimize(nlp, x0, method="L-BFGS-B", bounds=bounds)
    x = res.x
    if fit_ls:
        return np.exp(x[:d]), np.exp(x[d]), np.exp(x[d + 1])
    return np.exp(prior["ls"]), np.exp(x[0]), np.exp(x[1])


def eca_predict(variant, theta, zl, r, zt, ybar):
    """theta: dict(ls=array of log length-scales, af=log a_f, an=log a_n) from meta-learning."""
    if variant == "ECA":
        ls, af, an = map_hyper(zl, r, ybar, dict(ls=theta["ls"], af=theta["af"], an=theta["an"], sd=1.0))
    elif variant == "ECA-fixed":
        ls, af, an = np.exp(theta["ls"]), np.exp(theta["af"]), np.exp(theta["an"])
    elif variant == "ECA-no-meta":
        prior = dict(ls=np.zeros(zl.shape[1]), af=np.log(.1), an=np.log(.1), sd=1.5, sd_ls=1.5)
        ls, af, an = map_hyper(zl, r, ybar, prior, fit_ls=True)
    elif variant == "ECA-unbounded":
        ls, af, an = np.exp(theta["ls"]), 10 * np.exp(theta["af"]), np.exp(theta["an"]) / 10
    elif variant == "ECA-isotropic":
        ls, af, an = map_hyper(zl, r, ybar, dict(ls=theta["ls"], af=theta["af"], an=theta["an"], sd=1.0))
    else:
        raise ValueError(variant)
    return gp_mean(zl, r, zt, ls, af * ybar, an * ybar), (ls, af, an)


def local_six(x, y, ex):
    """The six local learners of the thesis on the 16 inputs (same settings)."""
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.metrics.pairwise import rbf_kernel
    out = {"Local mean": np.full(len(ex), y.mean())}
    z = core.norm(); xx = z.fit_transform(x); ee = z.transform(ex)
    out["Local ridge"] = core.pos(Ridge(alpha=10).fit(xx, y).predict(ee))
    out["Local RBF"] = core.kernel(rbf_kernel(xx), rbf_kernel(ee, xx), y)
    out["Local RF"] = core.pos(RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features=1.0, n_jobs=1,
                                                     random_state=5993).fit(xx, y).predict(ee))
    out["Local Extra Trees"] = core.pos(ExtraTreesRegressor(n_estimators=80, min_samples_leaf=3, max_features=1.0,
                                                            n_jobs=1, random_state=5993).fit(xx, y).predict(ee))
    out["Local LightGBM"] = core.pos(core.tree().fit(x, y).predict(ex))
    return out


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


# ------------------------------------------------------------------ ECA2 (Amendment 2): adapt the source model,
# shrinking every adjustment toward the unadapted source prediction.
# delta(x) = a + b * s(x) + g(z(x)), with s the source prediction centred on its mean over the labelled rows and
# divided by its SD over the source rows; a, b, g zero-mean Gaussian (process) priors whose SDs are fractions of the
# labelled mean concentration. With few or inconsistent labels the posterior mean stays near zero (direct transfer).
ECA2_KEYS = ("aa", "ab", "ag", "an")


def eca2_cov(za, sa, zb, sb, ls, aa, ab, ag):
    """Prior covariance of delta in units of ybar^2."""
    out = aa ** 2 + ab ** 2 * np.outer(sa, sb)
    if ag > 0:
        out = out + ag ** 2 * kern(za, zb, ls)
    return out


def eca2_fit_predict(zl, sl, r, zt, st, ybar, ls, aa, ab, ag, an):
    k = eca2_cov(zl, sl, zl, sl, ls, aa, ab, ag) + (an ** 2 + 1e-9) * np.eye(len(zl))
    c = np.linalg.cholesky(k * ybar ** 2)
    alpha = np.linalg.solve(c.T, np.linalg.solve(c, r))
    return ybar ** 2 * eca2_cov(zt, st, zl, sl, ls, aa, ab, ag) @ alpha


def eca2_logml(zl, sl, r, ybar, ls, aa, ab, ag, an):
    k = (eca2_cov(zl, sl, zl, sl, ls, aa, ab, ag) + (an ** 2 + 1e-9) * np.eye(len(zl))) * ybar ** 2
    try:
        c = np.linalg.cholesky(k)
    except np.linalg.LinAlgError:
        return -1e12
    a = np.linalg.solve(c, r)
    return -0.5 * a @ a - np.log(np.diag(c)).sum() - 0.5 * len(r) * np.log(2 * np.pi)


def theta2_unpack(t):
    """t: dict from theta2.json for one family -> (ls array, aa, ab, ag, an) on the natural scale."""
    ls = np.exp(np.asarray(t["ls"], float)) if t.get("ls") is not None else np.ones(len(COND))
    ag = np.exp(t["ag"]) if t.get("ag") is not None else 0.0
    return ls, np.exp(t["aa"]), np.exp(t["ab"]), ag, np.exp(t["an"])


def eca2_ratios(th):
    """Signal-to-noise form of a theta2 family: (ls, rho_a, rho_b, rho_g). The posterior mean depends only on these
    ratios, so the meta-learned absolute scale is not identified and is not used."""
    ls, aa, ab, ag, an = theta2_unpack(th)
    return ls, aa / an, ab / an, ag / an


def eca2_post_mean(zl, sl, r, zt, st, ls, ra, rb, rg):
    k = eca2_cov(zl, sl, zl, sl, ls, ra, rb, rg) + (1 + 1e-9) * np.eye(len(zl))
    c = np.linalg.cholesky(k)
    return eca2_cov(zt, st, zl, sl, ls, ra, rb, rg) @ np.linalg.solve(c.T, np.linalg.solve(c, r))


def eca2_profile_ll(zl, sl, r, ls, ra, rb, rg):
    """Log marginal likelihood with the noise variance profiled out (sigma_n^2 = r' R^-1 r / n)."""
    k = eca2_cov(zl, sl, zl, sl, ls, ra, rb, rg) + (1 + 1e-9) * np.eye(len(zl))
    try:
        c = np.linalg.cholesky(k)
    except np.linalg.LinAlgError:
        return -1e12
    a = np.linalg.solve(c, r); n = len(r)
    s2 = max(a @ a / n, 1e-12)
    return -0.5 * n * np.log(s2) - np.log(np.diag(c)).sum() - 0.5 * n * (1 + np.log(2 * np.pi))


def eca2_predict(variant, th2, zl, sl, r, zt, st, ybar=None):
    """th2: theta2.json content. Returns (delta at test rows, (ls, rho_a, rho_b, rho_g)).
    ECA2: meta-learned ratios. ECA2-MAP: rho_a, rho_b, rho_g re-estimated on the target labels (profile likelihood,
    log-normal priors with SD 1 centred on the meta-learned ratios), length-scales kept. ECA2-no-meta: all ratios and
    length-scales from the target labels under generic priors. ECA2-affine / ECA2-isotropic: their own meta-learned
    families. ECA2-unbounded: meta-learned ratios times 10."""
    if variant == "ECA2":
        hp = eca2_ratios(th2["ARD"])
    elif variant == "ECA2-affine":
        hp = eca2_ratios(th2["affine"])
    elif variant == "ECA2-isotropic":
        hp = eca2_ratios(th2["isotropic"])
    elif variant == "ECA2-unbounded":
        ls, ra, rb, rg = eca2_ratios(th2["ARD"])
        hp = (ls, 10 * ra, 10 * rb, 10 * rg)
    elif variant in ("ECA2-MAP", "ECA2-no-meta"):
        d = zl.shape[1]
        if variant == "ECA2-MAP":
            ls, ra, rb, rg = eca2_ratios(th2["ARD"])
            mu = np.log([ra, rb, rg]); sd = np.ones(3)

            def unpack(x):
                return (ls, *np.exp(x))
        else:
            mu = np.r_[np.zeros(d), np.zeros(3)]; sd = np.full(d + 3, 1.5)

            def unpack(x):
                return (np.exp(x[:d]), *np.exp(x[d:]))

        def nlp(x):
            return -(eca2_profile_ll(zl, sl, r, *unpack(x)) - 0.5 * np.sum(((x - mu) / sd) ** 2))

        x = minimize(nlp, mu.copy(), method="L-BFGS-B", bounds=[(-12, 6)] * len(mu)).x
        hp = unpack(x)
    else:
        raise ValueError(variant)
    return eca2_post_mean(zl, sl, r, zt, st, *hp), hp


def scaled_source(raw_l, raw_t, src_sd):
    """Source prediction centred on the labelled rows' mean and divided by its SD over the source rows."""
    c = raw_l.mean()
    return (raw_l - c) / src_sd, (raw_t - c) / src_sd
