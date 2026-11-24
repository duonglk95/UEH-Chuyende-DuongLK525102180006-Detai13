"""Ham dung chung cho De tai 13 - Credit Scoring BNPL.

Du lieu: Home Credit - Credit Risk Model Stability (Kaggle 2024).
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve)

KAGGLE_DIR = "/kaggle/input/home-credit-credit-risk-model-stability/parquet_files/train"
KEYS = ("case_id", "WEEK_NUM", "MONTH", "target", "num_group1", "num_group2")
META = ["case_id", "date_decision", "MONTH", "WEEK_NUM", "target"]
SPLIT = "train"      # "test" khi du doan cho tap test cua cuoc thi (notebook 07)


# ---------------------------------------------------------------- duong dan
def get_data_dir() -> Path:
    """Thu muc chua train_*.parquet: bien moi truong HC_DATA_DIR > Kaggle > Data/raw."""
    if os.environ.get("HC_DATA_DIR"):
        return Path(os.environ["HC_DATA_DIR"])
    if Path(KAGGLE_DIR).exists():
        return Path(KAGGLE_DIR)
    if Path("/kaggle/input").exists():          # tu do thu muc chua train_base.parquet
        hits = glob.glob("/kaggle/input/**/train_base.parquet", recursive=True)
        if hits:
            return Path(hits[0]).parent
        raise FileNotFoundError("Khong thay train_base.parquet trong /kaggle/input. "
                                "Kiem tra Notebook da Add Input cuoc thi home-credit-credit-risk-model-stability chua.")
    return Path(__file__).resolve().parent.parent / "Data" / "raw"


def get_out_dir() -> Path:
    """Thu muc luu file trung gian: HC_OUT_DIR > /kaggle/working > Data/processed."""
    if os.environ.get("HC_OUT_DIR"):
        out = Path(os.environ["HC_OUT_DIR"])
    elif Path("/kaggle/working").exists():
        out = Path("/kaggle/working")
    else:
        out = Path(__file__).resolve().parent.parent / "Data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    return out


def table_files(name: str) -> list:
    """Cac file parquet cua mot bang, vd 'static_0' -> train_static_0_0.parquet, ...
    SPLIT = "test" doc test_*.parquet trong thu muc .../parquet_files/test."""
    d = get_data_dir()
    if SPLIT != "train":
        d = d.parent / SPLIT
    files = sorted(glob.glob(str(d / f"{SPLIT}_{name}.parquet")) +
                   glob.glob(str(d / f"{SPLIT}_{name}_[0-9]*.parquet")))
    if not files:
        raise FileNotFoundError(f"Khong tim thay bang '{name}' trong {d}. Xem Data/README.txt")
    return files


# ---------------------------------------------------------------- doc du lieu
def _to_date(c, dt):
    if dt == pl.String:
        return pl.col(c).str.to_date(strict=False)
    return pl.col(c).cast(pl.Date)


def cast_by_suffix(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Ep kieu theo hau to ten cot cua bo du lieu: P/A = so, M = chuoi, D = ngay."""
    exprs = []
    for c, dt in lf.collect_schema().items():
        if c in KEYS:
            exprs.append(pl.col(c).cast(pl.Int64))
        elif c == "date_decision" or c[-1] == "D":
            exprs.append(_to_date(c, dt))
        elif c[-1] in "PA":
            exprs.append(pl.col(c).cast(pl.Float64))
        elif c[-1] == "M":
            exprs.append(pl.col(c).cast(pl.String))
    return lf.with_columns(exprs)


def scan_table(name: str) -> pl.LazyFrame:
    parts = [cast_by_suffix(pl.scan_parquet(f)) for f in table_files(name)]
    return parts[0] if len(parts) == 1 else pl.concat(parts, how="diagonal_relaxed")


def load_base(sample_frac=None, seed=42) -> pl.LazyFrame:
    """Bang goc. sample_frac: lay mau ho so theo hash case_id (du lieu that, khong sinh them)."""
    base = scan_table("base")
    if sample_frac is not None and sample_frac < 1:
        base = base.filter(pl.col("case_id").hash(seed) % 1000 < int(sample_frac * 1000))
    return base


def collect(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect()


# ---------------------------------------------------------------- dac trung
def _dates_to_days(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Doi cot ngay thanh so ngay so voi date_decision (am = truoc ngay quyet dinh)."""
    cols = [c for c, dt in lf.collect_schema().items() if dt == pl.Date and c != "date_decision"]
    return lf.with_columns(
        [(pl.col(c) - pl.col("date_decision")).dt.total_days().cast(pl.Float32) for c in cols]
    ).drop("date_decision")


def _with_decision_date(name: str, base: pl.LazyFrame) -> pl.LazyFrame:
    keys = base.select("case_id", "date_decision")
    return _dates_to_days(scan_table(name).join(keys, on="case_id", how="inner"))


def depth0_features(name: str, base: pl.LazyFrame) -> pl.LazyFrame:
    """Bang 1 dong / ho so."""
    return _with_decision_date(name, base)


def depth1_features(name: str, base: pl.LazyFrame) -> pl.LazyFrame:
    """Bang nhieu dong / ho so -> gop ve 1 dong: so ban ghi, max, mean, gia tri dau."""
    lf = _with_decision_date(name, base)
    schema = lf.collect_schema()
    sort_col = "num_group1" if "num_group1" in schema else None
    exprs = [pl.len().alias(f"{name}__count")]
    for c, dt in schema.items():
        if c in KEYS:
            continue
        if dt == pl.Boolean:
            exprs.append(pl.col(c).cast(pl.Float32).mean().alias(f"{name}__{c}__mean"))
        elif dt.is_numeric():
            exprs.append(pl.col(c).max().alias(f"{name}__{c}__max"))
            exprs.append(pl.col(c).mean().alias(f"{name}__{c}__mean"))
        elif dt == pl.String:
            exprs.append(pl.col(c).first().alias(f"{name}__{c}__first"))
    if sort_col:
        lf = lf.sort("case_id", sort_col)
    return lf.group_by("case_id").agg(exprs)


def build_features(depth0=(), depth1=(), sample_frac=None) -> pl.DataFrame:
    base = load_base(sample_frac)
    out = base
    for name in depth0:
        out = out.join(depth0_features(name, base), on="case_id", how="left")
    for name in depth1:
        out = out.join(depth1_features(name, base), on="case_id", how="left")
    return collect(out)


def prune_features(df: pl.DataFrame, max_null=0.95, max_card=200) -> pl.DataFrame:
    """Bo cot gan nhu rong, cot hang so, cot chuoi qua nhieu gia tri; ep kieu gon bo nho."""
    n = df.height
    drop = []
    for c in df.columns:
        if c in META:
            continue
        s = df[c]
        if s.null_count() / n > max_null:
            drop.append(c)
        elif s.dtype == pl.String:
            k = s.n_unique()
            if k <= 1 or k > max_card:
                drop.append(c)
        elif s.dtype.is_numeric() and s.n_unique() <= 1:
            drop.append(c)
    df = df.drop(drop)
    feats = [c for c in df.columns if c not in META]
    return df.with_columns(
        [pl.col(c).cast(pl.Float32) for c in feats if df[c].dtype.is_numeric() or df[c].dtype == pl.Boolean]
        + [pl.col(c).cast(pl.Categorical) for c in feats if df[c].dtype == pl.String]
    )


# ---------------------------------------------------------------- chia du lieu
def time_split(df: pd.DataFrame, train=0.70, valid=0.15):
    """Chia train/valid/test theo WEEK_NUM (test = cac tuan cuoi)."""
    weeks = np.sort(df["WEEK_NUM"].unique())
    n = len(weeks)
    w_tr = weeks[: int(n * train)]
    w_va = weeks[int(n * train): int(n * (train + valid))]
    w_te = weeks[int(n * (train + valid)):]
    return (df[df["WEEK_NUM"].isin(w_tr)].copy(),
            df[df["WEEK_NUM"].isin(w_va)].copy(),
            df[df["WEEK_NUM"].isin(w_te)].copy())


# ---------------------------------------------------------------- danh gia
def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def evaluate(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"AUC": auc, "Gini": 2 * auc - 1, "PR_AUC": average_precision_score(y, p),
            "KS": ks_stat(y, p), "Brier": brier_score_loss(y, np.clip(p, 0, 1))}


def psi(expected, actual, bins=10) -> float:
    """Population Stability Index. >0.1: can theo doi, >0.25: troi manh."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    e = np.histogram(expected, cuts)[0] / len(expected)
    a = np.histogram(np.clip(actual, cuts[0], cuts[-1]), cuts)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def gini_by_week(df: pd.DataFrame, score_col="score") -> pd.Series:
    def _g(g):
        if g["target"].nunique() < 2:
            return np.nan
        return 2 * roc_auc_score(g["target"], g[score_col]) - 1
    return df.groupby("WEEK_NUM")[["target", score_col]].apply(_g)


# ---------------------------------------------------------------- WoE / IV
class WoEEncoder(BaseEstimator, TransformerMixin):
    """Ma hoa Weight of Evidence. Cot so: chia bin theo phan vi; gia tri thieu la 1 bin rieng.

    WoE = ln( %good / %bad ) voi good = khong vo no (target=0). Chi fit tren tap train.
    """

    def __init__(self, bins=10):
        self.bins = bins

    def _bin(self, s: pd.Series, col):
        if col in self.edges_:
            b = pd.cut(pd.to_numeric(s, errors="coerce"), self.edges_[col], labels=False)
            return pd.Series(b, index=s.index).fillna(-1).astype(int)
        o = s.astype(object)
        return o.where(o.notna(), "NA").astype(str)

    def fit(self, X: pd.DataFrame, y):
        X = pd.DataFrame(X)
        y = pd.Series(np.asarray(y), index=X.index)
        G, B = (y == 0).sum(), (y == 1).sum()
        self.edges_, self.maps_, iv = {}, {}, {}
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                v = X[c].dropna().to_numpy(dtype=float)
                e = np.unique(np.quantile(v, np.linspace(0, 1, self.bins + 1))) if len(v) else np.array([])
                e = np.concatenate([[-np.inf], e[1:-1], [np.inf]])
                self.edges_[c] = e
            b = self._bin(X[c], c)
            tab = pd.crosstab(b, y).reindex(columns=[0, 1], fill_value=0)
            g, bad = (tab[0] + 0.5) / G, (tab[1] + 0.5) / B
            woe = np.log(g / bad)
            self.maps_[c] = woe.to_dict()
            iv[c] = float(((g - bad) * woe).sum())
        self.iv_ = pd.Series(iv).sort_values(ascending=False)
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        out = {c: self._bin(X[c], c).map(self.maps_[c]).fillna(0.0).astype(float) for c in self.columns_}
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_)


# ---------------------------------------------------------------- dinh nghia cot
def load_definitions() -> pd.Series:
    """Mo ta cot tu feature_definitions.csv cua cuoc thi (Series: ten cot -> mo ta)."""
    d = get_data_dir()
    cands = [p for p in (d.parent.parent / "feature_definitions.csv", d.parent / "feature_definitions.csv",
                         d / "feature_definitions.csv") if p.exists()]
    if not cands and Path("/kaggle/input").exists():
        cands = [Path(p) for p in glob.glob("/kaggle/input/**/feature_definitions.csv", recursive=True)]
    if not cands:
        print("Khong tim thay feature_definitions.csv - mo ta cot se de trong")
        return pd.Series(dtype=str)
    fd = pd.read_csv(cands[0])
    return fd.set_index(fd.columns[0])[fd.columns[1]].astype(str)


def base_name(col: str) -> str:
    """Ten cot goc: 'applprev_1__maxdpd_577P__mean' -> 'maxdpd_577P'; 'cb__x_1P__max' -> 'x_1P'."""
    parts = col.split("__")
    return parts[1] if len(parts) >= 2 else col


DERIVED_DESC = {
    "ratio_annuity_income": "Derived: monthly annuity / main income (annuity_780A / maininc_215A).",
    "ratio_credit_income": "Derived: credit amount / main income (credamount_770A / maininc_215A).",
    "ratio_annuity_credit": "Derived: monthly annuity / credit amount (annuity_780A / credamount_770A).",
}


def describe(cols, defs: pd.Series) -> pd.Series:
    return pd.Series({c: DERIVED_DESC.get(c) or defs.get(base_name(c), "") for c in cols}, dtype=object)


DUE_WORDS = r"due|maturity|matur|end date|expir|planned|expected|until|next|termination"


def future_date_columns(df: pl.DataFrame, feats, defs: pd.Series, min_share=0.01) -> pd.DataFrame:
    """Cot ngay (da doi thanh so ngay so voi ngay quyet dinh) co gia tri > 0, tuc su kien SAU ngay quyet dinh.

    Giu lai cot co mo ta la ngay den han / ngay ket thuc du kien (biet truoc duoc); cac cot con lai co ty le
    > min_share bi danh dau loai vi co the chua thong tin chi biet sau khi ra quyet dinh.
    """
    import re
    rows = []
    for c in feats:
        b = base_name(c)
        if not b.endswith("D") or not df[c].dtype.is_numeric():
            continue
        share = float((df[c] > 0).fill_null(False).mean())
        if share <= 0:
            continue
        d = defs.get(b, "")
        expected = bool(re.search(DUE_WORDS, d, re.I))
        rows.append({"cot": c, "ty_le_sau_ngay_qd": share, "mo_ta": d,
                     "loai": (share > min_share) and not expected})
    out = pd.DataFrame(rows, columns=["cot", "ty_le_sau_ngay_qd", "mo_ta", "loai"])
    return out.sort_values("ty_le_sau_ngay_qd", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- kiem dinh
def gini_stability(df: pd.DataFrame, score_col="score", w_fall=88.0, w_resstd=-0.5) -> dict:
    """Chi so cua cuoc thi Home Credit 2024: mean(Gini tuan) + 88*min(0, do doc) - 0.5*std(phan du)."""
    g = gini_by_week(df, score_col).dropna()
    x = np.arange(len(g))
    a, b = np.polyfit(x, g.values, 1)
    res = g.values - (a * x + b)
    return {"gini_tb": float(g.mean()), "do_doc": float(a), "std_phan_du": float(res.std()),
            "stability": float(g.mean() + w_fall * min(0, a) + w_resstd * res.std())}


def bootstrap_ci(y, p, metric=roc_auc_score, n=500, seed=42, alpha=0.05):
    """Khoang tin cay bootstrap cho mot chi so (mac dinh AUC)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(metric(y[i], p[i]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(metric(y, p)), float(lo), float(hi)


def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x)
    t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n); out[j] = t
    return out


def delong_test(y, p1, p2) -> dict:
    """Kiem dinh DeLong (DeLong 1988; thuat toan nhanh Sun & Xu 2014) cho chenh lech AUC hai mo hinh tren cung mau."""
    from scipy.stats import norm
    y = np.asarray(y); preds = np.vstack([np.asarray(p1), np.asarray(p2)])
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(r) for r in pos]); ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    auc = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (auc[0] - auc[1]) / np.sqrt(var)
    return {"AUC_1": float(auc[0]), "AUC_2": float(auc[1]), "chenh": float(auc[0] - auc[1]),
            "z": float(z), "p_value": float(2 * norm.sf(abs(z)))}


def calibration_stats(y, p, bins=10) -> dict:
    """ECE (theo thap phan vi), he so do doc va chan cua hieu chinh (logit(y) ~ a + b*logit(p))."""
    from sklearn.linear_model import LogisticRegression
    y, p = np.asarray(y), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    ece = float((d["n"] * (d["y"] - d["p"]).abs()).sum() / d["n"].sum())
    lg = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lg, y)
    return {"ECE": ece, "calib_slope": float(lr.coef_[0, 0]), "calib_intercept": float(lr.intercept_[0]),
            "pd_tb": float(p.mean()), "ty_le_thuc": float(y.mean())}


# ---------------------------------------------------------------- nhom hanh vi & credit bureau
import re as _re

BEHAVIOR_GROUPS = {
    "tre_han":       r"days past due|\bdpd\b|overdue|paid late|late payment|past due|delinquen|arrear",
    "vay_nhieu_noi": r"number of (active |open |existing |closed )?(credits?|contracts?|loans?)\b(?! payment)|active (credit|contract)|"
                     r"credit bureau quer|inquir|number of (previous |rejected )?(credit )?applications|"
                     r"applications made by the client|applications that were rejected|financial institution|creditor|lender",
    "ganh_nang_no":  r"debt|outstanding|instal?l?ment amount|annuity|monthly payment|utili[sz]|credit limit",
    "dao_no":        r"refinanc|restructur|prolong|roll.?over|top.?up|early repayment",
}


def assign_group(desc):
    """Nhom hanh vi theo mo ta cot (thu tu uu tien: tre han > vay nhieu noi > no > dao no)."""
    for g, p in BEHAVIOR_GROUPS.items():
        if _re.search(p, desc or "", _re.I):
            return g
    return None


def credit_bureau_features(base: pl.LazyFrame, defs: pd.Series) -> pl.DataFrame:
    """Gop credit_bureau_a_1 ve 1 dong / ho so (chi tu cot that cua bang, khong sinh du lieu):
    so ban ghi; max/mean/sum cot so thuoc nhom hanh vi; so gia tri khac nhau cua to chuc cho vay, muc dich, vai tro;
    so hop dong mo trong 30/90/180 ngay truoc ngay nop ho so va so ngay tu hop dong gan nhat."""
    cb = _with_decision_date("credit_bureau_a_1", base)
    sch = cb.collect_schema()
    desc = {c: defs.get(c, "") for c in sch if c not in KEYS}
    num_cols = [c for c in desc if c[-1] in "PAL" and sch[c].is_numeric() and assign_group(desc[c])]
    cat_cols = [c for c in desc if c[-1] == "M" and _re.search(r"financial institution|creditor|lender|subject|purpose|contract type|type of", desc[c], _re.I)]
    open_cols = [c for c in desc if c[-1] == "D" and _re.search(r"(start|open|begin|conclu|sign)", desc[c], _re.I)
                 and not _re.search(DUE_WORDS, desc[c], _re.I)]
    exprs = [pl.len().alias("cb__so_ban_ghi")]
    for c in num_cols:
        exprs += [pl.col(c).max().alias(f"cb__{c}__max"), pl.col(c).mean().alias(f"cb__{c}__mean"),
                  pl.col(c).sum().alias(f"cb__{c}__sum")]
    for c in cat_cols:
        exprs.append(pl.col(c).drop_nulls().n_unique().alias(f"cb__{c}__so_gia_tri"))
    for c in open_cols:
        for w in (30, 90, 180):
            exprs.append(((pl.col(c) >= -w) & (pl.col(c) <= 0)).sum().alias(f"cb__{c}__mo_trong_{w}n"))
        exprs.append(pl.col(c).filter(pl.col(c) <= 0).max().alias(f"cb__{c}__gan_nhat"))
    out = collect(cb.select(["case_id"] + num_cols + cat_cols + open_cols).group_by("case_id").agg(exprs))
    return out, {"num": num_cols, "cat": cat_cols, "open": open_cols}


def cb_group(col: str, defs: pd.Series):
    if col == "cb__so_ban_ghi" or col.endswith("__so_gia_tri") or "__mo_trong_" in col or col.endswith("__gan_nhat"):
        return "vay_nhieu_noi"
    return assign_group(defs.get(base_name(col), ""))


def align_features(X: pd.DataFrame, feats, schema) -> pd.DataFrame:
    """Dua bang (vd tap test) ve dung danh sach cot va kieu du lieu luc huan luyen (schema polars cua file parquet train)."""
    X = X.copy()
    for c in feats:
        if c not in X.columns:
            X[c] = np.nan
        if schema[c] == pl.Categorical:
            X[c] = X[c].astype("string").astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X


# ---------------------------------------------------------------- bo nho
def free_memory(ns: dict, min_mb: float = 20.0) -> None:
    """Xoa cac bang du lieu lon con sot lai tu phan truoc trong notebook gop (moi phan tu doc lai du lieu tu file)."""
    import gc
    freed, names = 0.0, []
    for k in list(ns.keys()):
        if k.startswith("_"):
            continue
        v = ns[k]
        try:
            if isinstance(v, pd.DataFrame):
                mb = v.memory_usage(deep=False).sum() / 1e6
            elif isinstance(v, pl.DataFrame):
                mb = v.estimated_size() / 1e6
            elif isinstance(v, np.ndarray):
                mb = v.nbytes / 1e6
            else:
                continue
        except Exception:
            continue
        if mb >= min_mb:
            del ns[k]; freed += mb; names.append(k)
    gc.collect()
    print("Giai phong %.0f MB: %s" % (freed, ", ".join(names) or "(khong co)"))
