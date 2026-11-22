"""Download USGS instantaneous discharge (00060, cfs) for upstream->downstream gauge pairs; hourly means.
Output: data/river/hourly.csv (columns = site ids, index = UTC hour) + data/river/pairs.json."""
import io, json, os, time, urllib.request
import pandas as pd

PAIRS = {
    "delaware": ("01438500", "01463500"),
    "susquehanna": ("01536500", "01570500"),
    "potomac": ("01638500", "01646500"),
    "connecticut": ("01170500", "01184000"),
    "james": ("02035000", "02037500"),
    "missouri": ("06909000", "06934500"),
    "mississippi": ("07010000", "07020500"),
    "ohio": ("03294500", "03303280"),
    "allegheny_ohio": ("03049500", "03086000"),
}
OUT = "data/river"
os.makedirs(OUT + "/raw", exist_ok=True)
YEARS = [(f"{y}-10-01", f"{y + 1}-09-30") for y in range(2015, 2025)]


def fetch(site, a, b):
    fn = f"{OUT}/raw/{site}_{a[:4]}.csv"
    if os.path.exists(fn):
        return pd.read_csv(fn, index_col=0, parse_dates=True).iloc[:, 0]
    url = (f"https://nwis.waterservices.usgs.gov/nwis/iv/?format=rdb&sites={site}&parameterCd=00060"
           f"&startDT={a}&endDT={b}")
    for k in range(5):
        try:
            txt = urllib.request.urlopen(url, timeout=300).read().decode()
            break
        except Exception as e:
            print("retry", site, a, repr(e)[:100], flush=True); time.sleep(15 * (k + 1))
    else:
        raise RuntimeError(f"failed {site} {a}")
    lines = [l for l in txt.splitlines() if not l.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])), sep="\t", dtype=str)
    vcol = [c for c in df.columns if c.endswith("_00060") and not c.endswith("_cd")]
    if not vcol:
        s = pd.Series(dtype=float)
    else:
        tz = {"EST": "-05:00", "EDT": "-04:00", "CST": "-06:00", "CDT": "-05:00", "MST": "-07:00", "MDT": "-06:00",
              "PST": "-08:00", "PDT": "-07:00", "UTC": "+00:00"}
        ts = pd.to_datetime(df["datetime"] + df["tz_cd"].map(tz), utc=True, format="%Y-%m-%d %H:%M%z")
        s = pd.Series(pd.to_numeric(df[vcol[0]], errors="coerce").values, index=ts).dropna()
        s = s.resample("h").mean()
    s.name = site
    s.to_csv(fn)
    return s


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    sites = sorted({s for p in PAIRS.values() for s in p})
    jobs = [(site, a, b) for site in sites for a, b in YEARS]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda j: fetch(*j), jobs))
    print("fetched", len(res), flush=True)
    cols = {}
    for site in sites:
        s = pd.concat([fetch(site, a, b) for a, b in YEARS])
        cols[site] = s[~s.index.duplicated()]
    H = pd.DataFrame(cols).sort_index()
    H.to_csv(f"{OUT}/hourly.csv")
    json.dump(PAIRS, open(f"{OUT}/pairs.json", "w"), indent=1)
    print("DONE", H.shape, H.notna().mean().round(3).to_dict())
