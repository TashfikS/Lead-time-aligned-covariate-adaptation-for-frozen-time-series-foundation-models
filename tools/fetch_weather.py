"""AirLead: upwind -> downwind hourly mean-sea-level pressure (ERA5 via Open-Meteo archive), 2015-10-01..2025-09-30.
Pairs are chosen a priori along the mid-latitude westerlies (upwind station west of the downwind one, 400-900 km),
before any data were inspected. Output: data/weather/hourly.csv (columns '<pair>_up', '<pair>_dn'), pairs.json."""
import json, os, time, urllib.request
import pandas as pd

PAIRS = {  # name: ((lat, lon) upwind, (lat, lon) downwind)
    "omaha_chicago": ((41.26, -95.94), (41.88, -87.63)),
    "minneapolis_milwaukee": ((44.98, -93.27), (43.04, -87.91)),
    "kansascity_stlouis": ((39.10, -94.58), (38.63, -90.20)),
    "dallas_memphis": ((32.78, -96.80), (35.15, -90.05)),
    "brest_paris": ((48.39, -4.49), (48.86, 2.35)),
    "dublin_london": ((53.35, -6.26), (51.51, -0.13)),
    "amsterdam_berlin": ((52.37, 4.90), (52.52, 13.40)),
    "paris_frankfurt": ((48.86, 2.35), (50.11, 8.68)),
    "adelaide_melbourne": ((-34.93, 138.60), (-37.81, 144.96)),
    "beijing_seoul": ((39.90, 116.40), (37.57, 126.98)),
}
OUT = "data/weather"
os.makedirs(OUT + "/raw", exist_ok=True)


def fetch(lat, lon, var="pressure_msl"):
    fn = f"{OUT}/raw/{lat}_{lon}.csv"
    if os.path.exists(fn):
        return pd.read_csv(fn, index_col=0, parse_dates=True).iloc[:, 0]
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={lat}&longitude={lon}&start_date=2015-10-01&end_date=2025-09-30&hourly={var}&timezone=GMT")
    for k in range(5):
        try:
            d = json.loads(urllib.request.urlopen(url, timeout=300).read().decode())
            break
        except Exception as e:
            print("retry", lat, lon, repr(e)[:100], flush=True); time.sleep(30 * (k + 1))
    s = pd.Series(d["hourly"][var], index=pd.to_datetime(d["hourly"]["time"], utc=True), dtype=float, name=f"{lat}_{lon}")
    s.to_csv(fn)
    return s


if __name__ == "__main__":
    cols = {}
    for name, (u, v) in PAIRS.items():
        cols[name + "_up"] = fetch(*u); time.sleep(2)
        cols[name + "_dn"] = fetch(*v); time.sleep(2)
        print("ok", name, len(cols[name + "_up"]), flush=True)
    H = pd.DataFrame(cols).sort_index()
    H.to_csv(f"{OUT}/hourly.csv")
    json.dump({n: [n + "_up", n + "_dn"] for n in PAIRS}, open(f"{OUT}/pairs.json", "w"), indent=1)
    print("DONE", H.shape, H.notna().mean().round(4).min())
