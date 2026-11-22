import urllib.request, urllib.parse, re, time


def q(params, tries=4):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    x = None
    for k in range(tries):
        try:
            x = urllib.request.urlopen(urllib.request.Request(url), timeout=60).read().decode()
            break
        except Exception:
            time.sleep(20 * (k + 1))
    if x is None:
        return [("ERR", "", "")]
    out = []
    for e in x.split("<entry>")[1:]:
        t = " ".join(re.search(r"<title>(.*?)</title>", e, re.S).group(1).split())
        i = re.search(r"<id>(.*?)</id>", e).group(1).split("/abs/")[-1]
        p = re.search(r"<published>(.*?)</published>", e).group(1)[:10]
        a = " ".join(re.search(r"<summary>(.*?)</summary>", e, re.S).group(1).split())
        out.append((i, p, t[:140], a))
    return out
