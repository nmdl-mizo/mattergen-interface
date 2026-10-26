"""common.py -- common part of the Fig. 2 redesign, first stage (paths, data loading, Arial registration, SVG post-processing).
SVG post-processing is a copy of postprocess_svg of the Fig. 3 script (make_fig3.py, L445-L523): materialize <use>, remove clipPath, opacity0 -> none.
Assumes running with the evaluation python (environment-eval.yml) on a login node. Read-only."""
import os, re, glob, dataclasses
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

GC = '<ORIGINAL_ROOT>/workspace'
RN = GC + "/experiments/relax_novelty/results"
VR = GC + "/experiments/verification/results"
D = GC + "/paper/figure_sources/fig2"
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]   # same values as paperfig.py L31-40

E1_PATH = RN + "/gb_equivalence/gb_symfit_main.csv"
E2_PATH = VR + "/gb_stage2/gb_stage2_main.csv"
B_PATH = VR + "/summary_tables/m1_classes.csv"


def sid(arm, chunk, cif):
    return "%s_c%02d_i%02d" % (arm, int(chunk), int(cif))


def load_E1():
    return pd.read_csv(E1_PATH)


def load_E2():
    return pd.read_csv(E2_PATH)


def load_R():
    out = []
    for arm in ["proposed", "baseline"]:
        for f in sorted(glob.glob(RN + "/relax/main/%s_w2.0/chunk*/relax_rows.csv" % arm)):
            r = pd.read_csv(f)
            r["relax_dir"] = os.path.dirname(f)
            out.append(r)
    return pd.concat(out, ignore_index=True)


def setup_arial():
    import matplotlib.font_manager as fm
    if not any(e.name == "Arial" for e in fm.fontManager.ttflist):
        for f in sorted(glob.glob("/usr/share/fonts/liberation-sans/LiberationSans-*.ttf")):
            fm.fontManager.addfont(f)
        for e in [e for e in fm.fontManager.ttflist if e.name == "Liberation Sans"]:
            fm.fontManager.ttflist.append(dataclasses.replace(e, name="Arial"))
    return fm.findfont("Arial", fallback_to_default=False)


def mpl_setup():
    import matplotlib
    matplotlib.use("Agg")
    ff = setup_arial()
    matplotlib.rcParams.update({
        "font.family": "Arial", "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 7, "axes.linewidth": 0.5, "svg.hashsalt": "fig2",
    })
    return ff


# ---------------------------------------------------------------- SVG post-processing (copied from the Fig. 3 script)
SVGNS, XLNS = "http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink"
for p_, u_ in [("", SVGNS), ("xlink", XLNS), ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
               ("cc", "http://creativecommons.org/ns#"), ("dc", "http://purl.org/dc/elements/1.1/")]:
    ET.register_namespace(p_, u_)


def _parse_style(s):
    d = {}
    for part in (s or "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def _shift_path(d, dx, dy):
    toks = re.findall(r"[A-Za-z]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            assert t in "MLCQz" or t == "Z", "unsupported path command %r" % t
            out.append(t); i += 1
            continue
        out.append("%.5f %.5f" % (float(toks[i]) + dx, float(toks[i + 1]) + dy)); i += 2
    return " ".join(out)


def postprocess_svg(path):
    tree = ET.parse(path)
    root = tree.getroot()
    q = lambda t: "{%s}%s" % (SVGNS, t)
    parent = {c: p for p in root.iter() for c in p}
    defs_el = {e.get("id"): e for e in root.iter() if e.tag == q("path") and e.get("id") and parent[e].tag == q("defs")}
    n_use = 0
    used_ids = set()
    for u in [e for e in root.iter() if e.tag == q("use")]:
        ref = u.get("{%s}href" % XLNS)
        assert ref and ref.startswith("#") and ref[1:] in defs_el, ref
        assert u.get("transform") is None, "use with transform"
        r = defs_el[ref[1:]]
        used_ids.add(ref[1:])
        st = _parse_style(r.get("style")); st.update(_parse_style(u.get("style")))
        new = ET.Element(q("path"))
        new.set("d", _shift_path(r.get("d"), float(u.get("x", 0)), float(u.get("y", 0))))
        if st:
            new.set("style", "; ".join("%s: %s" % kv for kv in st.items()))
        p = parent[u]
        p.insert(list(p).index(u), new)
        p.remove(u)
        n_use += 1
    for i in used_ids:
        d_el = defs_el[i]
        parent[d_el].remove(d_el)
    n_clip = 0
    for e in list(root.iter()):
        if e.tag == q("clipPath"):
            parent[e].remove(e); n_clip += 1
    for e in root.iter():
        if "clip-path" in e.attrib:
            del e.attrib["clip-path"]
        if e.tag == q("text") and re.fullmatch(r"rotate\(-?0 [\d.\-]+ [\d.\-]+\)", e.get("transform", "")):
            del e.attrib["transform"]
    for e in root.iter():
        if e.get("style"):
            st = _parse_style(e.get("style"))
            ch = False
            if st.get("fill-opacity") == "0":
                st["fill"] = "none"; del st["fill-opacity"]; ch = True
            if st.get("stroke-opacity") == "0":
                st["stroke"] = "none"; del st["stroke-opacity"]; ch = True
            if ch:
                e.set("style", "; ".join("%s: %s" % kv for kv in st.items()))
    for e in list(root.iter()):
        if e.tag == q("defs") and len(e) == 0:
            parent[e].remove(e)
    tree.write(path, xml_declaration=True, encoding="utf-8")
    return n_use, n_clip


def save_svg(fig, path, **kw):
    fig.savefig(path, format="svg", facecolor="none", metadata={"Date": None}, **kw)
    return postprocess_svg(path)


# ---------------------------------------------------------------- verification.csv (each script replaces only its own rows)
def write_verification(script, rows):
    """rows: list of (item, expected, observed, result). Replaces the earlier rows of the same script."""
    p = D + "/verification.csv"
    new = pd.DataFrame([dict(script=script, item=i, expected=e, observed=o, result=r) for i, e, o, r in rows],
                       columns=["script", "item", "expected", "observed", "result"])
    if os.path.exists(p):
        old = pd.read_csv(p, dtype=str, keep_default_na=False)
        old = old[old.script != script]
        new = pd.concat([old, new.astype(str)], ignore_index=True)
    new.to_csv(p, index=False)


def rsvg(svg, png, w=2400):
    import subprocess
    r = subprocess.run(["/usr/bin/rsvg-convert", "-w", str(w), "-b", "white", "-o", png, svg], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
