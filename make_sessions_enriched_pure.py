import json, math, random, os

IN_PATH = os.path.join("data", "processed", "sessions_composition.json")
OUT_PATH = os.path.join("data", "processed", "sessions_enriched.json")

R = 95.0
EPS = 1e-6
KEYS = ["page_view", "product_view", "add_to_cart", "purchase"]

def stable_unit_vector_from_id(s: str):
    # deterministic 3D unit vector from id (no numpy)
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    rng = random.Random(h)
    x = rng.gauss(0, 1)
    y = rng.gauss(0, 1)
    z = rng.gauss(0, 1)
    n = math.sqrt(x*x + y*y + z*z)
    if n < 1e-12:
        return (1.0, 0.0, 0.0)
    return (x/n, y/n, z/n)

def normalize(v):
    s = sum(v)
    if s <= 0:
        return [0.25, 0.25, 0.25, 0.25]
    return [x / s for x in v]

def clr(p):
    # p assumed normalized & positive
    lp = [math.log(x) for x in p]
    m = sum(lp) / len(lp)
    return [x - m for x in lp]

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def mat_vec(M, v):
    return [dot(row, v) for row in M]

def vec_norm(v):
    return math.sqrt(dot(v, v))

def gram_schmidt_orthonormalize(v, basis):
    # subtract projections onto existing basis
    out = v[:]
    for b in basis:
        proj = dot(out, b)
        out = [o - proj*bi for o, bi in zip(out, b)]
    n = vec_norm(out)
    if n < 1e-12:
        return None
    return [x/n for x in out]

def power_iteration(M, basis, iters=200):
    # find next eigenvector orthogonal to basis
    v = [1.0, 0.5, -0.3, 0.2]
    v = gram_schmidt_orthonormalize(v, basis)
    if v is None:
        v = [1.0, 0.0, 0.0, 0.0]

    for _ in range(iters):
        w = mat_vec(M, v)
        w = gram_schmidt_orthonormalize(w, basis)
        if w is None:
            break
        v = w
    # eigenvalue estimate
    Mv = mat_vec(M, v)
    lam = dot(v, Mv)
    return v, lam

def percentile(values, p):
    # p in [0,100]
    if not values:
        return 0.0
    xs = sorted(values)
    k = int(round((p/100.0) * (len(xs)-1)))
    k = max(0, min(len(xs)-1, k))
    return xs[k]

def main():
    with open(IN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build compositions
    P = []
    ids = []
    for d in data:
        ids.append(d["id"])
        P.append([float(d.get(k, 0.0)) for k in KEYS])

    n = len(P)
    if n == 0:
        raise SystemExit("No sessions found in input JSON.")

    # Precompute smoothed normalized compositions for entropy / CLR
    P_eps = []
    for p in P:
        pp = [x + EPS for x in p]
        pp = normalize(pp)
        P_eps.append(pp)

    # dominant + purity + entropy
    dominant = []
    purity = []
    entropy = []
    for p_raw, p in zip(P, P_eps):
        imax = max(range(4), key=lambda i: p_raw[i])
        dominant.append(KEYS[imax])
        purity.append(max(p_raw))
        H = 0.0
        for x in p:
            H -= x * math.log(x)
        entropy.append(H / math.log(4.0))  # normalized [0,1]

    # ---- Contrast sphere coords ----
    contrast_xyz = []
    for sid, p in zip(ids, P):
        pv, pr, ac, pu = p
        x = pv - pu
        y = pr - pv
        z = ac - pr
        nn = math.sqrt(x*x + y*y + z*z)
        if nn < 1e-6:
            ux, uy, uz = stable_unit_vector_from_id(sid)
            contrast_xyz.append((ux*R, uy*R, uz*R))
        else:
            contrast_xyz.append((x/nn*R, y/nn*R, z/nn*R))

    # ---- CLR + PCA baseline (pure python) ----
    # CLR vectors in R^4 (sum to 0); compute mean and covariance (4x4)
    clr_vecs = [clr(p) for p in P_eps]
    mean = [0.0, 0.0, 0.0, 0.0]
    for v in clr_vecs:
        for i in range(4):
            mean[i] += v[i]
    mean = [m / n for m in mean]

    # covariance
    C = [[0.0]*4 for _ in range(4)]
    for v in clr_vecs:
        dv = [v[i] - mean[i] for i in range(4)]
        for i in range(4):
            for j in range(4):
                C[i][j] += dv[i] * dv[j]
    # use average (doesn't matter for eigenvectors)
    inv = 1.0 / max(1, n-1)
    for i in range(4):
        for j in range(4):
            C[i][j] *= inv

    # top-3 eigenvectors via orthogonal power iteration
    basis = []
    eigvals = []
    for _ in range(3):
        v, lam = power_iteration(C, basis, iters=250)
        basis.append(v)
        eigvals.append(lam)

    # project each centered clr vector onto eigenvectors
    pcs = []
    norms = []
    for v in clr_vecs:
        dv = [v[i] - mean[i] for i in range(4)]
        s1 = dot(dv, basis[0])
        s2 = dot(dv, basis[1])
        s3 = dot(dv, basis[2])
        pcs.append((s1, s2, s3))
        norms.append(math.sqrt(s1*s1 + s2*s2 + s3*s3))

    # scale to match radius
    p99 = percentile(norms, 99)
    scale = (R * 0.9) / (p99 if p99 > 1e-12 else 1.0)
    pca_xyz = [(a*scale, b*scale, c*scale) for (a,b,c) in pcs]

    # ---- Write enriched ----
    enriched = []
    for i, d in enumerate(data):
        rec = dict(d)
        rec["dominant"] = dominant[i]
        rec["purity"] = float(purity[i])
        rec["entropy"] = float(entropy[i])

        xc, yc, zc = contrast_xyz[i]
        rec["x_contrast"] = float(xc)
        rec["y_contrast"] = float(yc)
        rec["z_contrast"] = float(zc)

        xp, yp, zp = pca_xyz[i]
        rec["x_pca"] = float(xp)
        rec["y_pca"] = float(yp)
        rec["z_pca"] = float(zp)

        enriched.append(rec)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f)

    print("✅ Wrote:", OUT_PATH)
    print("Sessions:", len(enriched))

if __name__ == "__main__":
    main()
