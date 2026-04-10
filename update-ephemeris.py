#!/usr/bin/env python3
"""Fetch latest Orion ephemeris from JPL HORIZONS and regenerate Chebyshev file."""

import urllib.request, urllib.parse, json, math, sys, os
from datetime import datetime, timezone

# Mission time range (matching config.json)
MISSION_START = '2026-04-02T01:59:00'
MISSION_END   = '2026-04-10T23:54:00'

SC_ID   = '-1024'   # Orion / Artemis II
MOON_ID = '301'
SUN_ID  = '10'

TOLERANCE_SC_KM   = 0.5
TOLERANCE_MOON_KM = 0.5
TOLERANCE_SUN_KM  = 5.0

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geo-ORION-cheb.json')


def fetch_horizons(command, start, stop, step='4m'):
    """Fetch ecliptic J2000 geocentric position vectors from HORIZONS."""
    params = {
        'format': 'text', 'COMMAND': f"'{command}'", 'OBJ_DATA': "'NO'",
        'MAKE_EPHEM': "'YES'", 'EPHEM_TYPE': "'VECTORS'", 'CENTER': "'500@399'",
        'REF_PLANE': "'ECLIPTIC'", 'REF_SYSTEM': "'J2000'", 'VEC_TABLE': "'1'",
        'VEC_LABELS': "'NO'", 'CSV_FORMAT': "'YES'",
        'STEP_SIZE': f"'{step}'", 'START_TIME': f"'{start}'", 'STOP_TIME': f"'{stop}'",
    }
    url = 'https://ssd.jpl.nasa.gov/api/horizons.api?' + urllib.parse.urlencode(
        params, quote_via=urllib.parse.quote)
    with urllib.request.urlopen(url, timeout=120) as resp:
        text = resp.read().decode()
    in_data = False
    points = []
    for line in text.split('\n'):
        if '$$SOE' in line: in_data = True; continue
        if '$$EOE' in line: break
        if in_data and line.strip():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                points.append((float(parts[0]), float(parts[2]),
                               float(parts[3]), float(parts[4])))
    return points


def fetch_chunked(command, start, stop, step, chunk_days=2):
    """Fetch in time chunks to stay within HORIZONS output limits."""
    from datetime import datetime, timedelta
    fmt = '%Y-%m-%dT%H:%M:%S'
    t0 = datetime.strptime(start, fmt)
    t1 = datetime.strptime(stop, fmt)
    all_pts = []
    cs = t0
    while cs < t1:
        ce = min(cs + timedelta(days=chunk_days), t1)
        pts = fetch_horizons(command, cs.strftime(fmt), ce.strftime(fmt), step)
        if all_pts and pts and pts[0][0] == all_pts[-1][0]:
            pts = pts[1:]
        all_pts.extend(pts)
        cs = ce
    return all_pts


def chebfit(x, y, degree):
    """Pure-Python Chebyshev fit using normal equations (no numpy needed)."""
    n = len(x)
    # Build Chebyshev basis matrix T[i][j] = T_j(x_i)
    T = [[0.0]*n for _ in range(degree+1)]
    for i in range(n):
        T[0][i] = 1.0
        if degree >= 1:
            T[1][i] = x[i]
        for j in range(2, degree+1):
            T[j][i] = 2*x[i]*T[j-1][i] - T[j-2][i]

    # Normal equations: (T * T^T) c = T * y
    m = degree + 1
    A = [[0.0]*m for _ in range(m)]
    b = [0.0]*m
    for j in range(m):
        for k in range(m):
            s = 0.0
            for i in range(n):
                s += T[j][i] * T[k][i]
            A[j][k] = s
        s = 0.0
        for i in range(n):
            s += T[j][i] * y[i]
        b[j] = s

    # Solve via Cholesky-like Gaussian elimination
    for col in range(m):
        pivot = A[col][col]
        if abs(pivot) < 1e-30:
            continue
        for row in range(col+1, m):
            factor = A[row][col] / pivot
            for k in range(col, m):
                A[row][k] -= factor * A[col][k]
            b[row] -= factor * b[col]
    c = [0.0]*m
    for row in range(m-1, -1, -1):
        s = b[row]
        for k in range(row+1, m):
            s -= A[row][k] * c[k]
        c[row] = s / A[row][row] if abs(A[row][row]) > 1e-30 else 0.0
    return c


def chebval(c, t):
    """Evaluate Chebyshev series using Clenshaw recurrence (matches JS evalCheb)."""
    b1, b2 = 0.0, 0.0
    for i in range(len(c)-1, 0, -1):
        tmp = 2*t*b1 - b2 + c[i]
        b2, b1 = b1, tmp
    return t*b1 - b2 + c[0]


def fit_segments(points, tolerance_km, max_degree=16, target_span_min=30):
    """Fit adaptive Chebyshev segments to position data."""
    n = len(points)
    jds = [p[0] for p in points]
    xs  = [p[1] for p in points]
    ys  = [p[2] for p in points]
    zs  = [p[3] for p in points]

    def bisect_right(val):
        lo, hi = 0, n
        while lo < hi:
            mid = (lo+hi)//2
            if jds[mid] <= val: lo = mid+1
            else: hi = mid
        return lo

    segments = []
    i = 0
    while i < n - 1:
        best_seg = None
        best_end = i + 1
        for span_mult in [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
            end_jd = jds[i] + target_span_min * span_mult / (24*60)
            j_end = min(n-1, bisect_right(end_jd))
            if j_end <= i+1: j_end = min(n-1, i+2)
            if j_end == best_end and best_seg is not None:
                continue
            for degree in range(2, max_degree+1):
                if j_end - i < degree:
                    continue
                t_start, t_end = jds[i], jds[j_end]
                if t_end <= t_start:
                    continue
                # Normalise to [-1, 1]
                tn = [2*(jds[k]-t_start)/(t_end-t_start)-1 for k in range(i, j_end+1)]
                xsl = xs[i:j_end+1]
                ysl = ys[i:j_end+1]
                zsl = zs[i:j_end+1]
                cx = chebfit(tn, xsl, degree)
                cy = chebfit(tn, ysl, degree)
                cz = chebfit(tn, zsl, degree)
                max_err = 0.0
                for k in range(len(tn)):
                    dx = chebval(cx, tn[k]) - xsl[k]
                    dy = chebval(cy, tn[k]) - ysl[k]
                    dz = chebval(cz, tn[k]) - zsl[k]
                    e = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if e > max_err: max_err = e
                if max_err <= tolerance_km:
                    if j_end > best_end or best_seg is None:
                        best_seg = {
                            't_start': t_start, 't_end': t_end,
                            'cx': cx, 'cy': cy, 'cz': cz,
                            'i_end': j_end, 'max_err': max_err
                        }
                        best_end = j_end
                    break
        if best_seg is None:
            j_end = min(n-1, i+3)
            t_start, t_end = jds[i], jds[j_end]
            tn = [2*(jds[k]-t_start)/(t_end-t_start)-1 for k in range(i, j_end+1)]
            deg = min(max_degree, j_end-i-1)
            best_seg = {
                't_start': t_start, 't_end': t_end,
                'cx': chebfit(tn, xs[i:j_end+1], deg),
                'cy': chebfit(tn, ys[i:j_end+1], deg),
                'cz': chebfit(tn, zs[i:j_end+1], deg),
                'i_end': j_end, 'max_err': 999
            }
        segments.append(best_seg)
        i = best_seg['i_end']
    return segments


def main():
    print('Fetching SC (Orion) at 1-min resolution...')
    sc = fetch_chunked(SC_ID, MISSION_START, MISSION_END, '1m', chunk_days=2)
    print(f'  {len(sc)} points')

    print('Fetching Moon at 4-min resolution...')
    moon = fetch_chunked(MOON_ID, MISSION_START, MISSION_END, '4m', chunk_days=5)
    print(f'  {len(moon)} points')

    print('Fetching Sun at 10-min resolution...')
    sun = fetch_horizons(SUN_ID, MISSION_START, MISSION_END, '10m')
    print(f'  {len(sun)} points')

    if not sc or not moon or not sun:
        print('ERROR: empty data from HORIZONS', file=sys.stderr)
        sys.exit(1)

    print('Fitting SC Chebyshev segments...')
    sc_segs = fit_segments(sc, TOLERANCE_SC_KM, max_degree=16, target_span_min=30)
    sc_err = max(s['max_err'] for s in sc_segs)
    print(f'  {len(sc_segs)} segments, max error {sc_err:.4f} km')

    print('Fitting Moon Chebyshev segments...')
    moon_segs = fit_segments(moon, TOLERANCE_MOON_KM, max_degree=8, target_span_min=120)
    moon_err = max(s['max_err'] for s in moon_segs)
    print(f'  {len(moon_segs)} segments, max error {moon_err:.4f} km')

    print('Fitting Sun Chebyshev segments...')
    sun_segs = fit_segments(sun, TOLERANCE_SUN_KM, max_degree=6, target_span_min=240)
    sun_err = max(s['max_err'] for s in sun_segs)
    print(f'  {len(sun_segs)} segments, max error {sun_err:.4f} km')

    def clean(segs):
        return [{'t_start': s['t_start'], 't_end': s['t_end'],
                 'cx': s['cx'], 'cy': s['cy'], 'cz': s['cz']} for s in segs]

    output = {
        'format': 'chebyshev-ephemeris', 'version': '1.0',
        'metadata': {
            'source': f'JPL HORIZONS API (auto-update {datetime.now(timezone.utc).strftime("%Y-%m-%d")})',
            'created': datetime.now(timezone.utc).isoformat(),
            'tolerance_km': TOLERANCE_SC_KM,
            'segments_count': len(sc_segs) + len(moon_segs) + len(sun_segs),
            'bodies': ['MOON', 'SC', 'SUN'],
            'coordinate_frame': 'J2000',
            'units': {'time': 'julian_date', 'position': 'km'}
        },
        'time_range': {'start': sc[0][0], 'end': sc[-1][0]},
        'MOON': {'time_range': {'start': moon[0][0], 'end': moon[-1][0]}, 'segments': clean(moon_segs)},
        'SC':   {'time_range': {'start': sc[0][0],   'end': sc[-1][0]},   'segments': clean(sc_segs)},
        'SUN':  {'time_range': {'start': sun[0][0],  'end': sun[-1][0]},  'segments': clean(sun_segs)},
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f)
    size_kb = os.path.getsize(OUTPUT_FILE) / 1000
    print(f'\nWrote {OUTPUT_FILE} ({size_kb:.0f} KB)')


if __name__ == '__main__':
    main()
