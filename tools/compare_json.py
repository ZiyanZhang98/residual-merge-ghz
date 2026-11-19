#!/usr/bin/env python3
"""Compare a regenerated JSON result with the archived one.

    python tools/compare_json.py NEW.json ARCHIVED.json [--rtol 1e-9] [--ignore key ...]

Numbers are compared with a relative tolerance (default 1e-9: identical up to floating-point summation order);
everything else must be equal. Exits with status 1 and lists the first differences if they disagree.
"""
import argparse
import json
import math
import sys


def walk(a, b, path, out, rtol, ignore):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in ignore:
                continue
            if k not in a or k not in b:
                out.append('%s/%s: only in %s' % (path, k, 'new' if k in a else 'archived'))
            else:
                walk(a[k], b[k], '%s/%s' % (path, k), out, rtol, ignore)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append('%s: length %d vs %d' % (path, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, '%s[%d]' % (path, i), out, rtol, ignore)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if not (a == b or math.isclose(a, b, rel_tol=rtol, abs_tol=rtol*1e-3)):
            out.append('%s: %r vs %r' % (path, a, b))
    elif a != b:
        out.append('%s: %r vs %r' % (path, a, b))


def compare(new, old, rtol=1e-9, ignore=()):
    out = []
    walk(json.load(open(new)), json.load(open(old)), '', out, rtol, set(ignore))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('new')
    p.add_argument('archived')
    p.add_argument('--rtol', type=float, default=1e-9)
    p.add_argument('--ignore', nargs='*', default=[])
    a = p.parse_args()
    diffs = compare(a.new, a.archived, a.rtol, a.ignore)
    if diffs:
        print('DIFFERENT (%d): %s vs %s' % (len(diffs), a.new, a.archived))
        for d in diffs[:20]:
            print('  ', d)
        sys.exit(1)
    print('IDENTICAL: %s == %s' % (a.new, a.archived))


if __name__ == '__main__':
    main()
