import argparse
from pathlib import Path
import hashlib
import json


def md5(path: Path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description='Build local mazurka_h5 index JSON for mirdata')
    ap.add_argument('--h5-root', required=True, help='Root folder with mazurka opus subfolders and .h5 files')
    ap.add_argument('--out', default='mazurka_h5_local.json', help='Output JSON filename (in this folder)')
    ap.add_argument('--no-checksum', action='store_true', help='Do not compute md5 (faster)')
    args = ap.parse_args()

    root = Path(args.h5_root)
    out = Path(args.out)

    entries = {}
    for p in sorted(root.rglob('*.h5')):
        # track_id as opus/file (no extension)
        rel = p.relative_to(root)
        track_id = str(rel.with_suffix('')).replace('\\', '/')
        rel_str = str(rel).replace('\\', '/')
        checksum = None if args.no_checksum else md5(p)
        entries[track_id] = {
            'h5': [rel_str, checksum]
        }

    index = {
        'version': 'local',
        'tracks': entries,
    }
    out.write_text(json.dumps(index, indent=2))
    print(f'Wrote {out} with {len(entries)} tracks')


if __name__ == '__main__':
    main()

