"""Copy the official 2027 kit from a pinned checkout without editing its files."""
import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkout', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    checkout = args.checkout.resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout).decode().strip()
    archive = checkout / 'iclr2027.zip'
    target = root / 'official'
    target.mkdir(exist_ok=True)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    entries = {}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith('/') or '__MACOSX' in name or Path(name).name.startswith('.'):
                continue
            if not name.endswith(('.tex', '.sty', '.bst', '.bib', '.pdf')):
                continue
            destination = target / Path(name).name
            raw = z.read(name)
            if destination.exists() and destination.read_bytes() != raw:
                raise ValueError(f'Existing official file differs: {destination.name}')
            destination.write_bytes(raw)
            entries[destination.name] = dict(archive_member=name, sha256=sha(raw), bytes=len(raw))
    for name in ('iclr2027_conference.sty', 'iclr2027_conference.bst', 'natbib.sty', 'fancyhdr.sty'):
        raw = (target / name).read_bytes()
        (root / name).write_bytes(raw)
    # The archive is preserved verbatim as a downloadable original template.
    shutil.copyfile(archive, root / 'iclr2027-official.zip')
    provenance = dict(
        author_guide='https://iclr.cc/Conferences/2027/AuthorGuidelines',
        conference_download='https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip',
        acquired_from='https://github.com/ICLR/Master-Template', commit=commit,
        archive_path='iclr2027.zip', archive_sha256=sha(archive.read_bytes()),
        source_note='Conference download hostname failed local DNS; official ICLR GitHub repository supplied its 2027 kit.',
        retrieved_on='2026-09-09', official_files=entries,
        official_style_modified=False, main_text_limit_pages=9)
    (root / 'template_source.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(commit=commit, official_files=len(entries), archive_sha256=provenance['archive_sha256'])))

if __name__ == '__main__':
    main()
