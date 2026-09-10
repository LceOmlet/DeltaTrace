"""Verify the official kit, source references, and manuscript input closure."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    provenance = json.loads((ROOT / 'template_source.json').read_text())
    for name, row in provenance['official_files'].items():
        assert sha(ROOT / 'official' / name) == row['sha256'], name
    for name in ('iclr2027_conference.sty','iclr2027_conference.bst','natbib.sty','fancyhdr.sty'):
        assert (ROOT / name).read_bytes() == (ROOT / 'official' / name).read_bytes(), name
    assert sha(ROOT / 'iclr2027-official.zip') == provenance['archive_sha256']
    files = [ROOT / 'main.tex'] + sorted((ROOT / 'sections').glob('*.tex')) + sorted((ROOT / 'figures').glob('*.tex')) + sorted((ROOT / 'results').glob('*.tex'))
    source = '\n'.join(re.sub(r'(?<!\\)%.*', '', f.read_text(encoding='utf-8')) for f in files)
    for child in re.findall(r'\\input\{([^}]+)\}', source):
        assert (ROOT / (child + '.tex')).exists(), child
    graphics = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', source)
    for graphic in graphics:
        assert (ROOT / graphic).is_file(), graphic
    manifest = json.loads((ROOT / 'figures/figure_manifest.json').read_bytes())
    assert sha(ROOT / 'figures/data/cases.json') == manifest['fixture_sha256']
    assert sha(ROOT / 'figures/data/overview_role_case.json') == manifest['overview_fixture_sha256']
    assert sha(ROOT / 'figures/data/overview_case.json') == manifest['lookup_fixture_sha256']
    assert sha(ROOT / 'figures/build_figures.py') == manifest['builder_sha256']
    assert sha(ROOT / 'figures/draw_mechanism.py') == manifest['mechanism_builder_sha256']
    for name, digest in manifest['generated_files'].items():
        assert sha(ROOT / 'figures/generated' / name) == digest, name
    labels = re.findall(r'\\label\{([^}]+)\}', source)
    assert len(labels) == len(set(labels)), 'Repeated labels'
    refs = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', source)
    assert not set(refs) - set(labels), set(refs) - set(labels)
    citations = {key.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}', source) for key in group.split(',')}
    bib = set(re.findall(r'@\w+\{([^,]+),', (ROOT / 'references.bib').read_text()))
    assert citations == bib, (citations - bib, bib - citations)
    assert '\\iclrfinalcopy' not in source
    assert not re.search(r'(?i)\b(TODO|FIXME|TBD|we should|we must|you should|let us|consider|unfortunately|caveat)\b', source)
    report = dict(official_kit_verified=True, official_style_modified=False,
                  manuscript_tex_files=len(files), unique_labels=len(labels), verified_citation_keys=sorted(citations),
                  reference_closure=True, anonymous_style=True,
                  figure_files_verified=len(manifest['generated_files']),
                  included_vector_figures=len(graphics),
                  figure_manifest_sha256=sha(ROOT / 'figures/figure_manifest.json'),
                  source_files={str(f.relative_to(ROOT)):sha(f) for f in files+[ROOT/'references.bib']})
    destination = ROOT / 'source_verification.json'
    destination.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_files'}))

if __name__ == '__main__':
    main()
