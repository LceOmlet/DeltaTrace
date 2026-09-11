"""Vector DT mechanism figure and data-backed, aligned signed-token figures.

Run with matplotlib available. cases.json is a self-contained checked fixture.
PDF and SVG text remain vector; PNG files are viewing copies.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np
from draw_mechanism import draw as mechanism_3d, SIGNED_COLORS

HERE = Path(__file__).resolve().parent
OUT = HERE / 'generated'
PAPER = HERE.parent
INK = '#263442'
MUTED = '#617181'
LINE = '#D9E0E6'
PALE = '#F5F7F9'
POS = '#238778'
NEG = '#CA6854'
VIOLET = '#746B91'
CMAP = LinearSegmentedColormap.from_list('dt_signed', SIGNED_COLORS)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.2,
                     'text.color': INK, 'axes.labelcolor': INK,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.unicode_minus': False})


def text(ax, x, y, value, size=10.2, weight='normal', color=INK, **kwargs):
    artist = ax.text(x, y, value, fontsize=size, fontweight=weight, color=color,
                     va='top', linespacing=1.25, **kwargs)
    if hasattr(ax.figure, '_case_text_artists'):
        ax.figure._case_text_artists.append(artist)
    return artist


def excerpts(case):
    source = case['user_text']
    if case['dataset'] == 'niah_mq_q2':
        strings = [
            'One of the special magic numbers for deafening-opium is: 5443951.',
            'One of the special magic numbers for mammoth-barber is: 8698256.',
            'What are all the special magic numbers for deafening-opium, and mammoth-barber mentioned in the provided text?']
    else:
        strings = [
            'Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor',
            'he also did additional voices in films "Fantasia 2000" (1999) and "Sing" (2016)',
            'It was directed and written by Garth Jennings, co-directed by Christophe Lourdelet,',
            'How many consonants are there in the last name of the person who directed and wrote the 2016 film featuring the voice of Townsend Coleman?']
    return [(source.index(s), source.index(s) + len(s)) for s in strings]


def token_characters(case):
    """Distribute a multi-byte character across its overlapping BPE token bytes.

    Displayed excerpts are ASCII, so each displayed character has exactly one
    token owner; the assertion below prevents silently merging token colors.
    """
    owners = [[] for _ in case['user_text']]
    for i, token in enumerate(case['tokens']):
        lo, hi = token['char_span']
        for j in range(lo, hi):
            owners[j].append(i)
    return owners


def heatmap(ax, case, x, y, width, norm, font=10.15, sections=False, records=None):
    owners = token_characters(case)
    step = font / 72 * .602
    line = font / 72 * 1.34
    capacity = int(width / step)
    source = case['user_text']
    start_y = y
    used_spans = excerpts(case)
    for si, (lo, hi) in enumerate(used_spans):
        if si:
            if sections and si == len(used_spans)-1:
                y += .11
                text(ax,x,y,'Question',8.4,color=MUTED)
                y += .20
            else:
                text(ax, x, y + .005, '···', 9.4, color=MUTED)
                y += .15 if sections else .20
        groups = []
        for match in re.finditer(r'\S+\s*', source[lo:hi]):
            groups.append(list(range(lo + match.start(), lo + match.end())))
        lines, current = [], []
        for group in groups:
            if current and len(current) + len(group) > capacity:
                lines.append(current)
                current = []
            current.extend(group)
        if current:
            lines.append(current)
        for chars in lines:
            # Whitespace remains in the text but not as a colored leading block.
            while chars and source[chars[-1]].isspace():
                chars.pop()
            runs, run, last = [], [], None
            for j in chars:
                assert len(owners[j]) == 1, ('Displayed character spans multiple tokens', j)
                owner = owners[j][0]
                if owner != last and run:
                    runs.append((last, run))
                    run = []
                last = owner
                run.append(j)
            if run:
                runs.append((last, run))
            col = 0
            for owner, run in runs:
                token = case['tokens'][owner]
                color = CMAP(norm(token['score'])) if token['eligible'] else (1, 1, 1, 1)
                ax.add_patch(Rectangle((x + col * step, y), len(run) * step, line * .91,
                                       facecolor=color, edgecolor='none'))
                string = ''.join(source[j] for j in run)
                text(ax, x + col * step, y + .012, string, font,
                     fontfamily='DejaVu Sans Mono')
                if records is not None:
                    records.append({'model':case['model'], 'token_index':owner,
                                    'char_span':[run[0],run[-1]+1], 'score':token['score'],
                                    'eligible':token['eligible'], 'text':string,
                                    'rgba':list(color),
                                    'bounds':[x+col*step,y,len(run)*step,line*.91]})
                col += len(run)
            y += line
    return y - start_y, used_spans


def all_input_strip(ax, case, x, y, width, norm, spans):
    tokens = case['tokens']
    w = width / len(tokens)
    for j, t in enumerate(tokens):
        color = CMAP(norm(t['score'])) if t['eligible'] else 'white'
        ax.add_patch(Rectangle((x + j*w, y), w + .0001, .065, facecolor=color, edgecolor='none'))
    for lo, hi in spans:
        ids = [j for j,t in enumerate(tokens) if t['char_span'][0] < hi and t['char_span'][1] > lo]
        ax.plot([x + min(ids)*w, x + (max(ids)+1)*w], [y + .105, y + .105], color=INK, lw=1.1)
    text(ax, x, y + .15, 'start', 7.6, color=MUTED)
    text(ax, x + width, y + .15, 'end', 7.6, color=MUTED, ha='right')


def case_figure_compact(cases, dataset):
    """Paper subpanels with aligned token evidence and a compact deletion plot."""
    rows = [next(c for c in cases if c['dataset']==dataset and c['model']==f)
            for f in ('qwen3','qwen35')]
    fig = plt.figure(figsize=(8.8,6),facecolor='white')
    fig._case_text_artists=[]
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,8.8); ax.set_ylim(6,0); ax.axis('off')
    max_score=max(abs(t['score']) for c in rows for t in c['tokens'] if t['eligible'])
    norm=Normalize(vmin=-max_score,vmax=max_score)
    width=3.12
    columns=(.10,3.44)
    bottoms=[]; spans=[]; runs=[]
    for i,case in enumerate(rows):
        x=columns[i]
        text(ax,x,.09,('(a) Qwen3-8B','(b) Qwen3.5-9B')[i],9.8,'bold')
        text(ax,x,.42,'Context',8.4,color=MUTED)
        used,span=heatmap(ax,case,x,.64,width,norm,font=10.0,sections=True,records=runs)
        bottoms.append(.64+used);spans.append(span)
    strip_label=max(bottoms)+.17
    strip_y=strip_label+.20
    for i,case in enumerate(rows):
        text(ax,columns[i],strip_label,'Full input',8.2,color=MUTED)
        all_input_strip(ax,case,columns[i],strip_y,width,norm,spans[i])
    bar_y=4.53
    height=max(strip_y+.35,bar_y+.35)
    fig.set_size_inches(8.8,height)
    ax.set_ylim(height,0)
    ax.plot([6.72,6.72],[.08,strip_y+.27],color=LINE,lw=.55)

    right=6.94
    text(ax,right,.09,'(c) Deletion',9.8,'bold')
    text(ax,right,.42,'Answer',8.4,color=MUTED)
    if dataset=='niah_mq_q2':
        answer='5443951, 8698256'
        text(ax,right,.65,answer,9.3,fontfamily='DejaVu Sans Mono')
    else:
        answer='6 consonants in Jennings'
        text(ax,right,.65,'6 consonants\nin Jennings',9.3)
    cax=fig.add_axes([7.25/8.8,(height-2.70)/height,1.42/8.8,1.40/height])
    curve_data={}
    model_colors=['#397B9D','#94678F']
    for i,case in enumerate(rows):
        eligible=sum(t['eligible'] for t in case['tokens'])
        curve_data[case['model']]={}
        for method,key in [('DT','deletion'),('FT','deletion_ft')]:
            xs=np.asarray([len(d)/eligible for d in case[key]['deleted_user_indices']])
            ys=np.asarray(case[key]['normalized_model_response'])
            assert np.min(ys)>=-1e-12 and np.max(ys)<=1+1e-12
            curve_data[case['model']][method]={'removed_fraction':xs.tolist(),'normalized_score':ys.tolist(),
                'full_response_log_likelihood':case[key]['scores']}
            cax.plot(xs,ys,color=model_colors[i],lw=1.15,
                     linestyle='-' if method=='DT' else '--',
                     marker='o' if method=='DT' else None,markersize=1.7)
    cax.set_xlim(0,1);cax.set_ylim(-.035,1.035)
    cax.set_xticks([0,.5,1],['0','50','100'])
    cax.set_yticks([0,.5,1],['0','0.5','1'])
    cax.tick_params(labelsize=8.1,length=2,pad=2,colors=MUTED)
    cax.spines[['top','right']].set_visible(False)
    cax.spines[['bottom','left']].set_color(LINE)
    cax.set_xlabel('Input removed (%)',fontsize=8.2,labelpad=3)
    cax.set_ylabel('Normalized log-likelihood\nof full response',fontsize=7.4,labelpad=3,linespacing=1.15)
    cax.grid(axis='y',color=LINE,lw=.5)
    for i,model in enumerate(['Qwen3-8B','Qwen3.5-9B']):
        for j,method in enumerate(['DT','FT']):
            y=3.20+(i*2+j)*.22
            ax.plot([right,right+.23],[y+.055,y+.055],color=model_colors[i],lw=1.15,
                    linestyle='-' if method=='DT' else '--')
            text(ax,right+.30,y,f'{model} {method}',7.8,color=MUTED)

    center=right+.80
    text(ax,center,bar_y-.34,'Signed contribution\n(nats)',8.2,color=MUTED,ha='center')
    bar=fig.add_axes([(center-.77)/8.8,(height-bar_y-.085)/height,1.54/8.8,.085/height])
    bar.imshow(np.linspace(-1,1,512)[None,:],aspect='auto',cmap=CMAP,vmin=-1,vmax=1,
               extent=[-1,1,0,1])
    bar.set_xticks([-1,0,1],[f'{-max_score:.1f}','0',f'+{max_score:.1f}'])
    bar.tick_params(axis='x',labelsize=8.1,length=0,pad=2,colors=MUTED)
    bar.set_yticks([])
    for spine in bar.spines.values():spine.set_visible(False)
    meta={'dataset':dataset,'index':0,'vmin':-max_score,'vmax':max_score,
          'normalization':'linear, centered on zero', 'color_map':SIGNED_COLORS,
          'figure_inches':[8.8,height],
          'color_legend_position':'right column, below deletion curve and model legend',
          'panel_labels':['(a) Qwen3-8B','(b) Qwen3.5-9B','(c) Deletion'],
          'global_title':False, 'visible_development_identifier':False,
          'aligned_model_columns':{'x':list(columns),'width':width,'text_y':.64,'strip_y':strip_y},
          'text_excerpt_char_spans':{c['model']:s for c,s in zip(rows,spans)},
          'model_input_hashes':{c['model']:c['input_sha256'] for c in rows},
          'fixed_target':'entire stored response plus EOS; Answer summarizes only the answer',
          'answer_summary':answer, 'displayed_runs':runs,
          'deletion_curve':'saved original normalized_model_response for each model and method; released clipped cumulative-minimum normalization, with no additional smoothing',
          'deletion_baseline':'one-hop FlashTrace on the same model, input, and full fixed response plus EOS',
          'deletion_y_axis':'Normalized log-likelihood of full response',
          'deletion_series':curve_data,
          'scope':'development case illustration'}
    fig._case_meta=meta
    return fig,meta



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--overview-only', action='store_true',
                        help='Rebuild the main figure and preserve existing case assets.')
    selection.add_argument('--cases-only', action='store_true',
                           help='Rebuild the two case figures and preserve existing overviews.')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    data_path = HERE / 'data/cases.json'
    data = json.loads(data_path.read_bytes())
    overview_path = HERE / 'data/overview_role_case.json'
    overview_data = json.loads(overview_path.read_bytes())
    previous=json.loads((HERE/'figure_manifest.json').read_bytes())
    figs = []
    lookup_path=HERE/'data/overview_case.json'
    if args.cases_only:
        mechanism_meta=previous['mechanism_case']
    else:
        figs.append(('deltatrace-mechanism', mechanism_3d(overview_data)))
        figs.append(('deltatrace-lookup-overview',mechanism_3d(json.loads(lookup_path.read_bytes()))))
        mechanism_meta=figs[0][1]._dt_case
    specs = {}
    if args.overview_only:
        specs = json.loads((HERE/'figure_manifest.json').read_bytes())['cases']
    else:
        for ds, filename in [('niah_mq_q2','deltatrace-retrieval'), ('morehopqa','deltatrace-multihop')]:
            fig, meta = case_figure_compact(data['cases'], ds)
            figs.append((filename, fig)); specs[filename] = meta
    packet = PAPER / 'output/pdf/deltatrace-figures.pdf'
    for name, fig in figs:
        fig.savefig(OUT / (name+'.pdf'), metadata={'CreationDate':None, 'Author':''})
        svg = OUT / (name+'.svg')
        fig.savefig(svg)
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines())+'\n',
                       encoding='utf8', newline='\n')
        fig.savefig(OUT / (name+'.png'), dpi=600)
        plt.close(fig)
    # Concatenate the actual vector assets, keeping untouched case figures byte-stable.
    from pypdf import PdfReader, PdfWriter
    writer=PdfWriter()
    for name in ('deltatrace-mechanism','deltatrace-retrieval','deltatrace-multihop'):
        writer.add_page(PdfReader(OUT/(name+'.pdf')).pages[0])
    writer.add_metadata({'/Title':'DeltaTrace: mechanism and signed evidence','/Author':''})
    with packet.open('wb') as stream: writer.write(stream)
    (PAPER/'output/pdf/deltatrace-overview.pdf').write_bytes((OUT/'deltatrace-mechanism.pdf').read_bytes())
    for name in ('deltatrace-retrieval','deltatrace-multihop'):
        (PAPER/'output/pdf'/(name+'.pdf')).write_bytes((OUT/(name+'.pdf')).read_bytes())
    manifest = {
        'fixture_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'overview_fixture_sha256': hashlib.sha256(overview_path.read_bytes()).hexdigest(),
        'lookup_fixture_sha256': hashlib.sha256(lookup_path.read_bytes()).hexdigest(),
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'mechanism_builder_sha256': hashlib.sha256((HERE/'draw_mechanism.py').read_bytes()).hexdigest(),
        'mechanism': 'Wider paired forward/reverse trace on the left; equally sized Attention and GDN panels stacked on the right, with each content/control allocation to the right of its operator; actual signed playwright/composer contrast. A direct lookup overview is also supplied.',
        'mechanism_case': mechanism_meta,
        'cases': specs, 'generated_files': {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted(OUT.iterdir()) if p.suffix in ('.pdf','.svg','.png')}}
    (HERE / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'figures': len(figs), 'packet':str(packet), 'fixture_verified':True}))


if __name__ == '__main__':
    main()
