"""Vector DT mechanism figure and data-backed, aligned signed-token figures.

Run with matplotlib available. cases.json is a self-contained checked fixture.
PDF and SVG text remain vector; PNG files are viewing copies.
"""
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
    return ax.text(x, y, value, fontsize=size, fontweight=weight, color=color,
                   va='top', linespacing=1.25, **kwargs)


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


def heatmap(ax, case, x, y, width, norm, font=10.15, sections=False):
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
                text(ax,x,y,'QUESTION',9.8,color=MUTED)
                y += .23
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
                col += len(run)
            y += line
    return y - start_y, used_spans


def all_input_strip(ax, case, x, y, width, norm, spans):
    tokens = case['tokens']
    w = width / len(tokens)
    for j, t in enumerate(tokens):
        color = CMAP(norm(t['score'])) if t['eligible'] else 'white'
        ax.add_patch(Rectangle((x + j*w, y), w + .0001, .10, facecolor=color, edgecolor='none'))
    for lo, hi in spans:
        ids = [j for j,t in enumerate(tokens) if t['char_span'][0] < hi and t['char_span'][1] > lo]
        ax.plot([x + min(ids)*w, x + (max(ids)+1)*w], [y + .15, y + .15], color=INK, lw=1.8)
    text(ax, x, y + .20, 'input start', 8.4, color=MUTED)
    text(ax, x + width, y + .20, 'input end', 8.4, color=MUTED, ha='right')


def case_figure_compact(cases, dataset):
    """Aligned evidence/question panels with a small, distinct response view."""
    rows = [next(c for c in cases if c['dataset']==dataset and c['model']==f)
            for f in ('qwen3','qwen35')]
    fig = plt.figure(figsize=(8.8,6),facecolor='white')
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,8.8); ax.set_ylim(6,0); ax.axis('off')
    title = ('Retrieval: numbers and the keys that select them' if dataset=='niah_mq_q2'
             else 'Multi-hop reasoning: the name and the question about it')
    text(ax,.15,.10,title,13.3,'bold')
    text(ax,8.65,.16,'DT · development example 0',9.5,color=MUTED,ha='right')
    max_score=max(abs(t['score']) for c in rows for t in c['tokens'] if t['eligible'])
    norm=Normalize(vmin=-max_score,vmax=max_score)
    width=3.10
    bottoms=[]; spans=[]
    for i,case in enumerate(rows):
        x=.16+i*3.39
        text(ax,x,.58,'Qwen3-8B' if i==0 else 'Qwen3.5-9B',12.1,'bold')
        text(ax,x,.94,'CONTEXT',9.8,color=MUTED)
        used,span=heatmap(ax,case,x,1.19,width,norm,font=10.9,sections=True)
        bottoms.append(1.19+used);spans.append(span)
    baseline=max(bottoms)+.20
    text(ax,.16,baseline,'Complete input · marked spans are enlarged above',9.6,color=MUTED)
    for i,case in enumerate(rows):
        all_input_strip(ax,case,.16+i*3.39,baseline+.25,width,norm,spans[i])
    legend_y=max(baseline+.86,4.25)
    height=legend_y+.52
    fig.set_size_inches(8.8,height)
    ax.set_ylim(height,0)
    for x in (3.405,6.805):
        ax.plot([x,x],[.57,baseline+.54],color='#E6EBEF',lw=.6)

    # This compact target panel carries no attribution heatmap: y is fixed.
    right=7.00
    text(ax,right,.59,'FIXED RESPONSE',9.8,'bold',MUTED)
    if dataset=='niah_mq_q2':
        text(ax,right+.14,.88,'5443951\n8698256',15.2,'bold',VIOLET)
    else:
        text(ax,right+.14,.88,'6 consonants',14.0,'bold',VIOLET)
        text(ax,right+.14,1.24,'in “Jennings”',11.4,color=INK)
    text(ax,right,1.63,'Answer summary',9.8,color=MUTED)
    text(ax,right,1.99,'Score under deletion',10.8,'bold')
    cax=fig.add_axes([7.22/8.8,(height-3.32)/height,1.42/8.8,.98/height])
    for i,case in enumerate(rows):
        eligible=sum(t['eligible'] for t in case['tokens'])
        xs=np.asarray([len(d)/eligible for d in case['deletion']['deleted_user_indices']])
        ys=np.asarray(case['deletion']['normalized_model_response'])
        assert np.min(ys)>=-1e-12 and np.max(ys)<=1+1e-12
        cax.plot(xs,ys,color='#8393A1' if i==0 else VIOLET,
                 lw=1.15,linestyle='--' if i==0 else '-',
                 marker='o' if i==0 else None,markersize=1.6)
    cax.set_xlim(0,1);cax.set_ylim(-.04,1.04)
    cax.set_xticks([0,.5,1],['0','50','100'])
    cax.set_yticks([0,1],['0','1'])
    cax.tick_params(labelsize=9.0,length=2,pad=2,colors=MUTED)
    cax.spines[['top','right']].set_visible(False)
    cax.spines[['bottom','left']].set_color(LINE)
    cax.set_xlabel('input removed (%)',fontsize=9.1,labelpad=2)
    cax.set_ylabel('normalized score',fontsize=9.1,labelpad=2)
    cax.grid(axis='y',color=LINE,lw=.55)
    for j,(label,color,style) in enumerate([('Qwen3-8B','#8393A1','--'),('Qwen3.5-9B',VIOLET,'-')]):
        y=3.80+j*.23
        ax.plot([right,right+.25],[y+.06,y+.06],color=color,lw=1.25,linestyle=style)
        text(ax,right+.34,y,label,9.8,color=MUTED)

    # One signed scale covers both complete inputs, with zero at the center.
    text(ax,.16,legend_y,'− lowers the score',10.3,color=NEG)
    text(ax,8.64,legend_y,'+ raises the score',10.3,color=POS,ha='right')
    bar=fig.add_axes([3.02/8.8,(height-legend_y-.14)/height,2.80/8.8,.10/height])
    bar.imshow(np.linspace(-1,1,512)[None,:],aspect='auto',cmap=CMAP,vmin=-1,vmax=1,
               extent=[-1,1,0,1])
    bar.set_xticks([-1,0,1],[f'{-max_score:.1f}','0',f'{max_score:.1f}'])
    bar.tick_params(axis='x',labelsize=9.3,length=0,pad=1)
    bar.set_yticks([])
    for spine in bar.spines.values():spine.set_visible(False)
    text(ax,4.42,legend_y+.33,'Signed contribution (nats) · shared linear scale',
         9.6,color=MUTED,ha='center')
    meta={'dataset':dataset,'index':0,'vmin':-max_score,'vmax':max_score,
          'normalization':'linear, centered on zero',
          'color_map':SIGNED_COLORS,
          'text_excerpt_char_spans':{c['model']:s for c,s in zip(rows,spans)},
          'model_input_hashes':{c['model']:c['input_sha256'] for c in rows},
          'fixed_target':'entire stored response plus EOS; card summarizes only the answer',
          'deletion_curve':'saved original normalized_model_response; no smoothing',
          'scope':'development case illustration'}
    return fig,meta


def main():
    OUT.mkdir(exist_ok=True)
    data_path = HERE / 'data/cases.json'
    data = json.loads(data_path.read_bytes())
    figs = [('deltatrace-mechanism', mechanism_3d(data))]
    specs = {}
    for ds, filename in [('niah_mq_q2','deltatrace-retrieval'), ('morehopqa','deltatrace-multihop')]:
        fig, meta = case_figure_compact(data['cases'], ds)
        figs.append((filename, fig)); specs[filename] = meta
    packet = PAPER / 'output/pdf/deltatrace-figures.pdf'
    with PdfPages(packet, metadata={'Title':'DeltaTrace: mechanism and signed evidence', 'Author':''}) as pdf:
        for name, fig in figs:
            fig.savefig(OUT / (name+'.pdf'), metadata={'CreationDate':None, 'Author':''})
            svg = OUT / (name+'.svg')
            fig.savefig(svg)
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines())+'\n',
                           encoding='utf8', newline='\n')
            fig.savefig(OUT / (name+'.png'), dpi=190)
            pdf.savefig(fig)
            plt.close(fig)
    manifest = {
        'fixture_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'mechanism_builder_sha256': hashlib.sha256((HERE/'draw_mechanism.py').read_bytes()).hexdigest(),
        'mechanism': 'Actual Qwen3.5 input-token heatmap, exact response excerpt, one reverse path, and a local attention identity.',
        'mechanism_case': figs[0][1]._dt_case,
        'cases': specs, 'generated_files': {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted(OUT.iterdir()) if p.suffix in ('.pdf','.svg','.png')}}
    (HERE / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'figures': len(figs), 'packet':str(packet), 'fixture_verified':True}))


if __name__ == '__main__':
    main()
