"""A real signed-token example anchors the explanation of finite propagation.

Only the input heatmap is measured. The single reverse arrow and the local
attention identity describe the method; neither represents a measured edge.
"""
import math
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

INK = '#25323B'
MUTED = '#75818A'
BLUE = '#507FA6'
PURPLE = '#8270A5'
SIGNED_COLORS = ['#D5816C', '#FFFFFF', '#4F9C87']


def draw(data):
    case = next(c for c in data['cases'] if c['model'] == 'qwen35' and c['dataset'] == 'morehopqa')
    paired = [c for c in data['cases'] if c['dataset'] == 'morehopqa']
    limit = max(abs(t['score']) for c in paired for t in c['tokens'] if t['eligible'])
    norm = Normalize(vmin=-limit, vmax=limit)
    cmap = LinearSegmentedColormap.from_list('dt_signed', SIGNED_COLORS)
    source = case['user_text']
    target_excerpt = 'Therefore, the number of consonants in the last name "Jennings" is 6.'
    assert target_excerpt in case['target']

    fig = plt.figure(figsize=(10, 6.7), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10); ax.set_ylim(6.7, 0); ax.axis('off')

    def text(x, y, value, size=12, color=INK, weight='normal', **kw):
        family = kw.pop('fontfamily', 'DejaVu Sans')
        return ax.text(x, y, value, va='top', fontsize=size, color=color,
                       fontweight=weight, fontfamily=family, **kw)

    # Reading order: the response, one reverse traversal, then the actual input.
    text(.43, .15, 'Which words shaped this response?', 17, weight='bold')
    ax.add_patch(FancyBboxPatch((.45, .62), 6.25, .94,
                 boxstyle='round,pad=0.035,rounding_size=0.10',
                 facecolor='#FCF7EB', edgecolor='#C4A56B', linewidth=.85))
    text(.66, .75, 'RESPONSE EXCERPT', 9.5, '#9A8054', 'bold')
    text(.66, .98, 'Therefore, the number of consonants', 13.1)
    text(.66, 1.23, 'in the last name "Jennings" is 6.', 13.1, weight='bold')
    text(.51, 1.80, 'DT follows the fixed response\'s score change', 11.4, MUTED)
    text(.51, 2.02, 'back through the model to the input words.', 11.4, MUTED)
    ax.add_patch(FancyArrowPatch((5.91, 1.60), (5.91, 2.52),
                 arrowstyle='-|>', mutation_scale=16, color=PURPLE,
                 linewidth=2.3, zorder=8))
    text(6.08, 1.91, r'$\Delta F$', 14, PURPLE)

    # A single sheet presents measured token scores, not hidden intermediate views.
    x0, y0, width, height = 1.01, 2.42, 5.90, 3.15
    def project(u, v):
        return (x0 + u - .25*v, y0 + .07*u + v)
    corners = [project(0, 0), project(width, 0), project(width, height), project(0, height)]
    ax.add_patch(Polygon([(x+.035,y+.095) for x,y in corners], closed=True,
                 facecolor='#92ABB5', edgecolor='none', alpha=.16, zorder=1))
    near = [corners[3], corners[2], (corners[2][0],corners[2][1]+.047),
            (corners[3][0],corners[3][1]+.047)]
    ax.add_patch(Polygon(near, closed=True, facecolor='#C8D9DF',
                 edgecolor='#B5C7CF', linewidth=.55, zorder=2))
    ax.add_patch(Polygon(corners, closed=True, facecolor='#FEFFFF',
                 edgecolor='#B6C5CC', linewidth=.85, zorder=3))
    angle = -math.degrees(math.atan(.07))

    def ptext(u, v, value, size=12, color=INK, weight='normal', **kw):
        x,y = project(u,v)
        return text(x,y,value,size,color,weight,rotation=angle,
                    rotation_mode='anchor', zorder=6, **kw)

    ptext(.20, .13, 'Signed input contributions', 13.2, weight='bold')
    ptext(5.65, .17, 'Qwen3.5', 9.5, MUTED, ha='right')
    owners = [[] for _ in source]
    for i,t in enumerate(case['tokens']):
        for j in range(*t['char_span']): owners[j].append(i)
    font, step, line_height = 12.1, 12.1 / 72 * .602, 12.1 / 72 * 1.30
    capacity = int(5.49 / step)
    spans = []

    def paragraph(value, v):
        lo = source.index(value); hi = lo + len(value)
        spans.append([lo,hi])
        groups = [list(range(lo+m.start(),lo+m.end())) for m in re.finditer(r'\S+\s*', value)]
        lines, current = [], []
        for group in groups:
            if current and len(current)+len(group)>capacity:
                lines.append(current); current=[]
            current.extend(group)
        if current: lines.append(current)
        for chars in lines:
            while chars and source[chars[-1]].isspace(): chars.pop()
            runs, run, last = [], [], None
            for j in chars:
                assert len(owners[j])==1, ('Ambiguous displayed character',j)
                owner=owners[j][0]
                if owner!=last and run: runs.append((last,run)); run=[]
                run.append(j); last=owner
            if run: runs.append((last,run))
            col=0
            for owner, chars in runs:
                token=case['tokens'][owner]
                u=.21+col*step; w=len(chars)*step
                points=[project(u,v),project(u+w,v),project(u+w,v+line_height*.91),project(u,v+line_height*.91)]
                ax.add_patch(Polygon(points,closed=True,
                     facecolor=cmap(norm(token['score'])) if token['eligible'] else 'white',
                     edgecolor='none',zorder=4))
                ptext(u,v+.012,''.join(source[j] for j in chars),font,fontfamily='DejaVu Sans Mono')
                col+=len(chars)
            v+=line_height
        return v

    ptext(.21,.49,'CONTEXT',9.1,MUTED,weight='bold')
    v=paragraph('Townsend Putnam Coleman III',.70)
    ptext(.21,v+.005,'…',11,MUTED); v+=.18
    v=paragraph('he also did additional voices in films "Fantasia 2000" (1999) and "Sing" (2016)',v)
    ptext(.21,v+.005,'…',11,MUTED); v+=.18
    v=paragraph('It was directed and written by Garth Jennings',v)
    v+=.20
    ptext(.21,v,'QUESTION',9.1,MUTED,weight='bold'); v+=.21
    v=paragraph('How many consonants are there in the last name of the person who directed and wrote the 2016 film featuring the voice of Townsend Coleman?',v)
    assert v<height-.09,(v,height)

    # A compact operator lens explains the mechanism beside the example.
    # Color here distinguishes operator roles; it is not a measured attribution.
    text(7.32,.22,'Content and selection',14.1,weight='bold')
    text(7.32,.50,'both receive credit',14.1,weight='bold')
    text(7.32,1.05,r'$\Delta(PV) =$',17)
    text(7.32,1.47,r'$P_1\,\Delta V$',17,BLUE)
    text(8.58,1.47,r'$+$',17,MUTED)
    text(8.91,1.47,r'$\Delta P\,V_0$',17,PURPLE)

    text(7.32,2.12,'Content carried',12.5,BLUE,'bold')
    text(7.32,2.47,'"Jennings"',15.0,weight='bold')
    text(7.32,2.83,'The name supplies information.\nIts change travels with the\nweight that selected it.',10.6,linespacing=1.45)

    text(7.32,3.68,'Selection changed',12.5,PURPLE,'bold')
    text(7.32,4.03,'"last name"',15.0,weight='bold')
    text(7.32,4.39,'The question guides what is used.\nA change in that selection\nalso receives a contribution.',10.6,linespacing=1.45)

    text(7.32,5.38,'Contributions add up',11.7,weight='bold')
    text(7.32,5.76,r'$\sum_i A_i = \Delta F$',18.0,PURPLE)
    text(7.32,6.26,'across all input sources',10.2,MUTED)

    # One quantitative legend; no decorative rules or word underlines.
    cax=fig.add_axes([.191,.024,.358,.014])
    cax.imshow(np.linspace(-limit,limit,512)[None,:],aspect='auto',cmap=cmap,norm=norm,
               extent=[-limit,limit,0,1])
    cax.set_xticks([-limit,0,limit],[f'{-limit:.1f}','0',f'{limit:.1f}'])
    cax.tick_params(axis='x',labelsize=8.9,length=0,pad=2)
    cax.set_yticks([])
    for spine in cax.spines.values(): spine.set_visible(False)
    text(.34,6.40,'Lowers score',10.4,'#BC6B59')
    text(6.90,6.40,'Raises score',10.4,'#408C78',ha='right')
    text(3.70,6.22,'Token contribution · nats',9.5,MUTED,ha='center')

    fig._dt_case = {'model':case['model'],'dataset':case['dataset'],'index':case['index'],
        'input_sha256':case['input_sha256'],'text_excerpt_char_spans':spans,
        'target_excerpt':target_excerpt,'target_excerpt_exact_match':True,
        'fixed_target':'entire stored response plus EOS',
        'vmin':-limit,'vmax':limit,'normalization':'linear, centered on zero',
        'color_map':SIGNED_COLORS,'shared_scale':'both models in the full multi-hop case',
        'schematic_elements':'one reverse-traversal arrow, attention product identity, and contribution conservation identity',
        'perspective':'one sheet displaying input token scores; no intermediate-state scores'}
    return fig
