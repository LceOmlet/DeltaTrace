"""A layered vector view of DT's two executions and backward attribution.

The planes and internal paths are schematic. Terminal span scores are the
stored Qwen3.5 multi-hop example, with no measured internal edge weights.
"""
import math

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

INK = '#263645'
GRAY = '#8392A0'
LINE = '#D8E1E8'
BLUE = '#477FC5'
PURPLE = '#8065AE'
POS = '#238778'
NEG = '#CA6854'


def draw(data):
    fig = plt.figure(figsize=(8.8, 5.15), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 8.8); ax.set_ylim(5.15, 0); ax.axis('off')

    def text(x, y, s, size=12, color=INK, weight='normal', **kw):
        return ax.text(x, y, s, va='top', fontsize=size, color=color,
                       fontweight=weight, fontfamily='DejaVu Sans', **kw)

    def arrow(a, b, color=GRAY, width=1.3, rad=0, **kw):
        p = FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=11,
                            linewidth=width, color=color,
                            connectionstyle=f'arc3,rad={rad}', **kw)
        ax.add_patch(p); return p

    def line(points, color=LINE, width=1, **kw):
        xs, ys = zip(*points); ax.plot(xs, ys, color=color, lw=width, **kw)

    def plane(x, y, w, h, fill='#F6FAFD', edge='#A9BDCE', paired=True, back_label=''):
        """An oblique plane; objects use the same in-plane coordinates."""
        def project(u, v): return (x + u - .38*v, y + .085*u + v)
        points = [project(0, 0), project(w, 0), project(w, h), project(0, h)]
        # Short extrusion and a restrained shadow preserve the white page.
        ax.add_patch(Polygon([(a+.025,b+.068) for a,b in points], closed=True,
                             facecolor='#DDE6EC', edgecolor='none', alpha=.60, zorder=1))
        if paired:
            ghost = [(a+.12,b-.19) for a,b in points]
            ax.add_patch(Polygon(ghost, closed=True, facecolor='#FAFBFC',
                                 edgecolor='#CDD5DC', linewidth=.8, zorder=2))
            if back_label:
                text(x+.26,y-.15,back_label,9.9,'#98A4AF',
                     rotation=-math.degrees(math.atan(.085)),rotation_mode='anchor',zorder=2.8)
        ax.add_patch(Polygon(points, closed=True, facecolor=fill, edgecolor=edge,
                             linewidth=.95, zorder=3))
        # Front thickness is visible only along the near edge.
        front = [points[3], points[2], (points[2][0],points[2][1]+.045),
                 (points[3][0],points[3][1]+.045)]
        ax.add_patch(Polygon(front, closed=True, facecolor='#D8E6EE',
                             edgecolor=edge, linewidth=.45, zorder=3))
        return project

    def ptext(project, u, v, s, size=11.5, **kw):
        x, y = project(u, v)
        return text(x, y, s, size=size, rotation=-math.degrees(math.atan(.085)),
                    rotation_mode='anchor', zorder=6, **kw)

    def grid(x, y, rows, cols, color, w=.42, h=.30, highlighted=None, alpha=.20):
        for r in range(rows):
            for c in range(cols):
                a = .70 if highlighted is not None and (r,c) in highlighted else alpha
                ax.add_patch(Rectangle((x+c*w/cols, y+r*h/rows), w/cols, h/rows,
                                       facecolor=color, edgecolor='white', linewidth=.55,
                                       alpha=a, zorder=6))
        ax.add_patch(Rectangle((x,y), w,h, facecolor='none', edgecolor=color,
                               linewidth=.55, zorder=6))

    # Three visual zones, one shared baseline and modest typographic hierarchy.
    text(.15, .12, 'Local change', 14, weight='bold')
    text(2.64, .12, 'Following the evidence', 14, weight='bold')
    text(7.03, .12, 'Signed sources', 14, weight='bold')
    line([(.15,.43),(2.11,.43)])
    line([(2.64,.43),(6.59,.43)])
    line([(7.03,.43),(8.64,.43)])

    # Local attention change: two visibly different routes, one output sum.
    text(.15, .61, 'Attention', 12.5, weight='bold')
    text(.15, .88, 'Content carried', 11.5, BLUE, 'bold')
    grid(.22, 1.16, 3, 3, BLUE, highlighted={(0,0),(1,1),(2,2)})
    text(.42, 1.52, r'$P_1$', 13, ha='center')
    text(.78, 1.19, '×', 15, BLUE, ha='center')
    grid(.98, 1.16, 3, 1, BLUE, w=.20, h=.30, alpha=.55)
    text(1.08, 1.52, r'$\Delta V$', 13, ha='center')
    arrow((1.29,1.30),(1.70,1.62),BLUE,1.7,rad=-.15)
    text(.15, 1.91, 'Selection', 11.5, PURPLE, 'bold')
    grid(.22, 2.20, 3, 3, PURPLE, highlighted={(0,1),(1,2),(2,0)})
    text(.42, 2.55, r'$\Delta P$', 13, ha='center')
    text(.78, 2.22, '×', 15, PURPLE, ha='center')
    grid(.98, 2.20, 3, 1, '#8D9AA7', w=.20, h=.30, alpha=.45)
    text(1.08, 2.55, r'$V_0$', 13, ha='center')
    arrow((1.29,2.34),(1.70,1.86),PURPLE,1.7,rad=.13)
    ax.add_patch(Circle((1.80,1.74),.135,facecolor='white',edgecolor='#859BAC',lw=1.0,zorder=8))
    text(1.80,1.635,'+',15,ha='center',zorder=9)
    text(1.80,1.37,r'$\Delta O$',13,ha='center')
    # A short relation explicitly ties the decomposition to the large model view.
    line([(2.03,1.74),(2.36,1.74),(2.36,2.43),(2.83,2.43)], '#B4C2CE', .9,
         linestyle=(0,(2.5,2.5)))
    text(.15, 3.07, 'Memory', 12.5, weight='bold')
    text(.15, 3.34, 'Evidence carried forward.', 10.8, GRAY)
    # Memory glyph: retained state plus a gated new write.
    for x,label in [(.30,r'$S_{t-1}$'),(1.70,r'$S_t$')]:
        ax.add_patch(Polygon([(x-.16,3.91),(x+.13,3.91),(x+.22,3.82),(x-.07,3.82)],
                             facecolor='#E8F0F6',edgecolor='#9FB3C4',lw=.8))
        ax.add_patch(Rectangle((x-.16,3.91),.29,.27,facecolor='#F5F8FA',edgecolor='#9FB3C4',lw=.8))
        ax.add_patch(Polygon([(x+.13,3.91),(x+.22,3.82),(x+.22,4.09),(x+.13,4.18)],
                             facecolor='#CEDDE7',edgecolor='#9FB3C4',lw=.8))
        text(x,4.28,label,12,ha='center')
    arrow((.52,4.04),(1.44,4.04),BLUE,1.8)
    ax.add_patch(Circle((.88,4.04),.085,facecolor='white',edgecolor=PURPLE,lw=1,zorder=8))
    text(.88,3.955,'×',11,PURPLE,ha='center',zorder=9)
    text(.88,3.62,'keep',11,PURPLE,ha='center')
    line([(.88,3.83),(.88,3.94)],PURPLE,1.0)
    ax.add_patch(Circle((1.23,4.04),.085,facecolor='white',edgecolor='#91A5B5',lw=1,zorder=8))
    text(1.23,3.955,'+',11,ha='center',zorder=9)
    arrow((1.23,4.48),(1.23,4.16),BLUE,1.3)
    text(1.23,4.60,'new write',11,ha='center')

    # Paired reference/original surfaces are activations in two executions.
    # They do not represent recursive attribution passes or CoT targets.
    top = plane(2.98,.82,3.43,.66,fill='#F4F8FD',back_label='same fixed response')
    mid = plane(2.95,2.19,3.43,.84,fill='#F4F9FC',back_label='reference activations')
    bottom = plane(2.88,3.92,3.54,.67,fill='#F5F9FC')
    ptext(top,.14,.10,'Fixed response',12,weight='bold')
    ptext(top,.21,.37,'… Jennings has 6 consonants.',11.5)
    ptext(mid,.15,.03,'Attention and memory',12,weight='bold')
    ptext(bottom,.15,.08,'Input',12,weight='bold')
    ptext(bottom,.19,.30,'… Garth Jennings …',11.5)
    ptext(bottom,.19,.50,'… consonants … last name …',11.5)
    # Fine cue highlights follow the same plane geometry.
    for pr,u,v,w,c in [(bottom,.90,.45,.67,BLUE),(bottom,1.87,.66,.69,PURPLE)]:
        a=pr(u,v); b=pr(u+w,v); line([a,b],c,2.4,zorder=7)

    # A simple fan-in motif gives attention a visual, rather than verbal, form.
    p0=mid(.48,.54); p1=mid(.99,.54); p2=mid(1.50,.54)
    dst=mid(1.14,.36)
    for j,p in enumerate([p0,p1,p2]):
        ax.add_patch(Rectangle((p[0]-.075,p[1]-.015),.15,.11,
                               facecolor=['#D7E5F3','#89B0DD','#D7E5F3'][j],edgecolor=BLUE,lw=.5,zorder=7))
        arrow((p[0],p[1]-.03),dst,BLUE,.8+(.7 if j==1 else 0),zorder=7)
    ptext(mid,.28,.72,'carry / select',10.7,color=BLUE)
    # A compact state block depicts persistence and a later read.
    q=mid(2.63,.49)
    ax.add_patch(Polygon([(q[0]-.20,q[1]),(q[0]+.17,q[1]),(q[0]+.28,q[1]-.11),(q[0]-.09,q[1]-.11)],
                         facecolor='#CDDFED',edgecolor=BLUE,lw=.6,zorder=7))
    ax.add_patch(Rectangle((q[0]-.20,q[1]),.37,.16,facecolor='#E8F0F7',edgecolor=BLUE,lw=.6,zorder=7))
    arrow((q[0]-.47,q[1]+.07),(q[0]-.22,q[1]+.07),PURPLE,1.2,zorder=7)
    arrow((q[0]+.18,q[1]+.07),(q[0]+.46,q[1]+.07),BLUE,1.2,zorder=7)
    ptext(mid,2.12,.72,'retain / read',10.7,color=BLUE)

    # Only a few continuous paths are emphasized across the surfaces.
    # Width is stylistic and carries no numerical meaning.
    t=top(3.10,.50); a=mid(1.14,.36); m=mid(2.66,.49)
    model_edge=mid(3.10,-.02)
    arrow((t[0],t[1]+.08),model_edge,BLUE,2.15,rad=-.08,zorder=10)
    text(6.18,1.88,r'$\Delta F$',12.5,BLUE)
    b_content=bottom(1.36,.38); b_cue=bottom(2.53,.57)
    arrow(mid(.99,.94),(b_content[0],b_content[1]-.26),BLUE,2.15,rad=.16,zorder=10)
    arrow(mid(1.48,.96),(b_cue[0]-.13,b_cue[1]-.14),PURPLE,1.65,rad=-.05,zorder=10)
    memory_path=arrow(mid(2.66,.94),(b_content[0]+.07,b_content[1]-.25),BLUE,1.45,rad=-.10,zorder=10)
    memory_path.set_path_effects([path_effects.Stroke(linewidth=3.5,foreground='white'),path_effects.Normal()])
    # Path labels occupy open gaps between planes.
    text(2.73,3.51,'content',11.2,BLUE,weight='bold')
    text(5.44,3.61,'selection',11.2,PURPLE,weight='bold')
    text(5.74,3.43,'memory',10.8,BLUE)
    # The original/reference legend labels the paired geometry, not a third step.
    ax.add_patch(Rectangle((2.78,4.94),.14,.08,facecolor='#FAFBFC',edgecolor='#CDD5DC',lw=.7))
    text(2.99,4.90,'EOS reference',10.8,GRAY)
    ax.add_patch(Rectangle((4.60,4.94),.14,.08,facecolor='#D8E6EE',edgecolor='#A9BDCE',lw=.7))
    text(4.81,4.90,'original input',10.8,GRAY)

    # True terminal contributions from the fixed input, separate from schematic edges.
    case=next(c for c in data['cases'] if c['model']=='qwen35' and c['dataset']=='morehopqa')
    text(7.03,.64,'Qwen3.5 example',11.0,GRAY)
    text(7.03,.95,'Input words',11.8,weight='bold')
    words=['Garth','Jennings','last name','consonants']
    values=[]
    bars=[]
    zero=7.56
    scale=.088
    for i,word in enumerate(words):
        lo=case['user_text'].index(word); hi=lo+len(word)
        ts=[t for t in case['tokens'] if t['char_span'][0]<hi and t['char_span'][1]>lo]
        value=sum(t['score'] for t in ts); values.append(value)
        bars.append({'text':word,'char_span':[lo,hi],
                     'token_local_indices':[t['local_index'] for t in ts],
                     'signed_sum':value})
        y=1.40+i*.62
        text(7.03,y-.17,word,11.7)
        end=zero+value*scale
        ax.add_patch(Rectangle((min(zero,end),y+.12),abs(end-zero),.145,
                               facecolor=POS if value>=0 else NEG,edgecolor='none',zorder=5))
        text(8.62,y-.17,f'{value:+.2f}',10.8,POS if value>=0 else NEG,ha='right')
        line([(zero,y+.06),(zero,y+.32)],'#AAB7C2',.7,zorder=4)
    text(zero,3.63,'0',10.5,GRAY,ha='center')
    text(8.61,3.63,'nats',10.5,GRAY,ha='right')
    text(7.03,3.98,'+  raises the score',11.3,POS)
    text(7.03,4.24,'−  lowers the score',11.3,NEG)
    text(7.82,4.55,r'$\sum_i A_i=\Delta F$',14,ha='center')
    text(7.82,4.96,'over all sources',10.5,GRAY,ha='center')
    fig._dt_source_bars = {'model':case['model'],'dataset':case['dataset'],
                           'index':case['index'],'input_sha256':case['input_sha256'],
                           'aggregation':'sum of original token scores overlapping the named source span',
                           'bars':bars}
    return fig
