"""Figure 1 top panels: full attribution and examples of backward calculations.

Coordinates are in points on the existing 14-inch figure canvas. These
panels are schematic and do not read or regenerate experiment data.
"""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Circle
from matplotlib.path import Path


def draw_overview_top(ax):
    ink, blue, purple = '#20252b', '#245c89', '#67448f'
    gray, line = '#657581', '#d4dfe7'
    texts = []

    def text(x, y, s, size=20.4, color=ink, weight='normal', ha='center', **kw):
        t = ax.text(x/72, y/72, s, fontsize=size, color=color,
                    fontfamily='Times New Roman', fontweight=weight,
                    ha=ha, va=kw.pop('va', 'baseline'), **kw)
        texts.append(t)
        return t

    def box(x, y, w, h, fc='white', ec=line, lw=1):
        ax.add_patch(FancyBboxPatch((x/72, y/72), w/72, h/72,
                     boxstyle='round,pad=0,rounding_size=0.05',
                     facecolor=fc, edgecolor=ec, linewidth=lw))

    def arrow(points, color=purple, lw=1.6, scale=11):
        path = Path([(x/72, y/72) for x, y in points],
                    [Path.MOVETO]+[Path.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>',
                     mutation_scale=scale, color=color, linewidth=lw,
                     joinstyle='round'))

    def plane(cx, y, label, fc, ec):
        x, w, h = cx-57.96, 115.92, 31.68
        points = [(x+7.2, y), (x+w, y), (x+w-7.2, y+h), (x, y+h)]
        ax.add_patch(Polygon([(a/72, b/72) for a, b in points],
                             facecolor=fc, edgecolor=ec, linewidth=1))
        text(cx, y+h/2, label, va='center')

    def vector(x, y, color):
        for j in range(3):
            ax.add_patch(Rectangle(((x+16*j)/72, y/72), 14/72, 12/72,
                                  facecolor=color, edgecolor='white', linewidth=.5))

    # (a) One complete path from paired inputs to token contributions.
    text(16.56, 31, '(a) Overall method', 27.5, weight='bold', ha='left')
    text(16.56, 55.1, r'Two forward runs with fixed response $y$', ha='left')
    left, right, reverse = 95.84, 346.24, 221.04
    text(left, 92, r'$F(x_0;y)$', 25)
    text(right, 92, r'$F(x_1;y)$', 25)
    text(reverse, 92, r'$\Delta F$', 26)
    text(168.48, 108, r'$-$', 21, gray)
    text(272.88, 108, '+', 21, gray)
    arrow([(154.8, 82.08), (191.52, 82.08)], gray, 1.2, 10)
    arrow([(287.28, 82.08), (250.56, 82.08)], gray, 1.2, 10)
    for cx, fc, ec in [(left, '#f5f7f9', '#bbc9d5'),
                       (right, '#ecf4fb', '#87a8c7')]:
        arrow([(cx, 217.44), (cx, 100.08)], '#a5b3bf', 2.2, 15)
        plane(cx, 118.08, 'Later layers', fc, ec)
        plane(cx, 169.2, r'Operation $f$', fc, ec)
        for yy in (154.08, 159.12, 164.16):
            ax.add_patch(Circle((cx/72, yy/72), .013, facecolor=gray,
                                edgecolor='none'))
    text(17.28, 163.44, 'Forward', 20.4, gray, rotation=90, va='center')
    text(424.8, 163.44, 'Forward', 20.4, gray, rotation=90, va='center')
    ax.plot([153.8/72, 288.28/72], [133.92/72, 133.92/72],
            color='#c2b3da', linewidth=1.1, linestyle=(0, (2.4, 2)))
    box(reverse-48, 147, 96, 74, '#f6f1fb', '#b9aacb', 1)
    text(reverse, 171, r'Build $D_f$', 20.4, purple)
    text(reverse, 204, r'Apply $D_f^{\top}$', 20.4, purple)
    arrow([(153.8,185.04),(170.04,185.04)], gray, 1.5, 7)
    arrow([(288.28,185.04),(272.04,185.04)], gray, 1.5, 7)
    # Align long reverse arrows above and below the local-rule node.
    arrow([(reverse, 108), (reverse, 144)], purple, 3.2, 16)
    arrow([(reverse, 224), (reverse, 261)], purple, 3.2, 18)
    text(reverse, 284, r'$m_{e_i}$', 24, purple)
    box(left-66.96, 221.76, 133.92, 31.68, '#f5f7f9', '#bcc9d4')
    box(right-66.96, 221.76, 133.92, 31.68, '#ecf4fb', '#86a6c3')
    text(left, 237.6, 'EOS EOS ...', 20.4, va='center')
    text(right, 237.6, 'source words', 20.4, blue, va='center')
    text(left, 281, r'Reference $x_0$')
    text(right, 281, r'Original $x_1$', color=blue)
    text(125, 311, r'Input change $\Delta e_i$', 21.5, blue, 'bold')
    text(335, 311, 'From reverse pass', 20.4, purple, 'bold')
    text(69, 336, 'Token')
    text(69, 358, 'vector')
    text(180, 336, 'EOS')
    text(180, 358, 'vector')
    vector(46, 371, '#8faed0')
    vector(157, 371, '#b6c1cb')
    text(125, 384, r'$-$', 24)
    text(335, 336, 'Weights in the')
    text(335, 358, 'score difference')
    vector(292, 371, '#b19bca')
    text(370, 384, r'$m_{e_i}$', 23, purple)
    arrow([(125, 390), (202, 407)], blue, 1.4)
    arrow([(335, 390), (266, 407)], purple, 1.4)
    text(227, 429, 'Multiply matching entries and sum')
    text(227, 457, r'Token score  $A_i=\langle m_{e_i},\Delta e_i\rangle$', 23)


    def matrix(cx, y, rows, cols, color, cell=10):
        w = cols*cell
        for row in range(rows):
            for col in range(cols):
                ax.add_patch(Rectangle(((cx-w/2+col*cell)/72,(y+row*cell)/72),
                             (cell-1.2)/72,(cell-1.2)/72,
                             facecolor=color,edgecolor='white',linewidth=.35))

    # (b) and (c) are local examples inside the complete reverse pass in (a).
    box(458, 10, 546, 452, '#fcfdff', '#b9c8d3', .9)
    text(474, 36, 'Reverse-pass examples', 25.5, weight='bold', ha='left')
    arrow([(737,28),(762,28)],gray,1.4,10)
    text(800,35,'Forward',20.4,gray)
    arrow([(862,28),(843,28)],purple,1.4,10)
    text(934,35,r'Backward $D_f^{\top}$',20.4,purple)
    text(474, 66, '(b) Attention product', 25, weight='bold', ha='left')
    pc, vc, yc = 558, 734, 931
    text(pc, 95, 'Attention weights', 20.4)
    text(vc, 95, 'Value vectors', 20.4)
    text(yc, 95, 'Output', 20.4)
    matrix(pc, 108, 4, 4, '#c7b38e')
    matrix(vc, 108, 4, 2, '#8dacce')
    matrix(yc, 108, 4, 2, '#a9b9c5')
    text(648, 138, r'$\times$', 25, gray)
    arrow([(789,128),(880,128)],gray,1.7,12)
    text(pc, 172, r'$P$', 25, '#785823')
    text(vc, 172, r'$V$', 25, blue)
    text(yc, 172, r'$Y=PV$', 25)
    # Coefficients are vertically aligned with the activations they belong to.
    arrow([(905,200),(pc,200),(pc,216)],purple,1.8)
    arrow([(vc,200),(vc,216)],purple,1.8)
    text(pc, 242, r'$m_P$', 24, purple)
    text(vc, 242, r'$m_V$', 24, purple)
    text(yc, 206, r'$m_Y$', 24, purple)
    arrow([(990,200),(957,200)],purple,1.8)
    ax.plot([474/72,988/72],[260/72,260/72],color=line,linewidth=.8)

    # The memory diagram shows the recurrence dependencies in one time step.
    text(474, 288, '(c) Gated DeltaNet', 25, weight='bold', ha='left')
    past, now, out = 516, 746, 952
    text(past, 315, 'Memory',20.4)
    text(now, 315, 'Memory',20.4)
    text(out, 315, 'Readout',20.4)
    matrix(past,326,4,4,'#9db4c9',9)
    matrix(now,326,4,4,'#8dacce',9)
    matrix(out,326,4,1,'#a9b9c5',9)
    arrow([(past+27,344),(now-28,344)],gray,1.6,11)
    arrow([(now+28,344),(out-17,344)],gray,1.6,11)
    text(631, 332, 'Value, key, gates',20.4)
    text(631, 368, 'Update',20.4)
    text(851, 332, 'Query',20.4)
    text(851, 368, 'Read',20.4)
    text(past, 390, r'$S_{t-1}$',24)
    text(now, 390, r'$S_t$',24)
    text(out, 390, r'$o_t$',24)
    # Readout and later-state coefficients both enter the current state.
    text(past, 422, r'$m_{S_{t-1}}$',23,purple)
    text(now, 422, r'$m_{S_t}$',23,purple)
    text(out, 422, r'$m_{o_t}$',23,purple)
    arrow([(923,414),(778,414)],purple,1.7,11)
    arrow([(715,414),(555,414)],purple,1.7,11)
    arrow([(990,414),(980,414)],purple,1.7,9)
    arrow([(631,414),(631,433)],purple,1.5,10)
    arrow([(851,414),(851,433)],purple,1.5,10)
    text(631,454,'Value, key, gate coefficients',20.4,purple)
    text(865,454,'Query coefficients',20.4,purple)
    return texts
