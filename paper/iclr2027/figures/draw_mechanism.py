"""Balanced Attention/GDN mechanisms with a measured name-to-number example.

Operator diagrams are schematic. Only the text backgrounds and full-input
strip encode measured scores. Attention and GDN occupy identical panel areas.
"""
import re

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Circle
from matplotlib.path import Path

INK='#233547'
MUTED='#748293'
BLUE='#477BAE'
PURPLE='#8971B1'
SIGNED_COLORS=['#D5816C','#FFFFFF','#4F9C87']
LINE='#D4DFE7'
PALE_BLUE='#ECF4FB'
PALE_PURPLE='#F3EFF9'
W,H=14.0,9.5


def draw(data):
    roles=any(c['dataset']=='morehopqa' and c['index']==1 for c in data['cases'])
    dataset,index=('morehopqa',1) if roles else ('niah_mq_q2',6)
    case=next(c for c in data['cases'] if c['model']=='qwen35' and c['dataset']==dataset and c['index']==index)
    paired=[c for c in data['cases'] if c['dataset']==dataset and c['index']==index]
    full_limit=max(abs(t['score']) for c in paired for t in c['tokens'] if t['eligible'])
    # Saturation is explicitly labelled. Raw scores and span totals are unchanged.
    limit=2.5 if roles else full_limit
    norm=Normalize(-limit,limit,clip=True)
    cmap=LinearSegmentedColormap.from_list('dt_signed',SIGNED_COLORS)
    source=case['user_text']
    facts=[('billowy-method','9937326'),('bright-system','9153566')]
    verified_source=["William Shakespeare's play",'The score is by William Walton.'] if roles else [f'{name} is: {number}.' for name,number in facts]
    verified_target=['William Shakespeare','7 + 11 = 18'] if roles else [f'{number} for {name}' for name,number in facts]
    assert all(s in source for s in verified_source)
    assert all(s in case['target'] for s in verified_target)

    height=8.85 if roles else H
    footer_shift=H-height
    fig=plt.figure(figsize=(W,height),facecolor='white')
    ax=fig.add_axes([0,0,1,1])
    ax.set_xlim(0,W);ax.set_ylim(height,0);ax.axis('off')
    artists=[]

    def text(x,y,value,size=17,color=INK,weight='normal',**kw):
        size=max(size,17) if value not in ('−','+') else size
        t=ax.text(x,y,value,fontsize=size,color=color,fontweight=weight,
                  va=kw.pop('va','top'),fontfamily=kw.pop('fontfamily','DejaVu Sans'),
                  linespacing=1.25,**kw)
        artists.append(t)
        return t

    def box(x,y,w,h,fc='white',ec=LINE,r=.08,lw=1,z=1):
        p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                        facecolor=fc,edgecolor=ec,linewidth=lw,zorder=z)
        ax.add_patch(p);return p

    def arrow(a,b,color=MUTED,lw=1.5,scale=13,**kw):
        p=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=scale,
                          linewidth=lw,color=color,shrinkA=0,shrinkB=0,**kw)
        ax.add_patch(p);return p

    def path_arrow(points,color=PURPLE,lw=2.1,scale=14):
        p=FancyArrowPatch(path=Path(points,[Path.MOVETO]+[Path.LINETO]*(len(points)-1)),
                          arrowstyle='-|>',mutation_scale=scale,color=color,
                          linewidth=lw,joinstyle='round')
        ax.add_patch(p)

    def heading(x,y,letter,title,size=21.2):
        text(x,y,f'({letter})',size,weight='bold')
        text(x+.60,y,title,size,weight='bold')

    def plane(x,y,w,h,label,fc,ec):
        pts=[(x+.10,y),(x+w,y),(x+w-.10,y+h),(x,y+h)]
        ax.add_patch(Polygon([(a,b+.045) for a,b in pts],fc=ec,ec='none',alpha=.22))
        ax.add_patch(Polygon(pts,fc=fc,ec=ec,lw=1))
        text(x+w/2,y+h/2,label,17,ha='center',va='center')

    def matrix(x,y,rows,cols,cw,ch,color):
        # Uniform schematic cells; these are not attention-score heatmaps.
        for i in range(rows):
            for j in range(cols):
                ax.add_patch(Rectangle((x+j*cw,y+i*ch),cw-.024,ch-.024,
                                      fc=color,ec='white',lw=.5))

    def allocation_cards(x,left_label,right_label,left_formula,right_formula,out,formula_size=24):
        # Identical geometry gives both mechanisms equal visual weight.
        for xx,label,formula,fc,color in [
                (x,left_label,left_formula,PALE_BLUE,BLUE),
                (x+2.22,right_label,right_formula,PALE_PURPLE,PURPLE)]:
            box(xx,2.67,1.98,.75,fc,color,.08,.9)
            text(xx+.99,2.36,label,17,color,weight='bold',ha='center')
            text(xx+.99,2.89,formula,formula_size,color,ha='center')
        center=x+2.08
        path_arrow([(x+.99,3.44),(x+.99,3.58),(center-.18,3.91)],BLUE,2,12)
        path_arrow([(x+3.21,3.44),(x+3.21,3.58),(center+.18,3.91)],PURPLE,2,12)
        ax.add_patch(Circle((center,3.99),.15,fc='white',ec='#9CADBC',lw=1.1))
        text(center,3.99,'+',21,ha='center',va='center')
        arrow((center+.17,3.99),(center+.53,3.99),INK,1.7,12)
        text(center+.69,3.82,out,25)

    # (a) The paired endpoint graph and the finite reverse coefficient.
    heading(.22,.14,'a','Finite reverse trace')
    text(.23,.56,'Two inputs. One fixed response y.',17,MUTED)
    text(1.14,1.00,r'$F(x_0;y)$',19,ha='center')
    text(3.25,1.00,r'$F(x_1;y)$',19,ha='center')
    text(2.20,.92,r'$\Delta F$',22,PURPLE,ha='center')
    text(1.75,1.18,'−',13.7,MUTED,ha='center')
    text(2.64,1.18,'+',13.7,MUTED,ha='center')
    arrow((1.64,1.15),(1.88,1.15),MUTED,1.2,10)
    arrow((2.76,1.15),(2.53,1.15),MUTED,1.2,10)
    for xc,fc,ec in [(1.14,'#F5F7F9','#BBC9D5'),(3.25,PALE_BLUE,'#87A8C7')]:
        arrow((xc,2.93),(xc,1.37),'#B7C4CF',1.6,12,zorder=1)
        plane(xc-.81,1.58,1.62,.42,'MLP + skip',fc,ec)
        plane(xc-.81,2.26,1.62,.42,'Attn. / GDN',fc,ec)
        text(xc,2.01,'⋮',17,MUTED,ha='center')
    for yy in [1.80,2.48]:
        ax.plot([1.96,2.44],[yy,yy],color='#C2B3DA',lw=1.1,ls=(0,(2.4,2)))
        ax.add_patch(Circle((2.20,yy),.047,fc=PURPLE,ec='white',lw=.8,zorder=6))
    arrow((2.20,1.32),(2.20,3.12),PURPLE,3,17,zorder=5)
    text(2.20,3.23,r'$m_i$',17,PURPLE,ha='center')
    box(.33,3.00,1.62,.40,'#F5F7F9','#BCC9D4',.06)
    text(1.14,3.20,'EOS EOS …',17,MUTED,ha='center',va='center')
    box(2.44,3.00,1.62,.40,PALE_BLUE,'#86A6C3',.06)
    text(3.25,3.20,'source words' if roles else '9153566',17,BLUE,ha='center',va='center')
    text(1.14,3.49,r'Reference $x_0$',17,MUTED,ha='center')
    text(3.25,3.49,r'Original $x_1$',17,BLUE,ha='center')
    text(2.20,3.81,'One finite reverse traversal',17,PURPLE,ha='center')
    text(2.20,4.13,r'$A_i=\langle m_i,\,\Delta e_i\rangle$',22,PURPLE,ha='center')
    ax.plot([4.52,4.52],[.19,4.45],color=LINE,lw=.8)

    # (b) Attention: selection weights multiply the carried values.
    heading(4.92,.14,'b','Attention')
    text(4.95,.56,'Select and carry',17,MUTED)
    matrix(5.26,1.16,4,4,.185,.185,'#BDAFD6')
    matrix(6.95,1.16,4,2,.185,.185,'#8FAED0')
    matrix(8.41,1.16,4,2,.185,.185,'#B2BFCA')
    text(6.45,1.44,'×',23,MUTED,ha='center')
    text(7.87,1.44,'=',23,MUTED,ha='center')
    text(5.62,1.98,r'$P$',21,PURPLE,ha='center')
    text(7.12,1.98,r'$V$',21,BLUE,ha='center')
    text(8.58,1.98,r'$Y$',21,ha='center')
    allocation_cards(4.95,'Content','Selection',r'$P_1\,\Delta V$',r'$\Delta P\,V_0$',r'$\Delta Y$')
    ax.plot([9.39,9.39],[.19,4.45],color=LINE,lw=.8)

    # (c) GDN: same panel size, same two-channel allocation diagram.
    heading(9.65,.14,'c','Gated DeltaNet',21.2)
    text(9.67,.56,'Retain, write, and read',17,MUTED)
    text(9.83,1.69,r'$S_{t-1}$',20,BLUE,ha='center',va='center')
    arrow((10.22,1.69),(10.45,1.69),BLUE,1.5,10)
    box(10.48,1.46,.46,.45,PALE_PURPLE,'#BAABD0',.06)
    text(10.71,1.69,r'$\alpha$',20,PURPLE,ha='center',va='center')
    arrow((10.96,1.69),(11.22,1.69),BLUE,1.5,10)
    ax.add_patch(Circle((11.36,1.69),.12,fc='white',ec='#9CADBC',lw=1.1))
    text(11.36,1.69,'+',18,ha='center',va='center')
    text(11.36,1.03,r'$k u^{\top}$',21,BLUE,ha='center')
    arrow((11.36,1.36),(11.36,1.54),BLUE,1.5,10)
    arrow((11.50,1.69),(11.75,1.69),BLUE,1.5,10)
    box(11.79,1.45,.51,.47,PALE_BLUE,'#A5C0D9',.06)
    text(12.04,1.69,r'$S_t$',20,BLUE,ha='center',va='center')
    arrow((12.32,1.69),(12.55,1.69),BLUE,1.5,10)
    box(12.59,1.46,.44,.45,PALE_PURPLE,'#BAABD0',.06)
    text(12.81,1.69,r'$q$',20,PURPLE,ha='center',va='center')
    arrow((13.05,1.69),(13.30,1.69),INK,1.5,10)
    text(13.51,1.69,r'$o_t$',20,ha='center',va='center')
    text(11.74,2.02,'Retention: T = αS',17,MUTED,ha='center')
    allocation_cards(9.65,'Stored content','Retention gate',
                     r'$\alpha_1\,\Delta S$',r'$S_0\,\Delta\alpha$',r'$\Delta T$',23)

    # (d) A direct lookup makes source and answer visually match.
    ax.plot([.22,13.80],[4.70,4.70],color=LINE,lw=.85)
    heading(.22,4.90,'d','Choose the playwright, not the composer.' if roles else 'Find the names. Recover the numbers.',21.2)
    text(13.77,5.00,'Qwen3.5 · MoreHopQA example 1' if roles else 'Qwen3.5 · retrieval example 6',17,MUTED,ha='right')
    owners=[[] for _ in source]
    for i,t in enumerate(case['tokens']):
        for j in range(*t['char_span']):owners[j].append(i)
    spans,displayed=[],[]
    font=20.0;step=font/72*.602;line_height=font/72*1.31

    def paragraph(value,x,y,width=9.21,prefix='',suffix=''):
        lo=source.index(value);hi=lo+len(value)
        spans.append([lo,hi]);capacity=int(width/step)
        groups=[list(range(lo+m.start(),lo+m.end())) for m in re.finditer(r'\S+\s*',value)]
        lines,current=[],[]
        for group in groups:
            if current and len(current)+len(group)>capacity:lines.append(current);current=[]
            current.extend(group)
        if current:lines.append(current)
        if prefix:text(x-.08,y+.015,prefix,font,MUTED,ha='right')
        for chars in lines:
            while chars and source[chars[-1]].isspace():chars.pop()
            runs,run,last=[],[],None
            for j in chars:
                assert len(owners[j])==1,('Ambiguous displayed character',j)
                owner=owners[j][0]
                if owner!=last and run:runs.append((last,run));run=[]
                run.append(j);last=owner
            if run:runs.append((last,run))
            col=0
            for owner,chars in runs:
                token=case['tokens'][owner];xx=x+col*step;ww=len(chars)*step
                ax.add_patch(Rectangle((xx,y),ww,line_height*.89,
                    fc=cmap(norm(token['score'])) if token['eligible'] else 'white',ec='none',zorder=1))
                text(xx,y+.013,''.join(source[j] for j in chars),font,fontfamily='DejaVu Sans Mono',zorder=2)
                displayed.append({'token_index':owner,'char_span':[chars[0],chars[-1]+1],
                                  'score':token['score'],'bbox':[xx,y,ww,line_height*.89]})
                col+=len(chars)
            y+=line_height
        if suffix:text(x+col*step+.05,y-line_height+.015,suffix,font,MUTED)
        return y

    text(.26,5.43,'CONTEXT EXCERPTS',17,MUTED,weight='bold')
    span_scores=[]
    if roles:
        text(9.72,5.43,'Name-span total (nats)',17,MUTED,ha='right')
        paragraph("William Shakespeare's play",.57,5.82,prefix='…')
        paragraph('The score is by William Walton.',.57,6.30,prefix='…')
        for y,name,col in [(5.82,'William Shakespeare','#357D6D'),(6.30,'William Walton','#B86754')]:
            lo=source.rindex(name);hi=lo+len(name)
            selected=[t for t in case['tokens'] if t['char_span'][0]<hi and t['char_span'][1]>lo]
            total=sum(t['score'] for t in selected)
            span_scores.append({'name':name,'char_span':[lo,hi],'sum':total,'token_indices':[t['local_index'] for t in selected]})
            text(9.25,y,f'{total:+.2f}'.replace('-','−'),22,col,weight='bold',ha='right')
            excerpt=verified_source[0] if name=='William Shakespeare' else verified_source[1]
            offset=lo-source.index(excerpt)
            x1=.57+offset*step;x2=x1+len(name)*step;yy=y+line_height+.025
            ax.plot([x1,x1,x2,x2],[yy-.035,yy,yy,yy-.035],color=col,lw=1.05)
        text(.26,6.85,'REQUESTED ROLE (EXCERPT)',17,MUTED,weight='bold')
        question='the author of the play'
    else:
        text(9.72,5.43,'name selects · number supplies',17,MUTED,ha='right')
        paragraph('billowy-method is: 9937326.',.57,5.82,prefix='…')
        paragraph('bright-system is: 9153566.',.57,6.30,prefix='…')
        text(.26,6.85,'QUESTION',17,MUTED,weight='bold')
        question='What are all the special magic numbers for bright-system, and billowy-method mentioned in the provided text?'
    bottom=paragraph(question,.57,7.23,prefix='…' if roles else '',suffix='…' if roles else '')
    assert bottom<8.37,(bottom,'Question exceeds heatmap bounds')

    # Summary of exact recorded answer pairs, not a new generated response.
    answer_height=1.95 if roles else 2.66
    box(10.26,5.40,3.53,answer_height,'#FCF8ED','#D7BE8A',.10)
    text(10.50,5.63,'Answer',17,'#9A8054',weight='bold')
    answer_texts=[]
    if roles:
        answer_texts.append(text(10.51,6.13,'William',25,INK,weight='bold'))
        answer_texts.append(text(10.51,6.58,'Shakespeare',25,INK,weight='bold'))
    else:
        for y,(name,number) in zip([6.10,7.00],facts):
            answer_texts.append(text(10.51,y,number,28,INK,weight='bold'))
            answer_texts.append(text(10.51,y+.57,name,18,'#9A8054'))
    text(12.02,8.28-footer_shift,r'$\sum_i A_i=\Delta F$',24,PURPLE,ha='center')
    text(12.02,8.98-footer_shift,'Across all input sources',17,MUTED,ha='center')

    # A thin complete-input strip retains the unshown positive/negative context.
    sx,sy,sw=.57,8.50-footer_shift,9.14
    tw=sw/len(case['tokens'])
    for j,t in enumerate(case['tokens']):
        ax.add_patch(Rectangle((sx+j*tw,sy),tw+.0002,.09,
            fc=cmap(norm(t['score'])) if t['eligible'] else 'white',ec='none'))
    for lo,hi in spans:
        ids=[j for j,t in enumerate(case['tokens']) if t['char_span'][0]<hi and t['char_span'][1]>lo]
        ax.plot([sx+min(ids)*tw,sx+(max(ids)+1)*tw],[sy+.14,sy+.14],color=INK,lw=1.1)
    text(.57,8.72-footer_shift,'Full input',17,MUTED)
    text(9.71,8.72-footer_shift,'Shown spans underlined',17,MUTED,ha='right')

    # The role excerpt leaves room for the legend above the full-input strip.
    bx,by,bw,bh=(5.225,7.23,2.45,.09) if roles else (3.68,8.98-footer_shift,2.45,.09)
    tick_y=by+.17
    legend_artists=[]
    for i in range(256):
        ax.add_patch(Rectangle((bx+i*bw/256,by),bw/256+.0002,bh,fc=cmap(i/255),ec='none'))
    lo_label=f'≤−{limit:.1f}' if roles else f'{-limit:.1f}'
    hi_label=f'≥+{limit:.1f}' if roles else f'+{limit:.1f}'
    for xx,ss in [(bx,lo_label),(bx+bw/2,'0'),(bx+bw,hi_label)]:
        legend_artists.append(text(xx,tick_y,ss,17,MUTED,ha='center'))
    legend_artists.append(text(bx+bw/2,by-.28,'nats · color saturated' if roles else 'Contribution (nats)',17,MUTED,ha='center'))

    fig._dt_case={'model':case['model'],'dataset':case['dataset'],'index':case['index'],
        'input_sha256':case['input_sha256'],'text_excerpt_char_spans':spans,
        'answer_summary':'Playwright identified within the stored response: William Shakespeare' if roles else facts,
        'verified_source_fragments':verified_source,'verified_target_fragments':verified_target,
        'name_span_scores':span_scores,
        'fixed_target':'entire stored response plus EOS',
        'vmin':-limit,'vmax':limit,'normalization':'linear, zero-centered; explicitly saturated at endpoints' if roles else 'linear, centered on zero',
        'full_input_absolute_max':full_limit,
        'color_map':SIGNED_COLORS,'shared_scale':'one shared scale for all excerpts and full-input strip',
        'color_legend':{'bar_bounds':[bx,by,bw,bh],'tick_y':tick_y,'label_y':by-.28,
                        'placement':'above full-input strip, right of requested role' if roles else 'below full-input strip'},
        'selection_reason':'same-context playwright/composer role contrast with observed positive and negative name-span totals' if roles else 'direct name-to-number lookup; short exact evidence spans and answers',
        'mechanism_panel_areas':{'attention':[4.95,.14,4.20,4.31],'gdn':[9.65,.14,4.20,4.31]},
        'schematic_elements':'paired finite trace; PV product; GDN retain/write/read; two equally sized local content/control allocations',
        'gdn_retention_notation':'S abbreviates S_(t-1); T=alpha*S; endpoint subscripts 0 and 1 are reference and original',
        'heatmap_semantics':'unchanged raw token scores; uniformly colored matrix cells are schematic, not measured attention',
        'displayed_runs':displayed}
    fig._dt_text_artists=artists
    fig._dt_legend_artists=legend_artists
    fig._dt_box_checks=[(t,(10.26,5.40,3.53,answer_height)) for t in answer_texts]
    return fig
