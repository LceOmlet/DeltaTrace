"""Check the final overview's data binding and rendered text geometry."""
import json
from pathlib import Path

import build_figures  # shared fonts and backend
from draw_mechanism import draw

HERE=Path(__file__).resolve().parent


def main():
    data=json.loads((HERE/'data/overview_role_case.json').read_bytes())
    fig=draw(data)
    fig.canvas.draw()
    (HERE.parent/'preview').mkdir(exist_ok=True)
    fig.savefig(HERE.parent/'preview/overview-current.png',dpi=150)
    fig.canvas.draw()  # Restore the canvas renderer after the different-DPI preview.
    renderer=fig.canvas.get_renderer()
    texts=fig._dt_text_artists
    outside=[]
    for t in texts:
        b=t.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.x1+1 or b.y1 > fig.bbox.y1+1:
            outside.append(t.get_text())
    assert not outside,outside
    collisions=[]
    labels=[t for t in texts if 'Mono' not in ' '.join(t.get_fontfamily())]
    for i,a in enumerate(labels):
        ba=a.get_window_extent(renderer)
        for b in labels[i+1:]:
            bb=b.get_window_extent(renderer)
            w=min(ba.x1,bb.x1)-max(ba.x0,bb.x0)
            h=min(ba.y1,bb.y1)-max(ba.y0,bb.y0)
            if w>1 and h>1:collisions.append([a.get_text(),b.get_text()])
    assert not collisions,collisions
    for legend in fig._dt_legend_artists:
        a=legend.get_window_extent(renderer)
        for source_text in texts:
            if source_text in fig._dt_legend_artists:
                continue
            b=source_text.get_window_extent(renderer)
            assert min(a.x1,b.x1)-max(a.x0,b.x0)<=1 or min(a.y1,b.y1)-max(a.y0,b.y0)<=1, (legend.get_text(),source_text.get_text())
    for t,(x,y,w,h) in fig._dt_box_checks:
        b=t.get_window_extent(renderer)
        ax=fig.axes[0]
        left,bottom=ax.transData.transform((x,y+h))
        right,top=ax.transData.transform((x+w,y))
        assert b.x0>=left and b.x1<=right and b.y0>=bottom and b.y1<=top,t.get_text()
    meta=fig._dt_case
    case=next(c for c in data['cases'] if c['model']==meta['model'] and c['dataset']==meta['dataset'])
    for run in meta['displayed_runs']:
        t=case['tokens'][run['token_index']]
        assert t['score']==run['score']
        a,b=run['char_span'];lo,hi=t['char_span']
        assert lo<=a<b<=hi
    assert all(s in case['user_text'] for s in meta['verified_source_fragments'])
    assert all(s in case['target'] for s in meta['verified_target_fragments'])
    for span in meta['name_span_scores']:
        assert sum(case['tokens'][i]['score'] for i in span['token_indices'])==span['sum']
    panels=meta['mechanism_panel_areas']
    assert panels['attention'][2:]==panels['gdn'][2:]
    report={
        'figure_inches':list(fig.get_size_inches()),
        'canvas_clipping':outside,'label_collisions':collisions,
        'response_text_inside_box':True,
        'verified_rendered_token_runs':len(meta['displayed_runs']),
        'answer_pairs_match_source_and_target':True,
        'attention_and_gdn_equal_area':True,
        'shared_linear_scale':[meta['vmin'],meta['vmax']],
        'raw_scores_modified':False,
        'color_saturation':meta['normalization'],
        'name_span_scores':meta['name_span_scores'],
        'method_panels':'schematic; no measured branch or intermediate-state scores',
        'color_legend':meta['color_legend'],
        'legend_clear_of_all_source_text':True,
    }
    path=HERE/'overview_layout_verification.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report))


if __name__=='__main__':main()
