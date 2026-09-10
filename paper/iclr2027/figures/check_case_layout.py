"""Verify redesigned case geometry against the original token and curve data."""
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
from build_figures import case_figure_compact, CMAP

HERE=Path(__file__).resolve().parent
data_path=HERE/'data/cases.json'
data=json.loads(data_path.read_bytes())
reports=[]
for dataset in ('niah_mq_q2','morehopqa'):
    fig,meta=case_figure_compact(data['cases'],dataset)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    all_text=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    outside=[]
    for t in all_text:
        b=t.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.x1+1 or b.y1 > fig.bbox.y1+1:
            outside.append(t.get_text())
    assert not outside, outside
    labels=[t for t in all_text if 'Mono' not in ' '.join(t.get_fontfamily())]
    overlaps=[]
    for i,a in enumerate(labels):
        ba=a.get_window_extent(renderer)
        for b in labels[i+1:]:
            bb=b.get_window_extent(renderer)
            w=min(ba.x1,bb.x1)-max(ba.x0,bb.x0)
            h=min(ba.y1,bb.y1)-max(ba.y0,bb.y0)
            if w>1 and h>1:
                overlaps.append([a.get_text(),b.get_text()])
    assert not overlaps, overlaps
    assert not any('development' in t.get_text().lower() for t in all_text)
    assert max(t.get_fontsize() for t in fig._case_text_artists) == 10.0
    rows={c['model']:c for c in data['cases'] if c['dataset']==dataset}
    from matplotlib.colors import Normalize
    norm=Normalize(meta['vmin'],meta['vmax'])
    for run in meta['displayed_runs']:
        case=rows[run['model']]
        token=case['tokens'][run['token_index']]
        lo,hi=run['char_span']
        assert token['char_span'][0]<=lo<hi<=token['char_span'][1]
        assert run['text']==case['user_text'][lo:hi]
        assert run['score']==token['score'] and run['eligible']==token['eligible']
        expected=CMAP(norm(token['score'])) if token['eligible'] else [1,1,1,1]
        assert np.array_equal(run['rgba'],expected)
        x,y,w,h=run['bounds']
        col=meta['aligned_model_columns']['x'][0 if run['model']=='qwen3' else 1]
        assert x>=col and x+w<=col+meta['aligned_model_columns']['width']+.001
    for model,case in rows.items():
        eligible=sum(t['eligible'] for t in case['tokens'])
        for method,key in [('DT','deletion'),('FT','deletion_ft')]:
            series=meta['deletion_series'][model][method]
            assert series['normalized_score']==case[key]['normalized_model_response']
            expected=[len(d)/eligible for d in case[key]['deleted_user_indices']]
            assert series['removed_fraction']==expected and len(expected)==21
            assert case[key]['actual_input_hashes'][0]==case['input_sha256']
            raw=np.array(case[key]['scores'])
            normalized=np.minimum.accumulate(np.clip((raw-raw[-1])/abs(raw[0]-raw[-1]),0,1))
            assert np.allclose(normalized,series['normalized_score'],atol=1e-12,rtol=0)
        assert case['deletion']['scores'][0]==case['deletion_ft']['scores'][0]
        assert case['deletion']['scores'][-1]==case['deletion_ft']['scores'][-1]
    fragments=['5443951','8698256'] if dataset=='niah_mq_q2' else ['Jennings','6']
    assert all(fragment in c['target'] for fragment in fragments for c in rows.values())
    report={'dataset':dataset,'figure_inches':meta['figure_inches'],
            'canvas_clipping':outside,'label_overlaps':overlaps,
            'verified_token_runs':len(meta['displayed_runs']),
            'source_and_target_fragments_verified':True,
            'raw_scores_modified':False,'deletion_points_per_model_and_method':21,
            'deletion_series_count':4,'model_matched_FT_baselines':True,
            'normalization_reconstructed_from_full_response_log_likelihood':True,
            'shared_linear_scale':[meta['vmin'],meta['vmax']],
            'global_title':False,'visible_development_identifier':False,
            'max_text_fontsize':10.0}
    reports.append(report)
    plt.close(fig)
result={'fixture_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),'cases':reports}
(HERE/'case_layout_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
