"""Import the authors' baseline aggregates, preserving the published row choices."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
parser = argparse.ArgumentParser()
parser.add_argument('--release-dir', type=Path, required=True)
args = parser.parse_args()
release = args.release_dir
archive = release / 'exp2_table1_results.tar.gz'
checksums = dict(line.split(maxsplit=1)[::-1] for line in (release/'SHA256SUMS').read_text().splitlines() if line.strip())
assert sha(archive.read_bytes()) == checksums['exp2_table1_results.tar.gz']
tasks = list(csv.DictReader((HERE/'data/qwen3_full.csv').open()))
methods = {'Perturbation':'perturbation_all','REAGENT':'perturbation_REAGENT',
           'CLP':'perturbation_CLP','IFR':'ifr_all_positions','AttnLRP':'attnlrp'}
# Values printed in FlashTrace v4 Tables 1 and 2 (PDF pages 7 and 8), used only
# to check the source-row selection. Imported values retain CSV precision.
printed = {
 'Perturbation': {
  'RISE':[.095,.239,.499,.134,.186,.351,.384,.354,.458,.466,.133,.380,.249],
  'MAS':[.144,.327,.709,.187,.244,.458,.551,.517,.684,.701,.220,.520,.347],
  'Recovery10':[.391,.090,.010,.255,.161,.080]},
 'REAGENT': {
  'RISE':[.117,.260,.487,.188,.211,.369,.438,.397,.486,.495,.145,.388,.265],
  'MAS':[.197,.357,.694,.276,.291,.494,.668,.603,.745,.741,.235,.536,.368],
  'Recovery10':[.244,.085,.005,.180,.156,.074]},
 'CLP': {
  'RISE':[.098,.253,.510,.156,.217,.393,.374,.328,.423,.451,.101,.362,.249],
  'MAS':[.166,.320,.657,.216,.280,.511,.490,.420,.565,.597,.190,.491,.348],
  'Recovery10':[.399,.086,.008,.207,.146,.073]},
 'IFR': {
  'RISE':[.075,.115,.371,.069,.073,.205,.161,.102,.125,.153,.074,.354,.146],
  'MAS':[.140,.177,.460,.134,.142,.275,.231,.148,.173,.201,.166,.490,.228],
  'Recovery10':[.471,.328,.012,.575,.452,.179]},
 'AttnLRP': {
  'RISE':[.196,.263,.377,.140,.193,.285,.319,.324,.338,.357,.155,.368,.286],
  'MAS':[.326,.451,.602,.229,.325,.475,.521,.548,.572,.592,.249,.524,.458],
  'Recovery10':[.215,.204,.076,.254,.243,.159]}}
sources=[]; rows=[]; differences=[]
with tarfile.open(archive,'r:gz') as tar:
    members = {m.name:m for m in tar.getmembers() if m.isfile()}
    for index, task in enumerate(tasks):
        dataset=task['dataset']; count=int(task['count'])
        for label,stem in methods.items():
            # The paper uses the fast perturbation runs except MQ-Q2 and
            # MoreHopQA; the MATH archive also contains earlier plain runs.
            fast = label in ('Perturbation','REAGENT','CLP') and dataset not in ('niah_mq_q2','morehopqa')
            name=stem+('_fast' if fast else '')
            file_count=count if label in ('IFR','AttnLRP') else 100
            row={'dataset':dataset,'method':label,'released_task_count':count,'RISE':'','MAS':'','Recovery10':''}
            for view in ('faithfulness','recovery'):
                if view=='recovery' and not dataset.startswith('niah_'):continue
                relative=f'output/{view}/exp/exp2/data/{dataset}.jsonl/qwen-8B/{name}_{file_count}_examples.csv'
                candidates=[k for k in members if k.endswith('/'+relative) or k==relative]
                assert len(candidates)==1,(relative,candidates)
                member=candidates[0];raw=tar.extractfile(members[member]).read()
                assert raw == (release/'extracted'/relative).read_bytes()
                records=list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
                selected='Row Attr Scores Mean' if view=='faithfulness' else 'Row Attr Recovery Mean'
                # Table 1's IFR MQ-Q8 pair is the published Recursive row;
                # all other additional baseline means match the Row view.
                if (dataset,label,view)==('niah_mq_q8','IFR','faithfulness'):
                    selected='Recursive Attr Scores Mean'
                chosen=next(r for r in records if r[0]==selected)
                if view=='faithfulness':row['RISE'],row['MAS']=chosen[1:3]
                else:
                    row['Recovery10']=chosen[1]
                    used=next(int(r[1]) for r in records if r[0]=='Examples Used')
                    assert used==count,(dataset,label,used,count)
                destination=HERE/'data/published_baselines_raw'/view/dataset/(name+'.csv')
                destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
                sources.append({'dataset':dataset,'method':label,'metric_group':view,'archive_member':member,
                                'fixture':destination.relative_to(HERE).as_posix(),'sha256':sha(raw),'selected_row':selected})
            for metric in ('RISE','MAS','Recovery10'):
                if not row[metric]:continue
                delta=abs(float(row[metric])-printed[label][metric][index])
                assert delta<=.0011,(dataset,label,metric,row[metric],printed[label][metric][index])
                differences.append(delta)
            rows.append(row)
out=HERE/'data/published_baselines.csv'
with out.open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
report={'release':'table1-data-v1','archive_sha256':sha(archive.read_bytes()),
        'paper_crosscheck':'FlashTrace arXiv:2602.01914v4 Tables 1 and 2, PDF pages 7 and 8',
        'printed_values_checked':len(differences),'maximum_difference_from_printed_value':max(differences),
        'crosscheck_tolerance':.0011,'values_imported_from':'full-precision released CSV rows, not the rounded PDF',
        'faithfulness_row':'Row Attr Scores Mean, except IFR MQ-Q8: Recursive Attr Scores Mean, matching published Table 1',
        'recovery_row':'Row Attr Recovery Mean',
        'count_scope':'released_task_count identifies the prepared task selection; recovery Examples Used was checked explicitly. Some faithfulness filenames retain the requested cap of 100.',
        'recovery_selection':'NIAH only; VT and HotpotQA recovery excluded by author request; MATH/MoreHopQA unavailable.',
        'methods':list(methods),'source_csv_files':sources,'output_sha256':sha(out.read_bytes()),
        'published_printed_crosscheck':printed}
(HERE/'data/published_baseline_sources.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
provenance=json.loads((HERE/'sources.json').read_bytes())
for name in ('published_baselines.csv','published_baseline_sources.json'):
    path=HERE/'data'/name
    provenance['sources'][path.stem]={'source':'author table1-data-v1 / exp2_table1_results.tar.gz',
        'source_sha256':report['archive_sha256'],'fixture':path.relative_to(HERE).as_posix(),'fixture_sha256':sha(path.read_bytes())}
(HERE/'sources.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'method_task_rows':len(rows),'source_csv_files':len(sources),'printed_values_checked':len(differences),'max_printed_difference':max(differences)}))
