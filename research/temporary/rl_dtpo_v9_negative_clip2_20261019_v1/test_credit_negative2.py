"""Reuse the clip1 behavior suite with the approved negative2 event bound."""
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner=ROOT/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v1/source/test_credit_v9.py'
code=owner.read_text()
replacements=[
    ("monkeypatch.setenv('DT_CREDIT_MODE',MODE)",
     "monkeypatch.setenv('DT_CREDIT_MODE',MODE)\n    monkeypatch.setenv('DT_EXPONENTIAL_CREDIT_NEGATIVE_CLIP','2')"),
    ('clip_exponential_credit=True','clip_exponential_credit=True,exponential_credit_negative_clip=2.0'),
    ("{'clip_exponential_credit':True}","{'clip_exponential_credit':True,'exponential_credit_negative_clip':2.0}"),
    ('.clamp(-1,1)','.clamp(-2,1)'),
    ('-math.log(2)+.001','-math.log(3)+.001'),
    ('==2*g','==3*g'),('==-g','==-2*g'),
    ('==(-.1875,.1875,.375)','==(-.375,.1875,.5625)'),
    ("['event_u_clip']==[-1.,1.]","['event_u_clip']==[-2.,1.]"),
    ("torch.full((int(prior.sum()),),.25)","torch.full((int(prior.sum()),),.5)"),
    ('clipped.advantages.min()==-b[0,0]','clipped.advantages.min()==-2*b[0,0]'),
]
for old,new in replacements:
    assert old in code,old
    code=code.replace(old,new)
exec(compile(code,str(owner),'exec'),globals())


@pytest.mark.parametrize('bound',[0.,-2.,float('nan'),float('inf')])
def test_invalid_negative_credit_bound_is_rejected(bound):
    with pytest.raises(ValueError,match='negative clip'):
        reward_event_token_credit(torch.zeros(1,1,2),torch.ones(1,1),
            torch.ones(1,1,2,dtype=torch.bool),torch.ones(1,2,dtype=torch.bool),
            clip_exponential_credit=True,exponential_credit_negative_clip=bound)
