"""Task-local length pairing with an explicit, score-independent padding budget."""


def group_by_padding(cases,batch_size,length,max_padding_ratio=1.25):
    if batch_size not in (1,2):raise ValueError('Only measured sample B1/B2 are supported.')
    if max_padding_ratio<1:raise ValueError('Padding work ratio must be >=1.')
    ordered=sorted(cases,key=length)
    groups=[]
    for i in range(0,len(ordered),batch_size):
        group=ordered[i:i+batch_size]
        # Bound added linear/token work, not a fitted latency prediction. The
        # initial matched-length pilot gave ~20% speedup; >25% padding can
        # consume that benefit. This policy never reads attribution or labels.
        padded=len(group)*max(map(length,group));valid=sum(map(length,group))
        if padded>max_padding_ratio*valid:groups.extend([[c] for c in group])
        else:groups.append(group)
    return groups
