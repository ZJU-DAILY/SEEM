HORIZONS=(96,192,336,720)
def expected_counts(h):
 if h not in HORIZONS:raise ValueError('Unsupported Traffic horizon')
 return dict(train=12185-h,validation=1757-h,test=3509-h)
def role_split(count,horizon,seq_len=96):
 if count<4 or horizon<1 or seq_len!=96:raise ValueError('Invalid dimensions')
 a=count-count//4;b=a+seq_len+horizon-1
 if b>=count:raise ValueError('No B readers')
 return dict(A=[0,a],purge=[a,b],B=[b,count],window=seq_len+horizon)
