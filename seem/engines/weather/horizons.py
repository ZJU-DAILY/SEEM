HORIZONS = (96,192,336,720)


def role_split(count, horizon, seq_len=96):
    if horizon < 1 or seq_len != 96 or count < 4:
        raise ValueError('Invalid input/horizon/count')
    span = seq_len + horizon
    a_end = count - count // 4
    b_start = a_end + span - 1
    if b_start >= count:
        raise ValueError('No B readers after complete input+target boundary purge')
    return dict(A=[0, a_end], purge=[a_end, b_start], B=[b_start, count], window=span)


def expected_counts(horizon):
    if horizon not in HORIZONS: raise ValueError('Weather H96/H192/H336/H720 only')
    return dict(train=36792-horizon, validation=5271-horizon, test=10540-horizon)
