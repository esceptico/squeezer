from squeezer.reduce import Average, DictAverage


def test_average():
    average = Average()
    for i in range(1, 100):
        average.update(i)
    assert average.compute() == 50


def test_dict_average():
    average = DictAverage()
    for i in range(1, 100):
        average.update({'a': i, 'b': i * 2})
    res = average.compute()
    assert res['a'] == 50
    assert res['b'] == 100
