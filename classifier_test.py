from intelligent_placer_lib.objects_carver import find_objects_on_img


def test_classifier_1():
    path = "PlacerDataset/1.jpg"
    names, objects = find_objects_on_img(path)
    expected = ['мышь', 'масло для губ', 'футляр для наушников']
    assert sorted(names) == sorted(expected)


def test_classifier_2():
    path = "PlacerDataset/2.jpg"
    names, objects = find_objects_on_img(path)
    expected = ['флешка', 'рулетка', 'йо-йо', 'точилка']
    assert sorted(names) == sorted(expected)


def test_classifier_3():
    path = "PlacerDataset/3.jpg"
    names, objects = find_objects_on_img(path)
    expected = ['флешка', 'рулетка', 'йо-йо', 'точилка']
    assert sorted(names) == sorted(expected)


def test_classifier_4():
    path = "PlacerDataset/4.jpg"
    names, objects = find_objects_on_img(path)
    expected = ['флешка', 'рулетка', 'йо-йо', 'точилка']
    assert sorted(names) == sorted(expected)


def test_classifier_5():
    path = "PlacerDataset/5.jpg"
    names, objects = find_objects_on_img(path)
    expected = ['флешка', 'рулетка', 'йо-йо', 'точилка']
    assert sorted(names) == sorted(expected)
