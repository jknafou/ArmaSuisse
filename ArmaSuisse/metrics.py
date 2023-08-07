from sklearn.metrics import accuracy_score, f1_score

def EMR(prediction, gold):
    assert set(prediction.keys()) == set(gold.keys())
    per_question_score = 0
    for id in prediction.keys():
        per_question_score += int(prediction[id]['predicted_answers'] == gold[id]['correct_answers'])
    return per_question_score / len(prediction)

def HS(prediction, gold):
    assert set(prediction.keys()) == set(gold.keys())
    per_question_score = 0
    for id in prediction.keys():
        p_set = set(prediction[id]['predicted_answers'])
        g_set = set(gold[id]['correct_answers'])
        per_question_score += len(p_set & g_set) / len(p_set | g_set)
    return per_question_score / len(prediction)

def accuracy(prediction, gold):
    assert set(prediction.keys()) == set(gold.keys())
    predictions, golds = [], []
    for id in prediction.keys():
        predictions.append(prediction[id]['nbr_predicted_answers']-1)
        golds.append(gold[id]['nbr_correct_answers']-1)

    return accuracy_score(golds, predictions)

def f1(prediction, gold):
    assert set(prediction.keys()) == set(gold.keys())
    predictions, golds = [], []
    for id in prediction.keys():
        predictions.append(str(prediction[id]['nbr_predicted_answers']))
        golds.append(str(gold[id]['nbr_correct_answers']))
    return f1_score(golds, predictions, labels=[str(i+1) for i in range(5)], average='macro')