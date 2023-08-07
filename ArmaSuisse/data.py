from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass
import torch, numpy as np, pandas as pd, random, os, glob, json
from sklearn.metrics import accuracy_score, f1_score
from operator import itemgetter
from scipy.special import softmax


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def results2dataset(data_dir):
    dfs = pd.read_excel(data_dir + 'results.xlsx')
    dfs = dfs.fillna('0')
    dfs.columns = [c.replace(' ', '_') for c in dfs.columns]
    dataset = ''
    for i in range(len(dfs)):
        text = dfs.title[i] + '\t' + dfs.snippet[i]
        label = '1' if dfs.Relevant_to_Security[i] == 'YES' else '0'
        dataset += str(i) + '\t' + text + '\t' + label + '\n'

    with open(data_dir + 'dataset.csv', mode='w') as f:
        f.write(dataset)

def cross_fold_splitting(data_dir, n_fold=10):
    with open(data_dir + 'dataset.csv') as f:
        dataset = f.read()

    dataset = dataset.strip().split('\n')
    random.Random(1234).shuffle(dataset)

    for k, fold in enumerate(split(dataset, n_fold)):
        with open(data_dir + 'dataset_fold=' + str(k) + '.csv', mode='w') as f:
            f.write('\n'.join(fold))

class ClassificationDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset):
        self.examples = []
        for line in dataset:
            doc_id, title, snippet, label = line.split('\t')
            self.examples.append({
                'doc_id': int(doc_id),
                'title': title,
                'snippet': snippet,
                'label': int(label)})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

@dataclass
class collator_fn:

    tokenizer: PreTrainedTokenizerBase
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    padding = 'max_length'
    max_length: int = 512
    pad_to_multiple_of: int = 8

    def __call__(self, features):
        batch, index = [], []
        for i in range(len(features)):
            token_ids = self.tokenizer.encode(features[i]['title'], truncation=True, add_special_tokens=True)
            token_ids += self.tokenizer.encode(features[i]['snippet'], truncation=True, add_special_tokens=False)[:self.max_length-(len(token_ids)+1)] + [self.tokenizer.sep_token_id]
            index.append(features[i]['doc_id'])
            batch.append({'input_ids': token_ids,
                          'attention_mask': [1 for _ in token_ids],
                          'labels':features[i]['label'] })

        batch = self.tokenizer.pad(
            batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: torch.tensor(v, dtype=torch.int64).to(self.device) for k, v in batch.items()}
        return [batch, index]

class Data():
    def __init__(self, data_dir, tokenizer, batch_size, k_fold, n_fold):

        self.data_groups = ['train', 'dev', 'test']
        split_by_data_group = {}
        split_by_data_group['test'] = [k_fold]
        split_by_data_group['dev'] = [k_fold + 1 if k_fold < n_fold-1 else 0]
        split_by_data_group['train'] = [k for k in range(n_fold) if k not in split_by_data_group['test'] + split_by_data_group['dev']]

        self.dataset, self.dataset_dict = {}, {}
        for data_group in self.data_groups:
            self.dataset[data_group] = []
            for k in split_by_data_group[data_group]:
                with open(data_dir + 'dataset_fold=' + str(k) + '.csv') as f:
                    file = f.read()
                    self.dataset[data_group] += file.strip().split('\n')

            self.dataset_dict[data_group] = self.dataset2dict(self.dataset[data_group])

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.dataloader = {}
        for data_group in self.dataset.keys():
            self.dataloader[data_group] = {}
            self.dataloader[data_group]['dataset'] = ClassificationDataset(self.dataset[data_group])
            if data_group == 'train':
                self.dataloader[data_group]['sampler'] = torch.utils.data.sampler.RandomSampler(self.dataloader[data_group]['dataset'])

            else:
                self.dataloader[data_group]['sampler'] = torch.utils.data.sampler.SequentialSampler(self.dataloader[data_group]['dataset'])

            self.dataloader[data_group]['collator'] = collator_fn(tokenizer=self.tokenizer)#, device=device)
            self.dataloader[data_group] = torch.utils.data.dataloader.DataLoader(self.dataloader[data_group]['dataset'],
                                                                 batch_size=batch_size,
                                                                 sampler=self.dataloader[data_group]['sampler'],
                                                                 collate_fn=self.dataloader[data_group]['collator'])

    def predicion2dict(self, doc_ids, logits):
        dataset = {}
        for doc_id, logit in zip(doc_ids, logits):
            dataset[doc_id] = \
                {'label': np.argmax(logit),
                'probability': softmax(logit)[1]}

        return dataset

    def dataset2dict(self, dataset):
        dataset_dict = {}
        for line in dataset:
            doc_id, title, snippet, label = line.split('\t')
            dataset_dict[int(doc_id)] = {
                'title': title,
                'snippet': snippet,
                'label': int(label)}

        return dataset_dict

    def accuracy(self, prediction, gold):
        predictions, golds = [], []
        for doc_id in prediction.keys():
            predictions.append(prediction[doc_id]['label'])
            golds.append(gold[doc_id]['label'])

        return accuracy_score(golds, predictions)

    def f1_score(self, prediction, gold, average='macro'):
        predictions, golds = [], []
        for doc_id in prediction.keys():
            predictions.append(prediction[doc_id]['label'])
            golds.append(gold[doc_id]['label'])

        return f1_score(golds, predictions, average=average)

    def prediction_file_unification(self, model_dir):
        model_dir = '/'.join(model_dir.split('/')[:-1]) + '/'
        prediction_file_paths = [x[0] + '/test_prediction.csv' for x in os.walk(model_dir) if 'fold=' in x[0] and os.path.exists(x[0] + '/test_prediction.csv')]
        predictions = []
        for prediction_file_path in prediction_file_paths:
            with open(prediction_file_path) as f:
                file = f.read()
                predictions += file.strip().split('\n')

        predictions = [p.split('\t') for p in predictions]
        predictions = [(int(doc_id), l, p) for [doc_id, l, p] in predictions]
        predictions = sorted(predictions, key=itemgetter(0))

        file = ''
        for [doc_id, l, p] in predictions:
            file += str(doc_id) + '\t' + l + '\t' + p + '\n'

        with open(model_dir + 'dataset_prediction.csv', mode='w') as f:
            f.write(file)

    def ensemble(self, model_dir):
        prediction_file_paths = glob.glob(model_dir + '/*/dataset_prediction.csv')
        prediction = {}
        for prediction_file_path in prediction_file_paths:
            with open(prediction_file_path) as f:
                file = f.read()
            file = file.strip().split('\n')

            file = [p.split('\t') for p in file]
            file = {int(doc_id): float(p) for [doc_id, _, p] in file}
            for doc_id in file.keys():
                if doc_id not in prediction:
                    prediction[doc_id] = {
                        'probability': []
                    }

                prediction[doc_id]['probability'].append(file[doc_id])

        doc_ids = list(prediction.keys())
        doc_ids.sort()
        file = ''
        for doc_id in doc_ids:
            p = np.mean(prediction[doc_id]['probability'])
            l = 0 if p < .5 else 1
            prediction[doc_id] = {
                'label': l,
                'probability': p
            }
            file += str(doc_id) + '\t' + str(l) + '\t' + str(p) + '\n'

        with open(model_dir + 'ensemble_dataset_prediction.csv', mode='w') as f:
            f.write(file)

        gold = self.dataset_dict['train'] | self.dataset_dict['dev'] | self.dataset_dict['test']

        metrics = {
            'data_size': len(prediction),
            'accuracy': self.accuracy(prediction, gold),
            'macro_f1_score': self.f1_score(prediction, gold),
        }
        with open(model_dir + 'ensemble_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)