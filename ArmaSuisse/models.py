import torch, json
from data import *
from transformers import RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import trange, tqdm
from transformers.utils import (
    logging,
)
logger = logging.get_logger(__name__)

class Models():
    def __init__(self, model_name, learning_rate, max_epoch, warmup_proportion, batch_size, grad_acc, model_dir, k_fold, device_id):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.warmup_proportion = warmup_proportion
        self.batch_size = batch_size
        self.grad_acc = grad_acc
        self.device_id = device_id
        self.model_dir = model_dir + model_name.split('/')[-1] + '/'
        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
        self.model_dir += '/fold=' + str(k_fold)
        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.metrics = []
        self.n_gpu = torch.cuda.device_count()
        self.load_model(self.model_name)


    def load_model(self, model):
        logger.warning('loading model from :' + model)
        # if Huggingface model
        torch.cuda.set_device(self.device_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=2)
        self.model.to(self.device)
        # if self.n_gpu > 1:
        #     import torch.nn as nn
        #     self.model = nn.DataParallel(self.model, device_ids=[i for i in range(self.n_gpu)])

    def training(self, data):

        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-8)
        self.total_steps = len(data.dataloader['train']) * self.max_epoch
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=int(self.total_steps * self.warmup_proportion),
                                                num_training_steps=self.total_steps)
        best_dev_macro_f1_score, best_loss = 0.0, 1.0

        for epoch in trange(self.max_epoch, desc="Epoch"):
            grad_acc_step, tr_loss = 0, []
            pbar = tqdm(data.dataloader['train'])
            for step, [batch, _] in enumerate(pbar):
                output = self.model(**batch)
                loss = output.loss
                # if self.n_gpu > 1:
                #     loss = loss.mean()
                loss.backward()
                tr_loss.append(loss.item())
                grad_acc_step += 1
                if grad_acc_step == self.grad_acc:
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    grad_acc_step = 0
                    pbar.set_description('training loss=' + "{:2.4f}".format(np.mean(tr_loss[-10:])))

            doc_ids, logits, dev_loss = self.predict(data=data)
            prediction = data.predicion2dict(doc_ids, logits)
            self.metrics.append({
                'epoch': epoch + 1,
                'training_loss': np.mean(tr_loss),
                'dev_accuracy': data.accuracy(prediction, data.dataset_dict['dev']),
                'dev_macro_f1_score': data.f1_score(prediction, data.dataset_dict['dev']),
                'dev_loss': dev_loss
            })
            self.log_metrics()
            if self.metrics[-1]['dev_macro_f1_score'] >= best_dev_macro_f1_score:
                if self.metrics[-1]['dev_macro_f1_score'] == best_dev_macro_f1_score and dev_loss > best_loss:
                    continue
                # save model
                best_dev_macro_f1_score = self.metrics[-1]['dev_macro_f1_score']
                best_loss = dev_loss
                self.save_model()

            # if epoch == 0: break


    def testing(self, data):
        self.load_model(self.model_dir)
        doc_ids, logits, test_loss = self.predict(data=data, data_group='test')
        prediction = data.predicion2dict(doc_ids, logits)
        self.write_prediction(prediction)
        self.metrics.append({
            'test_accuracy': data.accuracy(prediction, data.dataset_dict['test']),
            'test_macro_f1_score': data.f1_score(prediction, data.dataset_dict['test']),
            'test_loss': test_loss
        })
        self.log_metrics()

        data.prediction_file_unification(self.model_dir)
        self.write_overall_results(data)

    def write_overall_results(self, data):
        with open('/'.join(self.model_dir.split('/')[:-1])  + '/dataset_prediction.csv') as f:
            prediction = f.read()

        prediction = prediction.strip().split('\n')
        prediction = [p.split('\t') for p in prediction]
        prediction = {int(doc_id): {'probability': float(p), 'label': int(l)} for doc_id, l, p in prediction}

        gold = data.dataset_dict['train'] | data.dataset_dict['dev'] | data.dataset_dict['test']

        metrics = {
            'data_size': len(prediction),
            'accuracy': data.accuracy(prediction, gold),
            'macro_f1_score': data.f1_score(prediction, gold),
        }
        logger.warning('Overall metrics so far: ')
        logger.warning(metrics)
        with open('/'.join(self.model_dir.split('/')[:-1]) + '/metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)



    def write_prediction(self, prediction):
        file = ''
        for doc_id in prediction.keys():
            file += str(doc_id) +\
                    '\t' + str(prediction[doc_id]['label']) +\
                    '\t' + str(prediction[doc_id]['probability']) + '\n'
        with open(self.model_dir + '/test_prediction.csv', mode='w') as f:
            f.write(file)

    def predict(self, data, data_group='dev'):
        self.model.eval()
        logits, doc_ids, eval_loss = [], [], []
        for step, [batch, doc_id] in enumerate(tqdm(data.dataloader[data_group])):
            doc_ids.append(doc_id)
            with torch.no_grad():
                output = self.model(**batch)
            loss = output.loss
            # if self.n_gpu > 1:
            #     loss = loss.mean()
            eval_loss.append(loss.item())
            logits.append(output.logits.detach().cpu().numpy())
            # if step == 5: break

        doc_ids = [item for sublist in doc_ids for item in sublist]
        logits = [item for sublist in logits for item in sublist]

        self.model.train()
        return doc_ids, logits, np.mean(eval_loss)

    def log_metrics(self):
        logger.warning('metrics: ')
        logger.warning(self.metrics[-1])
        with open(self.model_dir + '/metrics.json', 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=4)

    def save_model(self):
        # if self.n_gpu > 1:
        #     self.model.module.save_pretrained(self.model_dir)
        # else:
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)