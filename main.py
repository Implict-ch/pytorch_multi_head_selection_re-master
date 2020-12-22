import os
import json
import time
import argparse

import torch

from typing import Dict, List, Tuple, Set, Optional

from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

from torch.optim import Adam, SGD

from pytorch_transformers import AdamW, WarmupLinearSchedule

from lib.preprocessings import Chinese_selection_preprocessing, Conll_selection_preprocessing, Conll_bert_preprocessing, Datafountain_selection_preprocessing
from lib.dataloaders import Selection_Dataset, Selection_loader
from lib.metrics import F1_triplet, F1_ner
from lib.models import MultiHeadSelection
from lib.config import Hyper
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',
                    '-e',
                    type=str,
                    default='datafountain_selection_re',
                    help='experiments/exp_name.json')

parser.add_argument('--mode',
                    '-m',
                    type=str,
                    default='evaluation',
                    help='preprocessing|train|evaluation')

args = parser.parse_args()


class Runner(object):
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.model_dir = 'saved_models'
        self.device = torch.device('cpu')
        self.hyper = Hyper(os.path.join('experiments', self.exp_name + '.json'))

        self.gpu = self.hyper.gpu
        self.preprocessor = None
        self.triplet_metrics = F1_triplet()
        self.ner_metrics = F1_ner()
        self.optimizer = None
        self.model = None

    def _optimizer(self, name, model):
        m = {
            'adam': Adam(model.parameters()),
            'sgd': SGD(model.parameters(), lr=0.5),
            'adamw': AdamW(model.parameters())
        }
        return m[name]

    def _init_model(self):
        self.model = MultiHeadSelection(self.hyper).to(self.device)

    def preprocessing(self):
        if self.exp_name == 'conll_selection_re':
            self.preprocessor = Conll_selection_preprocessing(self.hyper)
        elif self.exp_name == 'chinese_selection_re':
            self.preprocessor = Chinese_selection_preprocessing(self.hyper)
        elif self.exp_name == 'conll_bert_re':
            self.preprocessor = Conll_bert_preprocessing(self.hyper)
        elif self.exp_name == 'datafountain_selection_re':
            self.preprocessor = Datafountain_selection_preprocessing(self.hyper)

        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=1)
        # for ner only
        self.preprocessor.gen_bio_vocab()

    def run(self, mode: str):
        if mode == 'preprocessing':
            self.preprocessing()
        elif mode == 'train':
            self._init_model()
            self.optimizer = self._optimizer(self.hyper.optimizer, self.model)
            self.train()
        elif mode == 'evaluation':
            self._init_model()
            self.load_model(epoch=self.hyper.evaluation_epoch)
            self.evaluation()
        else:
            raise ValueError('invalid mode')

    def load_model(self, epoch: int):
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir,
                             self.exp_name + '_' + str(epoch))))

    def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, self.exp_name + '_' + str(epoch)))

    def evaluation(self):
        dev_set = Selection_Dataset(self.hyper, self.hyper.dev)
        loader = Selection_loader(dev_set, batch_size=self.hyper.eval_batch, pin_memory=True)
        self.triplet_metrics.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        with torch.no_grad():
            with open('./output/sub00.csv', 'w') as file:
                id = 0
                for batch_ndx, sample in pbar:
                    tokens = sample.tokens_id.to(self.device)
                    selection_gold = sample.selection_id.to(self.device)
                    bio_gold = sample.bio_id.to(self.device)
                    text_list = sample.text
                    spo_gold = sample.spo_gold
                    bio_text = sample.bio


                    output = self.model(sample, is_train=False)

                    self.triplet_metrics(output['selection_triplets'], output['spo_gold'])
                    self.ner_metrics(output['gold_tags'], output['decoded_tag'])

                    for i in range(len(output['decoded_tag'])):
                        file.write(str(8001+id)+',')
                        if len(output['selection_triplets'][i]) != 0:
                            file.write(output['selection_triplets'][i][0]['predicate']+',')
                            file.write(output['selection_triplets'][i][0]['subject']+',')
                            file.write(output['selection_triplets'][i][0]['object']+'\n')
                        else:
                            if output['decoded_tag'][i].count('B') < 2:
                                file.write('Other'+','+'Other'+','+'Other')
                            else:
                                BIO = output['decoded_tag'][i]
                                tt = ''.join(reversed(BIO))
                                index1 = BIO.index('B')
                                index2 = len(tt)-tt.index('B')-1
                                file.write('Other'+','+text_list[i][index2]+','+text_list[i][index1])
                            file.write('\n')
                        id += 1
                        # file.write('sentence {} BIO:\n'.format(i))
                        # for j in range(len(text_list[i])):
                        #     file.write(text_list[i][j]+' ')
                        # file.write('\n')
                        # file.writelines(bio_text[i])
                        # file.write('\n')
                        #
                        # file.writelines(output['decoded_tag'][i])
                        # file.write('\n')
                        # file.writelines(output['gold_tags'][i])
                        # file.write('\n')
                        # file.write('sentence {} relation:\n'.format(i))
                        # file.write('\n')
                        # if len(output['selection_triplets']) == 0:
                        #     file.write('empty')
                        # else:
                        #     file.writelines(str(output['selection_triplets'][i]))
                        # file.write('\n')
                        # file.writelines(str(output['spo_gold'][i]))
                        # file.write('\n')

            triplet_result = self.triplet_metrics.get_metric()
            ner_result = self.ner_metrics.get_metric()
            # print('triplet_result=', triplet_result)
            # print('ner_result=', ner_result)

            print('Triplets-> ' +  ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in triplet_result.items() if not name.startswith("_")
            ]) + ' ||' + 'NER->' + ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in ner_result.items() if not name.startswith("_")
            ]))

    def train(self):
        #**
        self.load_model(epoch=self.hyper.epoch_num)
        # **
        train_set = Selection_Dataset(self.hyper, self.hyper.train)
        loader = Selection_loader(train_set, batch_size=self.hyper.train_batch, pin_memory=True)

        for epoch in range(self.hyper.epoch_num):
            self.model.train()
            pbar = tqdm(enumerate(BackgroundGenerator(loader)),
                        total=len(loader))

            for batch_idx, sample in pbar:
                # txt = sample.text
                # for j in range(len(txt)):
                #     print(txt[j])

                self.optimizer.zero_grad()
                output = self.model(sample, is_train=True)
                loss = output['loss']
                loss.backward()
                self.optimizer.step()

                pbar.set_description(output['description'](
                    epoch, self.hyper.epoch_num))
            ## self.hyper.epoch_num
            self.save_model(self.hyper.epoch_num)

            if epoch % self.hyper.print_epoch == 0 and epoch > 3:
                self.evaluation()


if __name__ == "__main__":
    config = Runner(exp_name=args.exp_name)
    config.run(mode=args.mode)
