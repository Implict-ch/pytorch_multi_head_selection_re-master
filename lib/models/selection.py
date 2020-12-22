import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import os
import copy

from typing import Dict, List, Tuple, Set, Optional
from functools import partial

from torchcrf import CRF
from pytorch_transformers import *

class MultiHeadSelection(nn.Module):
    def __init__(self, hyper) -> None:
        super(MultiHeadSelection, self).__init__()
        self.device = torch.device('cpu')
        self.hyper = hyper
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu
        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))

        self.id2bio = {v: k for k, v in self.bio_vocab.items()}

        self.word_embeddings = nn.Embedding(num_embeddings=len(
            self.word_vocab),
            embedding_dim=hyper.emb_size)

        self.relation_emb = nn.Embedding(num_embeddings=len(self.relation_vocab), embedding_dim=hyper.rel_emb_size)

        # bio + pad
        self.bio_emb = nn.Embedding(num_embeddings=len(self.bio_vocab),
                                    embedding_dim=hyper.bio_emb_size)

        if hyper.cell_name == 'gru':
            self.encoder = nn.GRU(hyper.emb_size,
                                  hyper.hidden_size,
                                  bidirectional=True,
                                  batch_first=True)

        elif hyper.cell_name == 'lstm':
            self.encoder = nn.LSTM(hyper.emb_size,
                                   hyper.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   num_layers=2)

        elif hyper.cell_name == 'bert':
            self.post_lstm = nn.LSTM(hyper.emb_size,
                                   hyper.hidden_size,
                                   bidirectional=True,
                                   batch_first=True)
            self.encoder = BertModel.from_pretrained('data/bert-base-uncased')
            for name, param in self.encoder.named_parameters():
                if '11' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    # print(name, param.size())
        else:
            raise ValueError('cell name should be gru/lstm/bert!')

        if hyper.activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif hyper.activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('unexpected activation!')

        self.emission = nn.Linear(hyper.hidden_size, len(self.bio_vocab) - 1)
        self.norm1 = nn.BatchNorm1d(self.hyper.max_text_len)
        self.dropout1 = nn.Dropout(0)

        self.tagger = CRF(len(self.bio_vocab) - 1, batch_first=True)
        self.selection_u = nn.Linear(hyper.hidden_size + hyper.bio_emb_size, hyper.rel_emb_size)
        self.norm2 = nn.BatchNorm1d(self.hyper.max_text_len)
        self.dropout2 = nn.Dropout(0)
        self.selection_v = nn.Linear(hyper.hidden_size + hyper.bio_emb_size, hyper.rel_emb_size)
        self.norm3 = nn.BatchNorm1d(self.hyper.max_text_len)
        self.dropout3 = nn.Dropout(0)
        self.selection_uv = nn.Linear(2 * hyper.rel_emb_size, hyper.rel_emb_size)
        self.norm4 = nn.BatchNorm2d(self.hyper.max_text_len, self.hyper.max_text_len)
        self.dropout4 = nn.Dropout2d(0)

        self.bert2hidden = nn.Linear(768, hyper.hidden_size)
        # for bert_lstm
        # self.bert2hidden = nn.Linear(768, hyper.emb_size)

        if self.hyper.cell_name == 'bert':
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased')

        # self.accuracy = F1Selection()

    def inference(self, mask, text_list, decoded_tag, selection_logits):
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(
                              -1, -1, len(self.relation_vocab),
                              -1)  # batch x seq x rel x seq
        selection_tags = (torch.sigmoid(selection_logits) * selection_mask.float()) > self.hyper.threshold
        selection_triplets = self.selection_decode(text_list, decoded_tag, selection_tags)
        return selection_triplets

    def masked_BCEloss(self, mask, selection_logits, selection_gold):
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(
                              -1, -1, len(self.relation_vocab),
                              -1)  # batch x seq x rel x seq
        selection_loss = F.binary_cross_entropy_with_logits(selection_logits,
                                                            selection_gold,
                                                            reduction='none')
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        selection_loss /= mask.sum()
        return selection_loss

    @staticmethod
    def description(epoch, epoch_num, output):
        return "L: {:.2f}, L_crf: {:.2f}, L_selection: {:.2f}, epoch: {}/{}:".format(
            output['loss'].item(), output['crf_loss'].item(),
            output['selection_loss'].item(), epoch, epoch_num)

    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:

        tokens = sample.tokens_id.to(self.device)
        # print(tokens[0])
        selection_gold = sample.selection_id.to(self.device)
        # print(selection_gold[0])
        bio_gold = sample.bio_id.to(self.device)
        # print(bio_gold[0])
        text_list = sample.text
        # print(text_list[0])
        spo_gold = sample.spo_gold
        # print(spo_gold[0])
        bio_text = sample.bio
        # print(bio_text[0])

        if self.hyper.cell_name in ('gru', 'lstm'):
            mask = tokens != self.word_vocab['<pad>']  # batch x seq
            bio_mask = mask

        elif self.hyper.cell_name in ('bert'):
            notpad = tokens != self.bert_tokenizer.encode('[PAD]')[0]
            notcls = tokens != self.bert_tokenizer.encode('[CLS]')[0]
            notsep = tokens != self.bert_tokenizer.encode('[SEQ]')[0]
            mask = notpad & notcls & notsep
            bio_mask = notpad & notsep  # fst token for crf cannot be masked
        else:
            raise ValueError('unexpected encoder name!')

        if self.hyper.cell_name in ('lstm', 'gru'):
            embedded = self.word_embeddings(tokens)
            o, h = self.encoder(embedded)
            o = (lambda a: sum(a) / 2)(torch.split(o, self.hyper.hidden_size, dim=2))

        elif self.hyper.cell_name == 'bert':
            # with torch.no_grad():
            o = self.encoder(tokens, attention_mask=mask)[
                0]
            o = self.bert2hidden(o)
        else:
            raise ValueError('unexpected encoder name!')

        emi = self.activation(self.dropout1(self.norm1(self.emission(o))))

        output = {}
        crf_loss = 0

        if is_train:
            crf_loss = -self.tagger(emi, bio_gold, mask=bio_mask, reduction='mean')
        else:
            decoded_tag = self.tagger.decode(emissions=emi, mask=bio_mask)

            output['decoded_tag'] = [list(map(lambda x : self.id2bio[x], tags)) for tags in decoded_tag]
            output['gold_tags'] = bio_text

            temp_tag = copy.deepcopy(decoded_tag)
            for line in temp_tag:
                line.extend([self.bio_vocab['<pad>']] *
                            (self.hyper.max_text_len - len(line)))
            bio_gold = torch.tensor(temp_tag).to(self.device)

        tag_emb = self.bio_emb(bio_gold)

        o = torch.cat((o, tag_emb), dim=2)

        # forward multi head selection
        B, L, H = o.size()
        u = self.activation(self.dropout2(self.norm2(self.selection_u(o)).unsqueeze(1).expand(B, L, L, -1)))
        v = self.activation(self.dropout3(self.norm3(self.selection_v(o)).unsqueeze(2).expand(B, L, L, -1)))
        uv = self.activation(self.dropout4(self.norm4(self.selection_uv(torch.cat((u, v), dim=-1)))))

        # print(uv.shape)
        # print(self.relation_emb.weight.shape)
        # correct one
        selection_logits = torch.einsum('bijh,rh->birj', uv, self.relation_emb.weight)

        if not is_train:
            output['selection_triplets'] = self.inference(
                mask, text_list, decoded_tag, selection_logits)
            output['spo_gold'] = spo_gold

        selection_loss = 0
        if is_train:
            selection_loss = self.masked_BCEloss(mask, selection_logits, selection_gold)

        loss = 2 * crf_loss + selection_loss
        output['crf_loss'] = crf_loss
        output['selection_loss'] = selection_loss
        output['loss'] = loss

        output['description'] = partial(self.description, output=output)
        return output

    def selection_decode(self, text_list, sequence_tags,
                         selection_tags: torch.Tensor
                         ) -> List[List[Dict[str, str]]]:
        reversed_relation_vocab = {
            v: k
            for k, v in self.relation_vocab.items()
        }

        reversed_bio_vocab = {v: k for k, v in self.bio_vocab.items()}

        text_list = list(map(list, text_list))

        def find_entity(pos, text, sequence_tags):
            entity = []

            if sequence_tags[pos] in ('B', 'O'):
                entity.append(text[pos])
            else:
                temp_entity = []
                while sequence_tags[pos] == 'I':
                    temp_entity.append(text[pos])
                    pos -= 1
                    if pos < 0:
                        break
                    if sequence_tags[pos] == 'B':
                        temp_entity.append(text[pos])
                        break
                entity = list(reversed(temp_entity))
            return ''.join(entity)

        batch_num = len(sequence_tags)
        result = [[] for _ in range(batch_num)]
        idx = torch.nonzero(selection_tags.cpu())
        for i in range(idx.size(0)):
            b, s, p, o = idx[i].tolist()

            predicate = reversed_relation_vocab[p]
            if predicate == 'N':
                continue
            tags = list(map(lambda x: reversed_bio_vocab[x], sequence_tags[b]))
            object = find_entity(o, text_list[b], tags)
            subject = find_entity(s, text_list[b], tags)
            assert object != '' and subject != ''

            triplet = {
                'object': object,
                'predicate': predicate,
                'subject': subject
            }
            result[b].append(triplet)
        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass
