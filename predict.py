# -*- coding: utf-8 -*-
# @author: Yiwen Jiang @Winning Health Group

import os
import json
import torch
import logging
import argparse

from tqdm import tqdm
from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

from trainer import build_model
from transformers import BertTokenizer
from data_loader_ir import IntentionRecognitionDatasetReader, SPECIAL_TOKENS, SPECIAL_LABELS

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

logger = logging.getLogger(__name__)

class IRPredictor(Predictor):    
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 transformer_load_path : str
                 ) -> None:
        super().__init__(model, dataset_reader)
        self.vocab = model.vocab
        self._transformer_indexers = BertTokenizer.from_pretrained(transformer_load_path)
        
    def predict(self, dialogue, speaker_ids) -> JsonDict:
        result = self.predict_json({"dialogue": dialogue, "speaker": speaker_ids})
        instances = dict()
        instances['actions'] = [self.vocab.get_token_from_index(i,namespace='action_labels') for i in result['actions']]
        instances['intentions'] = [self.vocab.get_token_from_index(i,namespace='intention_labels') for i in result['intentions']]
        return instances

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        dialogue = json_dict["dialogue"]
        speaker_ids = json_dict["speaker"]
        return self._dataset_reader.text_to_instance(dialogue, speaker_ids)

def read_input_file(input_path):
    eids, dialogues, speaker_ids = [], [], []
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for k in data.keys():
        dialogue, speaker_id = [], []
        for sent in data[k]['dialogue']:
            speaker = [SPECIAL_TOKENS[sent['speaker']]]
            utterance = list(sent['sentence'])
            utterance = speaker + utterance
            dialogue.append(utterance)
            speaker_id.append(sent['speaker'])
        eids.append(k)
        dialogues.append(dialogue)
        speaker_ids.append(speaker_id)
    return eids, dialogues, speaker_ids

def predict(pred_config):
    serialization_dir = pred_config.model_dir
    vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
    vocab = Vocabulary.from_files(vocabulary_dir)
    
    model_dir = os.path.join(serialization_dir, pred_config.model_name)
    model = build_model(vocab, pred_config.pretrained_model_dir, pred_config.pretrained_hidden_size)
    device = torch.device(pred_config.cuda_id if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model = model.to(device)
    
    dataset_reader = IntentionRecognitionDatasetReader(transformer_load_path=pred_config.pretrained_model_dir)
    predictor = IRPredictor(model=model,
                            dataset_reader=dataset_reader,
                            transformer_load_path=pred_config.pretrained_model_dir)
    
    eids, dialogues, speaker_ids = read_input_file(os.path.join(pred_config.test_input_file))
    predict_result = dict()
    predict_subres = dict()
    for i in tqdm(range(0, len(eids))):
        predict_result[eids[i]] = dict()
        predict_subres[eids[i]] = dict()
        result = predictor.predict(dialogues[i], speaker_ids[i])
        assert len(result['actions']) == len(result['intentions'])
        for idx, j in enumerate(zip(result['actions'], result['intentions'])):
            _act, _int = j
            if _act not in SPECIAL_LABELS and _int not in SPECIAL_LABELS:
                res = _int + '-' + _act
            else:
                res = _act if _act in SPECIAL_LABELS else _int
            # Restore BAD index from original test file
            if eids[i] == '10708561' and int(idx+1) >= 43:
                predict_result[eids[i]][str(idx+2)] = res
                predict_subres[eids[i]][str(idx+2)] = dict()
                predict_subres[eids[i]][str(idx+2)]['act'] = _act
                predict_subres[eids[i]][str(idx+2)]['int'] = _int
            else:
                predict_result[eids[i]][str(idx+1)] = res
                predict_subres[eids[i]][str(idx+1)] = dict()
                predict_subres[eids[i]][str(idx+1)]['act'] = _act
                predict_subres[eids[i]][str(idx+1)]['int'] = _int
    pred_path = os.path.join(pred_config.test_output_file)
    with open(pred_path, 'w', encoding='utf-8') as json_file:
        json.dump(predict_result, json_file, ensure_ascii=False, indent=4)
    pred_path_sub = os.path.join(pred_config.test_output_file + '.sub')
    with open(pred_path_sub, 'w', encoding='utf-8') as json_file:
        json.dump(predict_subres, json_file, ensure_ascii=False, indent=4)
    logger.info("Prediction Done!")

if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_input_file", default="./data/IMCS_test.json", type=str)
    parser.add_argument("--test_output_file", default="IMCS-IR_test.json", type=str)
    parser.add_argument("--model_dir", default="./save_model", type=str)
    parser.add_argument("--model_name", default="best.th", type=str)
    parser.add_argument("--pretrained_model_dir", default="./plms", type=str)
    parser.add_argument("--pretrained_hidden_size", default=768, type=int)
    parser.add_argument("--cuda_id", default='cuda:0', type=str)
    
    pred_config = parser.parse_args()
    predict(pred_config)
    
    