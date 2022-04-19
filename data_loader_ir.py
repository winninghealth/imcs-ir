# -*- coding: utf-8 -*-
# @author: Yiwen Jiang @Winning Health Group

import json
import torch
import logging
logger = logging.getLogger(__name__)

from typing import Dict, List
from overrides import overrides
from transformers import BertTokenizer

from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TensorField, LabelField, ListField

SPECIAL_TOKENS = {'患者':'[unused1]', '医生':'[unused2]'}
SPECIAL_LABELS = {'Other', 'Diagnose'}

WINDOW = 30 # Decided by my GPU Restriction

class IntentionRecognitionDatasetReader(DatasetReader):
    def __init__(self, transformer_load_path: str, **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._transformer_indexers = BertTokenizer.from_pretrained(transformer_load_path)
    
    @overrides
    def _read(self, file_path):
        with open(file_path, "r", encoding='utf-8') as file:
            data_file = json.load(file)
            for eid in data_file.keys():
                dialogue, speaker_ids, intentions, actions = [], [], [], []
                for sid in data_file[eid]['dialogue']:
                    speaker_ids.append(sid['speaker'])
                    speaker = [SPECIAL_TOKENS[sid['speaker']]]
                    utterance = list(sid['sentence'])
                    utterance = speaker + utterance
                    dialogue.append(utterance)
                    if sid['dialogue_act'] not in SPECIAL_LABELS:
                        intention, action = sid['dialogue_act'].split('-')
                    else:
                        intention = sid['dialogue_act']
                        action = sid['dialogue_act']
                    intentions.append(intention)
                    actions.append(action)
                # If you have sufficient GPU Memory, Put Whole Dialogue in will be better.
                for i in range(0, len(dialogue), WINDOW):
                    y = i + WINDOW
                    yield self.text_to_instance(dialogue[i:y],
                                                speaker_ids[i:y],
                                                intentions[i:y],
                                                actions[i:y])
                    if y >= len(dialogue):
                        break
    
    def text_to_instance(
        self,
        dialogue: List[List[str]],
        speaker_ids: List[str],
        intentions: List[str] = None,
        actions: List[str] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}
        dialogue = [['[CLS]'] + utterance + ['[SEP]'] for utterance in dialogue]
        dialogue_field = [self._transformer_indexers.convert_tokens_to_ids(utterance) for utterance in dialogue]
        dialogue_field = [TensorField(torch.tensor(u)) for u in dialogue_field]
        fields["dialogue"] = ListField(dialogue_field)
        speaker_field = [LabelField(speaker, label_namespace='speaker_labels') for speaker in speaker_ids]
        fields["speaker"] = ListField(speaker_field)
        if intentions != None:
            intents_field = [LabelField(intention, label_namespace='intention_labels') for intention in intentions]
            fields["intentions"] = ListField(intents_field)
        if actions != None:
            actions_field = [LabelField(action, label_namespace='action_labels') for action in actions]
            fields["actions"] = ListField(actions_field)
        return Instance(fields)

