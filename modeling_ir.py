# -*- coding: utf-8 -*-
# @author: Yiwen Jiang @Winning Health Group

import torch
import torch.nn as nn
from overrides import overrides
from modeling_bert import BertModel
from typing import Dict, Optional, cast, List

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import FBetaMeasure
from allennlp.modules import Seq2SeqEncoder, ConditionalRandomField

class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class IntentionLabelTagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        transformer_load_path: str,
        dialogue_encoder: Seq2SeqEncoder,
        dropout: Optional[float] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.utterance_encoder = BertModel.from_pretrained(transformer_load_path)
        self.dialogue_encoder = dialogue_encoder
        self.act_decoder = ClassificationHead(input_dim=self.dialogue_encoder.get_output_dim(),
                                              inner_dim=self.dialogue_encoder.get_output_dim(),
                                              num_classes=self.vocab.get_vocab_size('action_labels'),
                                              pooler_dropout=0.3)
        self.intent_decoder = ClassificationHead(input_dim=self.dialogue_encoder.get_output_dim(),
                                                 inner_dim=self.dialogue_encoder.get_output_dim(),
                                                 num_classes=self.vocab.get_vocab_size('intention_labels'),
                                                 pooler_dropout=0.3)
        self.speaker_embeds = self.utterance_encoder.embeddings.speaker_embeddings
        self.dropout = torch.nn.Dropout(dropout) if dropout else None
        self.calculate_f1_act = {
            "F1-macro-act": FBetaMeasure(average='macro'),
            "F1-class-act": FBetaMeasure(average=None)
        }
        self.calculate_f1_int = {
            "F1-macro-int": FBetaMeasure(average='macro'),
            "F1-class-int": FBetaMeasure(average=None)
        }
        self.crf_act = ConditionalRandomField(self.vocab.get_vocab_size('action_labels'))
        self.crf_int = ConditionalRandomField(self.vocab.get_vocab_size('intention_labels'))
        initializer(self)
    
    @overrides
    def forward(self, dialogue, speaker, intentions = None, actions = None, **kwargs,):
        batch_size, utter_len, seq_len = dialogue.shape
        dialogue = dialogue.reshape(batch_size * utter_len, seq_len)
        '''
        utterance feature
        '''
        speaker_ids = torch.repeat_interleave(speaker, seq_len, -1)
        speaker_ids = speaker_ids.reshape(batch_size * utter_len, seq_len)
        encoded_utterance = self.utterance_encoder(input_ids=dialogue,
                                                   attention_mask=dialogue != 0,
                                                   speaker_ids=torch.clamp(speaker_ids,min=0),
                                                   use_cache=True,
                                                   return_dict=True)['last_hidden_state']
        encoded_utterance = encoded_utterance.reshape(batch_size, utter_len, seq_len, -1)
        encoded_utterance = encoded_utterance[:,:,0,:]
        encoded_utterance = self.dropout(encoded_utterance) if self.dropout else encoded_utterance
        '''
        dialogue feature
        '''
        speaker_embeds = self.speaker_embeds(speaker)
        encoded_utterance = encoded_utterance + speaker_embeds
        encoded_dialogue = self.dialogue_encoder(encoded_utterance, None)
        encoded_dialogue = self.dropout(encoded_dialogue) if self.dropout else encoded_dialogue
        # [batch_size, utterance_number, utterance_embedding_size]
        '''
        decoder
        '''
        encoded_dialogue_act = self.act_decoder(encoded_dialogue)
        encoded_dialogue_int = self.intent_decoder(encoded_dialogue)
        '''
        metric
        '''
        output = dict()
        '''
        actions metric
        '''
        labels_mask = speaker != -1
        best_paths_act = self.crf_act.viterbi_tags(encoded_dialogue_act,labels_mask,top_k=1)
        predicted_acts = cast(List[List[int]], [x[0][0] for x in best_paths_act])
        output['actions'] = predicted_acts
        if actions != None:
            class_probabilities = encoded_dialogue_act * 0.0
            for i, instance_tags in enumerate(predicted_acts):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1
            self.calculate_f1_act['F1-macro-act'](class_probabilities, actions, labels_mask)
            self.calculate_f1_act['F1-class-act'](class_probabilities, actions, labels_mask)
        '''
        intentions metric
        '''
        best_paths_int = self.crf_int.viterbi_tags(encoded_dialogue_int,labels_mask,top_k=1)
        predicted_ints = cast(List[List[int]], [x[0][0] for x in best_paths_int])
        output['intentions'] = predicted_ints
        if intentions != None:
            class_probabilities = encoded_dialogue_int * 0.0
            for i, instance_tags in enumerate(predicted_ints):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1
            self.calculate_f1_int['F1-macro-int'](class_probabilities, intentions, labels_mask)
            self.calculate_f1_int['F1-class-int'](class_probabilities, intentions, labels_mask)
        '''
        loss
        '''
        if actions != None and intentions != None:
            log_likelihood_act = self.crf_act(encoded_dialogue_act, actions, labels_mask)
            log_likelihood_int = self.crf_int(encoded_dialogue_int, intentions, labels_mask)
            output["loss"] = (-log_likelihood_act) + (-log_likelihood_int)        
        return output
    
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = dict()
        '''
        actions metric
        '''
        act_macro = self.calculate_f1_act['F1-macro-act'].get_metric(reset)
        act_class = self.calculate_f1_act['F1-class-act'].get_metric(reset)
        metrics_to_return['Macro-act-f1'] = act_macro['fscore']
        idx2label = self.vocab.get_index_to_token_vocabulary(namespace='action_labels')
        for idx in range(len(act_class['fscore'])):
            lc= idx2label[idx]
            metrics_to_return[lc+'-act-f1'] = act_class['fscore'][idx]
        '''
        intentions metric
        '''
        int_macro = self.calculate_f1_int['F1-macro-int'].get_metric(reset)
        int_class = self.calculate_f1_int['F1-class-int'].get_metric(reset)
        metrics_to_return['Macro-int-f1'] = int_macro['fscore']
        idx2label = self.vocab.get_index_to_token_vocabulary(namespace='intention_labels')
        for idx in range(len(int_class['fscore'])):
            lc= idx2label[idx]
            metrics_to_return[lc+'-int-f1'] = int_class['fscore'][idx]
        '''
        average
        '''
        metrics_to_return['Macro-F1'] = (act_macro['fscore'] + int_macro['fscore']) / 2
        return metrics_to_return
    
