import torch.nn as nn
from transformers import MobileBertModel, MobileBertPreTrainedModel

class MobileBertForSequenceClassificationBCE(MobileBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.mobilebert = MobileBertModel(config)
        
        classifier_dropout = getattr(config, "classifier_dropout", 0.1)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.mobilebert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        logits = self.classifier(self.dropout(outputs.pooler_output)).squeeze(-1)
        return logits