from torch import nn
from transformers import BartForConditionalGeneration


class BartLayer(nn.Module):
    def __init__(self, model_name):
        super(BartLayer, self).__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(model_name)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, source_ids, target_ids, source_att_mask):
        x = self.bart(
            input_ids=source_ids,
            attention_mask=source_att_mask,
            labels=target_ids,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
        )

        loss = x.loss
        logits = self.softmax(x.logits)
        return logits, loss

    def evaluate(self, source_ids, tokenizer, beam_size, max_length):
        candidate_ids = self.bart.generate(
            source_ids, num_beams=beam_size, max_length=max_length, early_stopping=True
        )
        return [
            tokenizer.decode(
                candidate, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for candidate in candidate_ids
        ]
