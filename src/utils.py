import torch

def tokenize_sample(text, tokenizer):
    return tokenizer.encode_plus(
        text,
        max_length=512,
        add_special_tokens = True,
        padding = 'max_length',
        truncation = True,
        return_attention_mask = True,
        return_tensors = 'pt'
    )

def tokenize(df, tokenizer):
    token_id = []
    attention_masks = []

    for sample in df['text'].values:
        encoding_dict = tokenize_sample(sample, tokenizer)
        token_id.append(encoding_dict['input_ids'])
        attention_masks.append(encoding_dict['attention_mask'])

    tokens = torch.cat(token_id, dim=0)
    att_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['cls'].astype(int).values) - 1
    
    return tokens, labels, att_masks