import os
from transformers import BertTokenizer


flag3d_coarse_action_description = {}
flag3d_text = []
i = 0
for text_name in os.listdir('data/flag3d_txt/'):
    if text_name.endswith('002.txt'):
        f = open('data/flag3d_txt/' + text_name)
        line = f.readline()
        flag3d_coarse_action_description[i] = line
        flag3d_text.append(line)
        i += 1
# print(flag3d_text)
# print(flag3d_coarse_action_description)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_input = tokenizer(flag3d_text,padding='max_length',
                       max_length =750,
                       truncation=True,
                       return_tensors="pt")
print(bert_input['input_ids'])
print(bert_input['token_type_ids'])
print(bert_input['attention_mask'])
max = 0
for te in flag3d_text:
    if len(te) > max:
        max = len(te)
print(max)
