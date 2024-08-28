
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_example import ExampleData

def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
    if prompt is None:
        # prompt = "Given a passage A and a passage B, determine whether passage A contains overlapping information by providing a prediction of either 'Yes' or 'No'."
        prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    sep = "\n"
    prompt_inputs = tokenizer(prompt,
                              return_tensors=None,
                              add_special_tokens=False)['input_ids']
    sep_inputs = tokenizer(sep,
                           return_tensors=None,
                           add_special_tokens=False)['input_ids']
    inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(f'A: {query}',
                                 return_tensors=None,
                                 add_special_tokens=False,
                                 max_length=max_length * 3 // 4,
                                 truncation=True)
        passage_inputs = tokenizer(f'B: {passage}',
                                   return_tensors=None,
                                   add_special_tokens=False,
                                   max_length=max_length,
                                   truncation=True)
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs['input_ids'],
            sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)
    return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
    )

model_path = "D:/llm-models/reranker/bge-reranker-v2-gemma-tuned/"

model = AutoModelForCausalLM.from_pretrained(model_path,
                                                local_files_only=True).to('cuda')

tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                local_files_only=True)

yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
no_loc = tokenizer('N0', add_special_tokens=False)['input_ids'][0]

eval = {"Yes": [], "No": []}

for item in ExampleData.example_data:

    pair = [(item[0], item[1])]
    label = item[2]
    with torch.no_grad():
        inputs = get_inputs(pair, tokenizer).to('cuda')
        scores = model(**inputs, return_dict=True).logits[:, -1, [yes_loc]].view(-1, ).float()
        eval[label].append(scores[0].to('cpu').float())
        print(item[0], item[1], scores[0].to('cpu').float().numpy())
