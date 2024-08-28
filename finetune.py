from data_example import ExampleData
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AdamW
import torch
from peft import get_peft_config, PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

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


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


model_path = "D:/llm-models/reranker/bge-reranker-v2-gemma/"
lora_path = "D:/llm-models/reranker/bge-reranker-v2-gemma-peft/"
save_merge_path = "D:/llm-models/reranker/bge-reranker-v2-gemma-tuned/"

model = AutoModelForCausalLM.from_pretrained(model_path,
                                                local_files_only=True).to('cuda')

tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                local_files_only=True)


dataset = CustomDataset(ExampleData.train_data)

batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define LoRA Config
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        'down_proj',
        'o_proj',
        'k_proj',
        'q_proj',
        'gate_proj',
        'up_proj',
        'v_proj'
        ]
)

# Wrap the model with PEFT
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()



yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
no_loc = tokenizer('No', add_special_tokens=False)['input_ids'][0]
optimizer = AdamW(model.parameters(), lr=3e-6)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        passages_A, passages_B, labels = batch
        pairs = list(zip(passages_A, passages_B))
        
        inputs = get_inputs(pairs, tokenizer)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        optimizer.zero_grad()
        
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits[:, -1, [no_loc, yes_loc]]
        
        loss = loss_fn(logits, torch.tensor([1 if label == 'Yes' else 0 for label in labels], device=device))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")


model.save_pretrained(lora_path)

# base_model_path = 'google/gemma-2b'
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             local_files_only=True)

model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()
model.save_pretrained(save_merge_path)
tokenizer.save_pretrained(save_merge_path)