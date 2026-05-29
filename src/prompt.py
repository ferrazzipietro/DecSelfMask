from pydantic import BaseModel
from typing import List, Optional, Dict, Any



class PromptBase(BaseModel):
    input_text: Optional[str] = None
    prompt: Optional[str] = None
    item: Optional[str] = None
    answer: Optional[str] = None
    template: str
    tokens_before: int = 0
    tokens_after: int = 0
    
    def create_prompt_from_example(self):
        self.prompt = self.template.format(input_text=self.input_text, item=self.item, answer=self.answer)

    def _count_added_tkns_by_prompt(self, tokenizer):
        text_before, text_after = self.prompt.split(self.input_text)
        self.tokens_before = tokenizer(text_before, return_tensors="pt", add_special_tokens=False)['input_ids'].shape[1]
        self.tokens_after = tokenizer(text_after, return_tensors="pt", add_special_tokens=False)['input_ids'].shape[1]


class PromptNoGTMesh(BaseModel):
    input_text: Optional[str] = None
    prompt: Optional[str] = None
    template: str
    item: str = ''
    tokens_before: int = 0
    tokens_after: int = 0

    def __init__(self, note_text:str, target_item:str, tokenizer):
        
        template = self.make_template(tokenizer)
        super().__init__(input_text=note_text, template=template, item=target_item)
        self.create_prompt_from_example()
        self._count_added_tkns_by_prompt(tokenizer)
   
    def make_template(self, tokenizer):
        if tokenizer.chat_template is None:
            t = "{input_text}\n One of the mesh terms in this abstract is {item}"
            t = tokenizer.bos_token + t 
        else:
            u = "{input_text}\n One of the mesh terms in this abstract is"
            chat = [{"role":"user", "content": u}, {"role":"assistant", "content": "{item}"}]
            t = tokenizer.apply_chat_template(chat, tokenize=False)
        return t
    
    def create_prompt_from_example(self):
        self.prompt = self.template.format(input_text=self.input_text, item=self.item)

    def _count_added_tkns_by_prompt(self, tokenizer):
        text_before, text_after = self.prompt.split(self.input_text)
        self.tokens_before = tokenizer(text_before, return_tensors="pt", add_special_tokens=False)['input_ids'].shape[1]
        self.tokens_after = tokenizer(text_after, return_tensors="pt", add_special_tokens=False)['input_ids'].shape[1]


class PromptNoGTAdmission(BaseModel):
    input_text: Optional[str] = None
    prompt: Optional[str] = None
    template: str
    item: str = ''
    tokens_before: int = 0
    tokens_after: int = 0

    def __init__(self, note_text:str, target_item:str, tokenizer):
        
        template = self.make_template(tokenizer)
        super().__init__(input_text=note_text, template=template, item=target_item)
        self.create_prompt_from_example()
        self._count_added_tkns_by_prompt(tokenizer)
   
    def make_template(self, tokenizer):
        if tokenizer.chat_template is None:
            t = "{input_text}\n The patient admission type is {item}"
            t = tokenizer.bos_token + t 
        else:
            u = "{input_text}\n The patient admission type is"
            chat = [{"role":"user", "content": u}, {"role":"assistant", "content": "{item}"}]
            t = tokenizer.apply_chat_template(chat, tokenize=False)
        return t
    
    def create_prompt_from_example(self):
        self.prompt = self.template.format(input_text=self.input_text, item=self.item)

    def _count_added_tkns_by_prompt(self, tokenizer):
        text_before, text_after = self.prompt.split(self.input_text)
        self.tokens_before = tokenizer(text_before, return_tensors="pt", add_special_tokens=False)['input_ids'].shape[1]
        self.tokens_after = tokenizer(text_after, return_tensors="pt", add_special_tokens=False)['input_ids'].shape[1]



class PromptNoGT(BaseModel):
    input_text: Optional[str] = None
    prompt: Optional[str] = None
    template: str
    item: str = ''
    tokens_before: int = 0
    tokens_after: int = 0

    def __init__(self, note_text:str, target_item:str, tokenizer):
        
        template = self.make_template(tokenizer)
        super().__init__(input_text=note_text, template=template, item=target_item)
        self.create_prompt_from_example()
        self._count_added_tkns_by_prompt(tokenizer)
   
    def make_template(self, tokenizer):
        if tokenizer.chat_template is None:
            t = "{input_text}\n Data la storia del paziente, nel paziente si riscontra {item}"
            t = tokenizer.bos_token + t 
        else:
            u = "{input_text}\n Data la storia del paziente, nel paziente si riscontra"
            chat = [{"role":"user", "content": u}, {"role":"assistant", "content": "{item}"}]
            t = tokenizer.apply_chat_template(chat, tokenize=False)
        return t
    
    def create_prompt_from_example(self):
        self.prompt = self.template.format(input_text=self.input_text, item=self.item)

    def _count_added_tkns_by_prompt(self, tokenizer):
        text_before, text_after = self.prompt.split(self.input_text)
        self.tokens_before = tokenizer(text_before, return_tensors="pt", add_special_tokens=False)['input_ids'].shape[1]
        self.tokens_after = tokenizer(text_after, return_tensors="pt", add_special_tokens=False)['input_ids'].shape[1]


class PromptNoGTDyspneaClassification(BaseModel):
    input_text: Optional[str] = None
    prompt: Optional[str] = None
    template: str
    tokens_before: int = 0
    tokens_after: int = 0

    def __init__(self, note_text:str, tokenizer):
        
        template = self.make_template(tokenizer)
        super().__init__(input_text=note_text, template=template)
        self.create_prompt_from_example()
        self._count_added_tkns_by_prompt(tokenizer)
   
    def make_template(self, tokenizer):
        if tokenizer.chat_template is None:
            t = "{input_text}\n Data la storia del paziente, il paziente presenta\nDispnea"
            t = tokenizer.bos_token + t 
        else:
            u = "{input_text}\n Data la storia del paziente, il paziente presenta"
            chat = [{"role":"user", "content": u}, {"role":"assistant", "content": "Dispnea"}]
            t = tokenizer.apply_chat_template(chat, tokenize=False)
        return t
    
    def create_prompt_from_example(self):
        self.prompt = self.template.format(input_text=self.input_text)

    def _count_added_tkns_by_prompt(self, tokenizer):
        text_before, text_after = self.prompt.split(self.input_text)
        self.tokens_before = tokenizer(text_before, return_tensors="pt", add_special_tokens=False)['input_ids'].shape[1]
        self.tokens_after = tokenizer(text_after, return_tensors="pt", add_special_tokens=False)['input_ids'].shape[1]



class Prompt(PromptBase):

    def __init__(self, example: dict, span_pos, tokenizer):
        if 'spans' in example:
            item = example['spans'][span_pos]['labels'][0].split(' - ')[0]
            answer = example['spans'][span_pos]['labels'][0].split(' - ')[-1]
            print(f"item: {item} ---> {answer}")
        template = self.make_template(tokenizer)
        super().__init__(input_text=example['text'], item=item, answer=answer, template=template)
        self.create_prompt_from_example()
        self._count_added_tkns_by_prompt(tokenizer)
   
    def make_template(self, tokenizer):
        if tokenizer.chat_template is None:
            t = "{input_text}\n Data la storia del paziente, la {item} è {answer}"
            t = tokenizer.bos_token + t 
        else:
            u = "{input_text}\n Data la storia del paziente, la {item} è "
            chat = [{"role":"user", "content": u}, {"role":"assistant", "content": "{answer}"}]
            t = tokenizer.apply_chat_template(chat, tokenize=False)
        return t



class PromptNER(PromptBase):

    def __init__(self, example: dict, span_pos, tokenizer, entity_type='CLINENTITY'):
        if 'spans' in example:
            item = ''
            answer = example['spans'][span_pos]['text']
            print(f"item: {item} ---> {answer}")
        template = self.make_template(tokenizer, entity_type)
        super().__init__(input_text=example['text'], item=item, answer=answer, template=template)
        self.create_prompt_from_example()
        self._count_added_tkns_by_prompt(tokenizer)
   
    def make_template(self, tokenizer, entity_type):
        if tokenizer.chat_template is None:
            t = "f'Extract the {entity_type}'\n{input_text}\n {answer}"
            t = tokenizer.bos_token + t 
        else:
            chat = [{"role":"system", "content": f'Extract the {entity_type}'}, {"role":"user", "content": "{input_text}"}, {"role":"assistant", "content": "{answer}"}]
            t = tokenizer.apply_chat_template(chat, tokenize=False)
        return t

class PromptNERNoAssistant(PromptBase):

    def __init__(self, example: dict, answer, tokenizer, entity_type='CLINENTITY'):
        if 'spans' in example:
            item = ''
            print(f"item: {item} ---> {answer}")
        template = self.make_template(tokenizer, entity_type)
        super().__init__(input_text=example['text'], item=item, answer=answer, template=template)
        self.create_prompt_from_example()
        self._count_added_tkns_by_prompt(tokenizer)
   
    def make_template(self, tokenizer, entity_type):
        if tokenizer.chat_template is None:
            t = "f'Extract the {entity_type}'\n{input_text}\n {answer}"
            t = tokenizer.bos_token + t 
        else:
            chat = [{"role":"system", "content": f'Extract the {entity_type}'}, {"role":"user", "content": "{input_text}.\nEntity: {answer}"}] # , {"role":"assistant", "content": "}]
            t = tokenizer.apply_chat_template(chat, tokenize=False)
        return t
    
class PromptNERNoAnswer(PromptBase):

    def __init__(self, example: dict, tokenizer, entity_type='CLINENTITY'):
        if 'spans' in example:
            item = ''
        template = self.make_template(tokenizer, entity_type)
        super().__init__(input_text=example['text'], item=item, answer='', template=template)
        self.create_prompt_from_example()
        self._count_added_tkns_by_prompt(tokenizer)
   
    def make_template(self, tokenizer, entity_type):
        if tokenizer.chat_template is None:
            t = "f'Extract the {entity_type}'\n{input_text}\n"
            t = tokenizer.bos_token + t 
        else:
            chat = [{"role":"system", "content": f'Extract the {entity_type}'}, {"role":"user", "content": "{input_text}.\nEntity: "}] # , {"role":"assistant", "content": "}]
            t = tokenizer.apply_chat_template(chat, tokenize=False)
        return t