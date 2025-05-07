import gc
from typing import Optional

import PIL
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoProcessor


class ModelInference:
    def __init__(self,
                 path: str,
                 device: str,
                 return_full_text: bool = False,
                 do_sample: bool = False,
                 top_p: float = 1,
                 temperature: Optional[float] = None,
                 ):
        self.path = path
        self.device = device
        self.return_full_text = return_full_text
        self.do_sample = do_sample
        self.use_grammar = False
        self.top_p = top_p
        self.temperature = temperature
        self.model, self.processor = get_model_and_tokenizer(base_model_folder_path=self.path,
                                                             lora_ckpt_path='',
                                                             device=self.device)

    @torch.no_grad()
    def predict(self,
                messages: list[dict],
                images: list[PIL.Image.Image] | None = None,
                max_new_tokens: int = 100) -> list[str]:
        """

        Args:
            messages: Format:
                        messages = [
                                        {"role": "user", "content": [
                                            {"type": "image"},
                                            {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
                                        ]}
                                    ]
            images: PIL images.
            max_new_tokens: Number of generated new tokens.

        Returns:

        """
        # Add chat-template text, but do not tokenize
        formatted_prompt = self.processor.apply_chat_template(messages,
                                                              tokenize=False,
                                                              add_special_tokens=True,
                                                              add_generation_prompt=True,
                                                              padding=False,
                                                              truncation=False,
                                                              )
        # tokenize the chatted-text
        input_ids = self.processor(text=formatted_prompt,
                                   images=images,
                                   add_special_tokens=False,  # special token were added already above
                                   padding=True,  # Add padding to the same length for all sentences
                                   truncation=True,  # Truncate sentences that are too long for the model
                                   return_tensors='pt')

        input_ids = {k: v.to(self.model.device) for k, v in input_ids.items()}

        output = self.model.generate(input_ids=input_ids["input_ids"],
                                     attention_mask=input_ids["attention_mask"],
                                     token_type_ids=input_ids["token_type_ids"],
                                     pixel_values=input_ids["pixel_values"],
                                     do_sample=self.do_sample,
                                     max_new_tokens=max_new_tokens,
                                     num_return_sequences=1,
                                     pad_token_id=self.processor.tokenizer.pad_token_id,
                                     eos_token_id=self.processor.tokenizer.eos_token_id,
                                     # top_k=1,
                                     # top_p=self.top_p,
                                     temperature=self.temperature,
                                     )
        # decode the predicted indexes
        if not self.return_full_text:
            output = output[:, input_ids["input_ids"].shape[1]:]  # The generated text was left padded, thus this works

        decoded_output = self.processor.batch_decode(output,
                                                     skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)
        return decoded_output

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


def clean_cuda_mem():
    gc.collect()
    torch.cuda.empty_cache()


# Below is the function from LLM training repo, just with underscore at the end of the name and added device param
def get_model_and_tokenizer_quantized(folder_path: str, device: str = 'auto') -> tuple:
    # load tokenizer
    processor = AutoProcessor.from_pretrained(folder_path, use_fast=True)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    # load model
    quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(folder_path,
                                                 quantization_config=quantization_config,
                                                 device_map=device,
                                                 torch_dtype=torch.bfloat16,
                                                 ).eval()
    return model, processor


def get_model_and_tokenizer(base_model_folder_path: str, lora_ckpt_path: str = '', device: str = 'auto') -> tuple:
    # load tokenizer
    base_model, processor = get_model_and_tokenizer_quantized(folder_path=base_model_folder_path,
                                                              device=device)
    processor.padding_side = 'left'  # for batched inference with decoder only models
    if lora_ckpt_path:
        raise ValueError("Lora Ckpt is not supported for this version!")
        # base_model = PeftModel.from_pretrained(base_model, model_id=lora_ckpt_path)  # load adapters of the base model
        # base_model = base_model.merge_and_unload()  # Does not undo quantization to merge properly
        # clean_cuda_mem()

    return base_model, processor


if __name__ == '__main__':
    import numpy as np
    model_path = 'google/gemma-3-4b-it'
    device = 'auto'
    inference_object = ModelInference(path=model_path, device=device)
    messages = [
        # {"role": "user", "content": [
        #     {"type": "text", "text": "You are a helpful assistant."}
        # ]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Print all the text line by line from the image."}
        ]}
    ]
    pil_img = PIL.Image.open('page-0.png').convert('RGB')
    # fake to be a two-page document
    np_image1 = np.array(pil_img)
    np_image2 = np.array(pil_img)
    concatenated = np.concatenate((np_image1, np_image2), axis=0)
    output = inference_object(messages=messages, images=[concatenated], max_new_tokens=4000)
    print(output)
