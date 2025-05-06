import json
import re
from pathlib import Path

import PIL
import pandas as pd
import pymupdf
from pydantic_core import from_json

from model_inference import clean_cuda_mem


class DocumentExtractor:
    def __init__(self, model_id: str = 'gemma-3-4b-it-vllm'):
        self.device = 'auto'
        self.model_id = model_id
        with open(Path(__file__).parent / 'extraction-fields.json', 'r') as f:
            self.extraction_fields = json.load(f)
        self.field_descriptions_df = pd.read_csv(Path(__file__).parent / 'field_descriptions.csv')
        self.document_types = list(self.extraction_fields.keys())
        self.document_types.pop(0)
        self.document_types = [' '.join(i.split('_')).title() for i in self.document_types]  # beautify the text
        # self.vision_llm = ModelInference(path=self.model_id, device=self.device)
        self.image_text_extraction_prompt = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text",
                 "text": "Extract all the text line by line from the image."}
            ]}
        ]

    def extract(self, document_path: str):
        doc = pymupdf.open(document_path)
        document_pages = doc.page_count
        whole_document_text = ''
        for page in range(document_pages):
            doc_page = doc.load_page(page)
            doc_page_pil_img = self.convert_to_img(doc_page)
            doc_page_text = self.extract_text_from_image(doc_page_pil_img)
            whole_document_text += doc_page_text if whole_document_text == '' else f'\n{doc_page_text}'

        document_language = self.get_text_language(document_text=whole_document_text)

    def extract_text_from_image(self, pil_img: PIL.Image.Image) -> str:
        img_text = self.vision_llm(messages=self.image_text_extraction_prompt, images=[pil_img], max_new_tokens=2000)[0]
        clean_cuda_mem()
        return img_text

    def get_text_language(self, document_text: str) -> str:
        prompt = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                  {"role": "user", "content": [
                      {"type": "text", "text": 'I provide you a text from a bank document. Which language do you '
                                               'think it is written in from these ones {English, German, French, '
                                               'Spanish, Italian}? Provide answer in the following JSON format: '
                                               '```json\n{"language": "SELECTED_LANGUAGE"}\n```.'
                                               f'The document text:\n{document_text}'}
                  ]}
                  ]
        model_answer = self.vision_llm(messages=prompt, max_new_tokens=200)[0]
        model_answer = self.extract_json(model_answer)
        language = model_answer['language']
        return language

    @staticmethod
    def convert_to_img(document_page) -> PIL.Image.Image:
        page_img = document_page.get_pixmap()
        page_img = PIL.Image.frombytes("RGB", [page_img.width, page_img.height], page_img.samples)
        return page_img

    @staticmethod
    def extract_json(text):
        # match content between triple backticks and optionally a language identifier
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            text = match.group(1)
        return from_json(text, allow_partial=True)

    def get_general_information(self):
        field_to_extract = self.extraction_fields['general']
        text_with_field_descriptions = self.get_field_descriptions(fields_of_interest=field_to_extract)

    def get_document_spefic_information(self):
        pass

    def get_field_descriptions(self, fields_of_interest: list[str]) -> str:
        df = self.field_descriptions_df
        result_string = ""
        for field in fields_of_interest:
            match = df[df['Field Name'] == field]
            if not match.empty:
                description = match.iloc[0]['Description']
                result_string += f"{field}: {description}\n"
            else:
                raise ValueError("Field Description does not exist!")
        return result_string

    def get_json_format(self, field_names: list[str]) -> str:
        result = '```json\n'
        for field in field_names:
            json_string = json.dumps({field: f"ESTIMATED_{field.upper()}"})
            result += f'{json_string}\n'

        result += '```'
        return result


if __name__ == '__main__':
    c = DocumentExtractor()
    # c.get_general_information()
    print(c.get_json_format(['document_id', 'document_type']))
