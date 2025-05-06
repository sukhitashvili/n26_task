import json
import re
from pathlib import Path

import PIL
import numpy as np
import pandas as pd
import pymupdf
from pydantic_core import from_json

from model_inference import ModelInference


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
        self.vision_llm = ModelInference(path=self.model_id, device=self.device)
        # self.image_text_extraction_prompt = [
        #     {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        #     {"role": "user", "content": [
        #         {"type": "image"},
        #         {"type": "text",
        #          "text": "Extract all the text line by line from the image."}
        #     ]}
        # ]

    def extract(self, document_path: str):
        doc = pymupdf.open(document_path)
        document_pages = doc.page_count
        if document_pages > 2: raise ValueError('Document pages should not be more than 2 for performance reasons.')
        whole_document_image = None
        for page in range(document_pages):
            doc_page = doc.load_page(page)
            doc_page_pil_img = self.convert_to_img(doc_page)
            if whole_document_image is None:
                whole_document_image = np.array(doc_page_pil_img)
            else:
                whole_document_image = np.concatenate((whole_document_image, doc_page_pil_img), axis=0)

        general_info = self.get_general_information(pil_img=whole_document_image)

        print(general_info)

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
    def read_json(text):
        # Extract content between triple backticks (optionally with json tag)
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            content = match.group(1)
            # Split into lines and parse each line as a JSON object
            lines = content.strip().splitlines()
            result = {}
            for line in lines:
                obj = from_json(line, allow_partial=True)
                result.update(obj)
            return result
        return {}

    def get_general_information(self, pil_img: PIL.Image.Image):
        fields_to_extract = self.extraction_fields['general']
        text_with_field_descriptions = self.get_field_descriptions(fields_of_interest=fields_to_extract)
        json_format = self.get_json_format(field_names=fields_to_extract)
        prompt_text = "You are provided pages of a bank document PDF file. Identify the following fields:\n"
        prompt_text += text_with_field_descriptions
        prompt_text += json_format
        prompt = self.get_image_prompt(text=prompt_text)
        information = \
            self.vision_llm(messages=prompt, images=[pil_img], max_new_tokens=2000)[0]
        return information

    def get_document_spefic_information(self):
        pass

    def get_field_descriptions(self, fields_of_interest: list[str]) -> str:
        df = self.field_descriptions_df
        result_string = ""
        for field in fields_of_interest:
            match = df[df['Field Name'] == field]
            if not match.empty:
                description = match.iloc[0]['Description']
                result_string += f"{field}: {description}.\n"
            else:
                raise ValueError("Field Description does not exist!")
        return result_string

    def get_json_format(self, field_names: list[str]) -> str:
        result = "\nProvide answer in the following JSON format:\n"
        result += '```json\n'
        for field in field_names:
            json_string = json.dumps({field: f"ESTIMATED_{field.upper()}"})
            result += f'{json_string}\n'

        result += '```'
        return result

    def get_image_prompt(self, text: str):
        prompt = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text",
                 "text": text}
            ]}
        ]
        return prompt


if __name__ == '__main__':
    c = DocumentExtractor()
    c.extract(document_path='documents/doc-01.pdf')
    # print(c.get_json_format(['document_id', 'document_type']))
