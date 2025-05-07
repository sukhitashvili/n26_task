import json
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
        self.field_descriptions_df = pd.read_csv(Path(__file__).parent / 'field_descriptions.csv')
        with open(Path(__file__).parent / 'extraction-fields.json', 'r') as f:
            self.extraction_fields = json.load(f)
        self.document_types = ['credit', 'garnishment', 'investment', 'personal account']
        self.vision_llm = ModelInference(path=self.model_id, device=self.device)

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

        general_info = self.extract_general_information(pil_img=whole_document_image)
        document_type = general_info['document_type']
        if document_type not in self.document_types:
            raise ValueError('Document type should be one of the following: {}'.format(', '.join(self.document_types)))

        document_type = '_'.join(document_type.split(' '))
        document_specific_fields = self.extraction_fields[document_type]
        document_specific_information = self.extract_document_specific_information(pil_img=whole_document_image,
                                                                                   fields_to_extract=document_specific_fields)
        document_specific_information.update(general_info)

        return document_specific_information

    def extract_general_information(self, pil_img: PIL.Image.Image) -> dict:
        fields_to_extract = self.extraction_fields['general']
        prompt = self.generate_prompt(fields_names=fields_to_extract)
        information = \
            self.vision_llm(messages=prompt, images=[pil_img], max_new_tokens=2000)[0]
        output_data = self.read_json(text=information)
        return output_data

    def extract_document_specific_information(self, pil_img: PIL.Image.Image, fields_to_extract: list[str]) -> dict:
        prompt = self.generate_prompt(fields_names=fields_to_extract)
        information = \
            self.vision_llm(messages=prompt, images=[pil_img], max_new_tokens=2000)[0]
        output_data = self.read_json(text=information)
        return output_data

    def generate_prompt(self, fields_names: list[str]):
        prompt_text = "You are provided pages of a bank document PDF file. Identify the following fields:\n"
        text_with_field_descriptions = self.get_field_descriptions(fields_of_interest=fields_names)
        json_format = self.get_json_format(field_names=fields_names)
        prompt_text += text_with_field_descriptions
        prompt_text += json_format
        prompt = self.get_image_prompt(text=prompt_text)
        return prompt

    @staticmethod
    def convert_to_img(document_page) -> PIL.Image.Image:
        page_img = document_page.get_pixmap()
        page_img = PIL.Image.frombytes("RGB", [page_img.width, page_img.height], page_img.samples)
        return page_img

    @staticmethod
    def read_json(text) -> dict:
        idx_1 = text.find('{')
        idx_2 = text.find('}')
        the_json_str = text[idx_1:idx_2 + 1]
        the_json = from_json(the_json_str, allow_partial=True)
        return the_json

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

    @staticmethod
    def get_json_format(field_names: list[str]) -> str:
        result = ("\nProvide answer in the JSON format where:\n"
                  "- Dates should follow the format 'dd.mm.yyyy'\n"
                  "- Names should follow the format 'ﬁrst_name last_name'\n"
                  "- Addresses should follow the format 'street_name street_number, city zipcode, country'\n"
                  "- Monetary amounts should follow the format 'amount currency_symbol' using a full stop as decimal"
                  " separator for the amount\n"
                  "- Additional ﬁelds must follow the same format as originally found in the document.\n")
        result += '```json\n'
        for field in field_names:
            json_string = json.dumps({field: f"ESTIMATED_{field.upper()}"})
            result += f'{json_string}\n'

        result += '```'
        return result

    @staticmethod
    def get_image_prompt(text: str):
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