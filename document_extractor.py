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

    def get_general_information(self, pil_img: PIL.Image.Image) -> dict:
        fields_to_extract = self.extraction_fields['general']
        text_with_field_descriptions = self.get_field_descriptions(fields_of_interest=fields_to_extract)
        json_format = self.get_json_format(field_names=fields_to_extract)
        prompt_text = "You are provided pages of a bank document PDF file. Identify the following fields:\n"
        prompt_text += text_with_field_descriptions
        prompt_text += json_format
        prompt = self.get_image_prompt(text=prompt_text)
        information = \
            self.vision_llm(messages=prompt, images=[pil_img], max_new_tokens=2000)[0]
        output_data = self.read_json(text=information)
        return output_data

    def get_document_specific_information(self):
        pass

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

    def get_json_format(self, field_names: list[str]) -> str:
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
