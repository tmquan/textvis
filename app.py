import lightning as L
import gradio as gr
import pandas as pd
import numpy as np
import html
import random
import json
import stanza

import logging
logging.basicConfig(level=logging.INFO)

from functools import partial
from lightning.app.components.serve import ServeGradio

import markdown
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import transformers
import colorsys

def get_sentence_embeddings(sentence, size=384):
    # Placeholder implementation, replace with your sentence embedding method
    return np.random.rand(1, size)

def highlight_sentences(text, model):
    # return text

    paragraphs = text.splitlines(True)
    highlighted_html = []
    for paragraph in paragraphs:
        sentences = paragraph.split(".")
        for sentence in sentences:
            characters = re.findall(r'(?s)(.)', sentence)
            for character in characters:
                if character == '\n':
                    highlighted_html.append(character)
                    continue
                hue = random.uniform(0, 1)
                saturation = random.uniform(0.0, 0.5)
                value = random.uniform(0.5, 1)
                color = colorsys.hsv_to_rgb(hue, saturation, value)
                color = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
                highlighted_html.append(
                    f'<mark style="background-color: {color}">{character}</mark>')
    # Finally, join the span tags together into a single string
    return "".join(highlighted_html)

class TextVisualizationServeGradio(ServeGradio):
    inputs = []
    inputs.append(
        gr.Textbox(
            elem_id="itext",
            lines=15,
            label=f"Context",
            placeholder=f"Type a sentence or paragraph here."
        )
    )
    outputs = []
    outputs.append(
        gr.Markdown(
            elem_id="otext",
            label=f"Highlight",
        )
    )
    with open("data/Lorem/sent.txt", "r", encoding="utf8") as f:
        text = f.read()

    examples = [
        text, 
    ]
    
    def __init__(self, cloud_compute, *args, **kwargs):
        super().__init__(*args, cloud_compute=cloud_compute, **kwargs)
        self.ready = False  # required

    # Override original implementation to pass the custom css highlightedtext
    def run(self, *args, **kwargs):
        if self._model is None:
            self._model = self.build_model()

        # Partially call the prediction
        fn = partial(self.predict, *args, **kwargs)
        fn.__name__ = self.predict.__name__
        gr.Interface(
            fn=fn,
            # Override here
            css="#itext textarea {text-align: justify} #otext {text-align: justify}",
            # css="#itext span {white-space: pre-wrap; word-wrap: normal}; #otext div {white-space: pre-line;}",
            # css="p {text-align: justify;}",
            inputs=self.inputs,
            outputs=self.outputs,
            examples=self.examples
        ).launch(
            server_name=self.host,
            server_port=self.port,
            enable_queue=self.enable_queue,
            share=False,
        )

    def build_model(self):
        self.ready = True
        model = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
        return model

    def predict(self, text):
        return highlight_sentences(text=text, model=self._model)
        # return html.escape(texts)

        # nlp = stanza.Pipeline(lang='en', processors='tokenize')
        # doc = nlp(texts)
        # sentences = doc.sentences
        # words = [
        #     token.text for sentence in doc.sentences for token in sentence.tokens
        # ]
        # # for i, sentence in enumerate(doc.sentences):
        # #     print(f'====== Sentence {i+1} tokens =======')
        # #     print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')

        # # Remove duplicate words from text
        # seen = set()
        # result = []
        # for item in words:
        #     if item not in seen:
        #         seen.add(item)
        #         result.append(item)


        # # Create random sample weights for each unique word
        # weights = []
        # for i in range(len(result)):
        #     weights.append(random.random())
        
        # df_coeff = pd.DataFrame({
        #     'word': result,
        #     'num_code': weights
        # })


        # # Select the code value to generate different weights
        # word_to_coeff_mapping = {}
        # for row in df_coeff.iterrows():
        #     row = row[1]
        #     word_to_coeff_mapping[row[1]] = (row[0])

        # max_alpha = 0.8
        # highlighted_text = []
        # for word in words:
        #     weight = word_to_coeff_mapping.get(word)

        #     if weight is not None:
        #         highlighted_text.append(
        #             '<span style="background-color:rgba(135,206,250,' \
        #             + str(weight / max_alpha) + ');">' \
        #             + html_escape(word) \
        #             + '</span>'
        #         )
        #     else:
        #         highlighted_text.append(word)
        # highlighted_text = ' '.join(highlighted_text)
        # return highlighted_text
    
class LitRootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.textvis = TextVisualizationServeGradio(
            L.CloudCompute("cpu"), parallel=True)

    def configure_layout(self):
        tabs = []
        tabs.append({"name": "Text_Analytics", "content": self.textvis})
        return tabs

    def run(self):
        self.textvis.run()


app = L.LightningApp(LitRootFlow())
