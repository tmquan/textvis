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


# Prevent special characters like & and < to cause the browser to display something other than what you intended.
def html_escape(text):
    return html.escape(text)

class TextVisualizationServeGradio(ServeGradio):
    inputs = []
    inputs.append(
        gr.Textbox(
            elem_id=f"itext",
            lines=15,
            label=f"Context",
            placeholder=f"Type a sentence or paragraph here."
        )
    )
    outputs = []
    outputs.append(
        gr.HTML()
    )
    with open("data/Harry_Potter_Corpora/HarryJamesPotter.txt", "r", encoding="utf8") as f:
        HarryJamesPotter = f.read()

    examples = [
        HarryJamesPotter, 
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
            css="#htext span {white-space: pre-wrap; word-wrap: normal}",
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
        pass

    def predict(self, texts):
        return texts
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
