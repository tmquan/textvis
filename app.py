import lightning as L
import gradio as gr
import numpy as np
# import os
# os.system('python -m spacy download en_core_web_sm')

import json
import logging
logging.basicConfig(level=logging.INFO)

from functools import partial
from lightning.app.components.serve import ServeGradio

import spacy
from spacy import displacy

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
        gr.HighlightedText(
            elem_id="htext",
            label=f"Highlight",
            show_legend=True,
            combine_adjacent=True, 
            adjacent_separator="",
        ), #.style(color_map={"Harry": "green", "James Potter": "red"}, container=True),
    )
    outputs.append(
        gr.JSON(),
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
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(texts)  # doc is class Document
        pos_tokens = []

        for token in doc:
            pos_tokens.extend([(token.text, token.pos_)])
        return pos_tokens, json.dumps(pos_tokens)

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
