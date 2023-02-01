import lightning as L
import gradio as gr
import numpy as np

from functools import partial
from lightning.app.components.serve import ServeGradio

import flash
from flash.text import TextClassificationData, TextEmbedder
import logging
logging.basicConfig(level=logging.INFO)


class TextVisualizationServeGradio(ServeGradio):
    inputs = []
    inputs.append(
        gr.Textbox(
            lines=20, 
            elem_id="itext",
            label=f"Context",
            placeholder="Type a sentence or paragraph here."
        )
    )
    outputs = []
    outputs.append(
        # gr.HighlightedText(
        #     # value=[("Harry", "first_name"), ("James Potter", "last_name")],
        #     value=[("Harry", ""), ("James Potter", "")],
        #     elem_id="htext",
        #     label=f"Highlight",
        # ),
        gr.Textbox(
            lines=20, 
            elem_id="otext",
            label=f"Context", 
            placeholder="Result is displayed here."
        )
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
        pass

    def predict(self, texts):
        return texts


class LitRootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.textvis = TextVisualizationServeGradio(
            L.CloudCompute("cpu"), parallel=True)

    def configure_layout(self):
        tabs = []
        tabs.append({"name": "Text Visualization", "content": self.textvis})
        return tabs

    def run(self):
        self.textvis.run()


app = L.LightningApp(LitRootFlow())
