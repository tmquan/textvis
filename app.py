import lightning as L
import gradio as gr
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

from functools import partial
from lightning.app.components.serve import ServeGradio

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import transformers
from nltk.tokenize import sent_tokenize

def get_sentence_embeddings(sentence, size=384):
    # Placeholder implementation, replace with your sentence embedding method
    return np.random.rand(1, size)

def highlight_uniform_sentences(text, model):
    # return text
    paragraphs = [p for p in text.split('\n') if p]
    sentences = sent_tokenize(text)

    highlighted_html = []
    empty = f'#{int(255):02x}{int(255):02x}{int(255):02x}'
    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)
        for sentence in sentences:
            characters = re.findall(r'(?s)(.)', sentence)
            # characters = "".join([token.text for token in sentence.tokens])
            for idx, character in enumerate(characters):
                color = plt.cm.hsv(idx/len(characters))
                color = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
                highlighted_html.append(
                    f'<mark style="background-color: {color}">{character}</mark>')
            highlighted_html.append(f'<mark style="background-color: {empty}">{" "}</mark>')
        highlighted_html.append(f'<br>')
        highlighted_html.append(f'<br>')
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
    # with open("data/Lorem/sent.txt", "r", encoding="utf8") as f:
    with open("data/Harry_Potter_Corpora/Harry_James_Potter.txt", "r", encoding="utf8") as f:
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
        return highlight_uniform_sentences(text=text, model=self._model)
        
    
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
