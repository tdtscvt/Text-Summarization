from flask import Flask, request, render_template
from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, pipeline
from textSummarizer.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

        pipe = pipeline("summarization", model=self.config.model_path, tokenizer=tokenizer)
        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        return output

prediction_pipeline = PredictionPipeline()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form.get('text')
    if not text:
        return render_template('index.html', summary="No text provided.")
    
    summary = prediction_pipeline.predict(text)
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
