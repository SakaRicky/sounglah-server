from flask import Flask
from flask_restful import Api, reqparse
from flask_cors import CORS
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

app = Flask(__name__)
api = Api(app)
CORS(app)

translate_put_args = reqparse.RequestParser()
translate_put_args.add_argument('srcLanguage', type=str, help="Source Language missing", required=True)
translate_put_args.add_argument('targetLanguage', type=str, help="Target Language missing", required=True)
translate_put_args.add_argument('text', type=str, help="Text to translate missing", required=True)

model_checkpoint = "Helsinki-NLP/opus-mt-en-mul"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained("rickySaka/eng-med")

@app.route("/translate", methods=["GET"])
def get():
    return "The server can now translate."

@app.route("/translate", methods=["POST"])
def post():
    args = translate_put_args.parse_args()
    translated = model.generate(**tokenizer(args["text"], return_tensors="pt", padding=True))
    with tokenizer.as_target_tokenizer():
        results = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    results
    print(results)

    return {"translate": {
        "srcLanguage": args["srcLanguage"], "targetLanguage": args["targetLanguage"], "translatedText": results
    }}

# api.add_resource(Translate, '/translate')

if __name__ == "__main__":
    app.run(threaded=True)