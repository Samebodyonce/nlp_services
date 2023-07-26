from transformers import pipeline

class ModelsHandler:
    def __init__(self):
        self.models = {
            "cardiffnlp": self.create_model('cardiffnlp/twitter-xlm-roberta-base-sentiment'),
            "ivanlau": self.create_model('ivanlau/language-detection-fine-tuned-on-xlm-roberta-base'),
            "svalabs": self.create_model('svalabs/twitter-xlm-roberta-crypto-spam'),
            "EIStakovskii": self.create_model('EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus'),
            "jy46604790": self.create_model('jy46604790/Fake-News-Bert-Detect')
        }

    def create_model(self, model_name):
        try:
            model = pipeline('text-classification', model=model_name)
            return model
        except Exception as e:
            print(f"Failed to load model {model_name}. Error: {e}")
            return None

    def process_model(self, model, text):
        try:
            return model(text)[0] if model else {"error": "Model not loaded"}
        except Exception as e:
            return {"error": str(e)}
