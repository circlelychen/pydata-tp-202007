import os
import logging

from nlpcore.tagger.mytagger import MyTagger

DICT_PATH = os.path.join("assets/ckiptagger_dict.txt")

class NLPClient(object):
    def __init__(self, CKIPTAGGER_MODEL_PATH=None):
        self.logger = logging.getLogger(__name__)

        model_path = CKIPTAGGER_MODEL_PATH
        if not model_path:
            model_path = os.getenv("CKIPTAGGER_MODEL_PATH", None)

        self._mytagger = MyTagger(model_path=model_path, dict_path=DICT_PATH)

    def check_models(self):
        self._mytagger.check_model_and_load()

    def post_document_analyze_syntax(self, text, language):
        if language not in ("zhtw"):
            raise ValueError("{0} does not support".format(language))
        self.check_models()
        return self._mytagger.segment(text, with_pos=True)
