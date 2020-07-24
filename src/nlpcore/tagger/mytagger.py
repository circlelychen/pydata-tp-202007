import os
import logging
from typing import Dict, List, Text

from ckiptagger import construct_dictionary, WS, POS

from .tagger import Tagger


class MyTagger(Tagger):
    def __init__(self, model_path=None, dict_path=None, coerce_dict=None):
        self._logger = logging.getLogger(__name__)

        self._recommend_dict = {}
        if dict_path:
            self._recommend_dict = construct_dictionary(self.load_userdict(dict_path))

        self._coerce_dict = {}
        if coerce_dict:
            self._coerce_dict = construct_dictionary(self.load_userdict(coerce_dict))

        self._model_path = ""
        if model_path:
            self._model_path = model_path

        self._ws = None
        self._pos = None

    def check_model_and_load(self):
        # To use GPU:
        #    1. Install tensorflow-gpu (see Installation)
        #    2. Set CUDA_VISIBLE_DEVICES environment variable, e.g. os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        #    3. Set disable_cuda=False, e.g. ws = WS("./data", disable_cuda=False)

        # Do not use CPU:
        disable_cuda = True
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            # To use CPU:
            disable_cuda = False

        if not self._ws or not self._pos:
            self._logger.info("ckiptagger WS/POS: Model Loading...")
            self._ws = WS(self._model_path, disable_cuda=disable_cuda)
            self._pos = POS(self._model_path, disable_cuda=disable_cuda)
            self._logger.info("ckiptagger WS/POS: Model Done...")

    @staticmethod
    def load_userdict(path):
        word_to_weigth = {}
        with open(path, "rb") as fin:
            for lineno, ln in enumerate(fin, 1):
                line = ln.strip()
                if not isinstance(line, Text):
                    try:
                        line = line.decode("utf-8").lstrip("\ufeff")
                    except UnicodeDecodeError:
                        raise ValueError("dictionary file %s must be utf-8" % path)
                if not line:
                    continue
                line = line.strip()
                word, freq = line.split(" ")[:2]
                word_to_weigth[word] = freq
                return word_to_weigth

    def segment(self, text: Text, with_pos: bool = False) -> List[Dict[Text, Text]]:
        """ Segment text on given text

        Args:
        text (str): text.

        Returns:
        list of dict:
        {
        "word": str,
        "tag": str
        }
        """
        self.check_model_and_load()

        ckip_tokens = self._ws(
            [text],
            recommend_dictionary=self._recommend_dict,
            coerce_dictionary=self._coerce_dict,
            )
        if with_pos:
            pos_list = self._pos(ckip_tokens)

        result = []
        if with_pos:
            for word, pos in zip(ckip_tokens[0], pos_list[0]):
                result.append({"word": word, "tag": pos})
        else:
            for word in zip(ckip_tokens[0]):
                result.append({"word": word, "tag": ""})

        return result

    def ner(self, text: Text) -> List[Dict]:
        """Named Entity Recognition on given text

        Args:
        text (str): text.

        Returns:
        list of dict:
        {
        "start": int,
        "end":int
        "entity": str,
        "value": str
        }
        """
        raise NotImplementedError()
