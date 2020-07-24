import os
import json

AML_KEYWORD_PATH = os.path.join("assets", "keyword.json")

EMPHASIS = "0"
CRIMINAL = "1"
LEGAL = "2"
COMPLIANCE = "3"
GEO = "4"

ORDERED_CATEGORIES = [EMPHASIS, CRIMINAL, LEGAL, COMPLIANCE, GEO]


def load_keyword(lang):
    if lang == "zhtw":
        return Keyword(AML_KEYWORD_PATH)
    else:
        raise ValueError(
            "AML_KEYWORD_{0}_PATH does not exist".format(language)
            )


class Keyword(object):
    def __init__(self, path):
        with open(path, "r") as fin:
            self._kw_dict = json.load(fin)

        from nltk.stem.porter import PorterStemmer

        self._stemmer = PorterStemmer()

    def values(self, category):
        if not category in ORDERED_CATEGORIES:
            raise ValueError("invalid category: {0}".format(category))
        return self._kw_dict.get(category, [])

    def stemming(self, category):
        if not category in ORDERED_CATEGORIES:
            raise ValueError("invalid category: {0}".format(category))
        return [self._stemmer.stem(item) for item in self._kw_dict.get(category, [])]

    @property
    def emphasis(self):
        return self._kw_dict.get(EMPHASIS, [])

    @property
    def criminal(self):
        return self._kw_dict.get(CRIMINAL, [])

    @property
    def legal(self):
        return self._kw_dict.get(LEGAL, [])

    @property
    def compliance(self):
        return self._kw_dict.get(COMPLIANCE, [])

    @property
    def geo(self):
        return self._kw_dict.get(GEO, [])

    @property
    def all_values(self):
        words = []
        for item_list in self._kw_dict.values():
            words.extend(item_list)
            return list(set(words))

    @property
    def all_stemming_values(self):
        words = []
        for item_list in self._kw_dict.values():
            words.extend([self._stemmer.stem(item) for item in item_list])
            return list(set(words))
