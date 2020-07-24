import re
import logging


# from ..account.utils.date import RE_DATE

RE_DATE = re.compile(
    r"[\d零一二两三四五六七八九十]+\s*年\s*" + r"[\d一二三四五六七八九十]+\s*月\s*" + r"[\d一二三四五六七八九十]+\s*[号號日]"
)

# 記者王宏舜╱即時報導
# 項程鎮
# 陳志賢、蕭博文、王己由
# 中央社記者蔡沛琪台北2016年6月29日電
RE_REPORTER = re.compile(
    r"^(?:" + r"記者(\w{2,4})╱?(?:即時)?報導" + r"|" + r"中央社記者(\w{2,4})台北.+" + r")$"
)

# ETNOTW
# CTRTOT
RE_NEWS_SOURCE = re.compile(r"^[A-Z]{3,}[A-Z0-9]+$")


class News(object):
    """ AML News,
    id, url, content, subject, category, meta
    """

    def __init__(self, case_id=None, data={}):
        self._cid = case_id
        self._data = data
        self._seg = None
        self._lang = None
        self._title_tokens = None
        self._title_seg = None
        self._content_tokens = None
        self._content_seg = None
        self._publish_date = None
        self._metadata = None

        self._logger = logging.getLogger(__name__)

    def __getitem__(self, key):
        if key in self._data:
            return self._data[key]
        return None

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for case_id, data in self._data.items():
            yield case_id, data

    def set(self, key, value):
        self._data[key] = value

    def has(self, key):
        return key in self._data

    @property
    def has_top1(self):
        return self.has("top1")

    @property
    def has_top2(self):
        return self.has("top2")

    @property
    def has_top3(self):
        return self.has("top3")

    @property
    def segment_func(self):
        return self._seg

    @segment_func.setter
    def segment_func(self, fn):
        """ any function that return
        {'tokens': [{'word': xxx, 'pos': {'tag': yyy} }]}
        """
        self._seg = fn

    @property
    def id(self):
        return self._cid

    @property
    def lang(self):
        # TODO: any other type?
        return self._data.get("lang", "")

    @property
    def title(self):
        # TODO: any other type?
        return self._data.get("subject", "")

    @property
    def content(self):
        return self._data.get("content", "")

    @property
    def title_tokens(self):
        if not self._title_tokens:
            self._title_seg, self._title_tokens, lang = self._preprocess(
                self.title, self.lang
            )
        return self._title_tokens

    @property
    def title_seg(self):
        if not self._title_seg:
            self._title_seg, self._title_tokens = self._preprocess(
                self.title, self.lang
            )
        return self._title_seg

    @property
    def language(self):
        if not self._lang:
            tpl = self.content_tokens
        return self._lang

    @property
    def content_tokens(self):
        if not self._content_tokens:
            self._content_seg, self._content_tokens, self._lang = self._preprocess(
                self.content, self.lang
            )
        return self._content_tokens

    @property
    def content_seg(self):
        if not self._content_seg:
            self._content_seg, self._content_tokens = self._preprocess(
                self.content, self.lang
            )
        return self._content_seg

    def _preprocess(self, text, lang):
        """ Return segmented_text, tokens, language """
        text = text.replace("\n", " ")
        result_dict_list = self._seg(text, language=lang)
        tokens = ["{0}".format(t["word"]) for t in result_dict_list]
        return " ".join(tokens), result_dict_list, lang

    def generate_keywords(self, key_dicts={}, POS={}):
        results = {}
        for tokens in [self.title_tokens, self.content_tokens]:
            for token in tokens:
                if len(token["word"]) < 2:
                    continue
                if token["word"] in key_dicts:
                    self._logger.debug(token)
                elif token["tag"] in POS:
                    self._logger.debug(token)
                else:
                    continue
                results.setdefault(token["word"], 0)
                results[token["word"]] += 1
        self._data["keywords"] = " ".join(results.keys())
        return results


class Dataset(object):
    """ AML Dataset
    per customer
    """

    def __init__(self, customer_id=None, seg_func=None):
        self._cid = customer_id
        self._cases = {}
        self._seg_func = seg_func

    def add_case(self, case_id, data={}):
        """ read_AML_dataset method.

        :param str case_id: case id
        :param dict data: data in dict struct
        """
        news = News(case_id, data)
        news.segment_func = self._seg_func
        self._cases[case_id] = news

    def __getitem__(self, key):
        if key in self._cases:
            return self._cases[key]
        raise KeyError

    def __len__(self):
        return len(self._cases)

    def __iter__(self):
        for case_id, data in self._cases.items():
            yield case_id, data
