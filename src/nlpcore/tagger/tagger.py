from abc import ABC, abstractmethod
from typing import Dict, List, Text


class Tagger(ABC):
    @abstractmethod
    def segment(self, text: Text, with_pos: bool = False) -> List[Dict[Text, Text]]:
        """ Segment text on given text

        Args:
            document (str): text.

        Returns:
            list of dict:
                {
                    "word": str,
                    "tag": str
                }
        """

    @abstractmethod
    def ner(self, text: Text) -> List[Dict]:
        """Named Entity Recognition on given text

        Args:
            document (str): text.

        Returns:
            list of dict:
                {
                    "start": int,
                    "end":int
                    "entity": str,
                    "value": str
                }
        """
