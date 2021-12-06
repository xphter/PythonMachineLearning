import os;

from typing import Tuple, List, Dict;
from ImportNumpy import *;


class PTB:
    TRAIN_DATA_FILE_NAME = "ptb.train.txt";
    TEST_DATA_FILE_NAME = "ptb.test.txt";
    VALIDATE_LABELS_FILE_NAME = "ptb.valid.txt";


    def __init__(self, folderPath : str):
        words, self._word2ID, self._id2Word = self._loadVocab(os.path.join(folderPath, PTB.TRAIN_DATA_FILE_NAME));

        self._trainCorpus = np.array([self._word2ID[w] for w in words], dtype = np.int32);
        self._testCorpus = self._loadCorpus(os.path.join(folderPath, PTB.TEST_DATA_FILE_NAME));
        self._validateCorpus = self._loadCorpus(os.path.join(folderPath, PTB.VALIDATE_LABELS_FILE_NAME));


    @property
    def word2ID(self) -> Dict[str, int]:
        return self._word2ID;


    @property
    def id2Word(self) -> Dict[int, str]:
        return self._id2Word;


    @property
    def trainCorpus(self) -> np.ndarray:
        return self._trainCorpus;


    @property
    def testCorpus(self) -> np.ndarray:
        return self._testCorpus;


    @property
    def validateCorpus(self) -> np.ndarray:
        return self._validateCorpus;


    def _loadWords(self, filePath : str) -> List[str]:
        if not os.path.isfile(filePath):
            raise FileNotFoundError(filePath);

        with open(filePath, "rt") as file:
            words = file.read().replace('\n', '<eos>').strip().split();

        return words;

    def _loadVocab(self, filePath : str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
        wordID = 0;
        word2ID, id2Word = {}, {};
        words = self._loadWords(filePath);

        for word in words:
            if word in word2ID:
                continue;

            wordID = len(word2ID);
            word2ID[word] = wordID;
            id2Word[wordID] = word;

        return words, word2ID, id2Word;


    def _loadCorpus(self, filePath : str) -> np.ndarray:
        words = self._loadWords(filePath);
        corpus = np.array([self._word2ID[w] for w in words], dtype = np.int32);

        return corpus;
