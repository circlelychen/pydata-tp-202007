.PHONY: clean test lint init check-readme

JOBS ?= 1

help:
	@echo "make"
	@echo "    prepare-ckiptagger-model"
	@echo "        Downlad ckiptagger model fies."
	@echo "    prepare-spacy-en-model"
	@echo "        Downlad spacy english model fies."

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	find . -name '__pycache__' -type d -exec rm -rf  {} +

prepare-spacy-en-model:
	pip install --no-index ./pypi/en_core_web_sm-2.3.0.tar.gz
	python -m spacy link en_core_web_sm en

prepare-ckiptagger-model:
	wget --no-check-certificate --no-proxy --progress=dot:giga -N http://ckip.iis.sinica.edu.tw/data/ckiptagger/data.zip
	mkdir -pv data/models
	unzip data.zip -d data/models/ckiptagger
	rm -f data.zip
