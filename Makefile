all:
	python setup.py build_ext --inplace
	rm -rf build

install:
	pip install .
	rm -rf build