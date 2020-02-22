.ONESHELL:

all: *.pdf

%.pdf: %.tex
	mkdir -p build
	cd build
	pdflatex --shell-escape ../$<
	pdflatex --shell-escape ../$<
	mv $@ ..

.PHONY: clean
clean:
	rm -f *.pdf build/*
