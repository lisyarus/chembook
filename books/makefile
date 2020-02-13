.ONESHELL:

all: *.pdf

%.pdf: %.tex
	mkdir -p build
	cd build
	pdflatex ../$<
	pdflatex ../$<
	mv $@ ..

.PHONY: clean
clean:
	rm -f *.pdf build/*
