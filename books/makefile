.ONESHELL:

all: *.pdf

%.pdf: %.tex images/*.png
	mkdir -p build
	ls build/images || ln -sv ../images build/images
	cd build
	pdflatex -shell-escape -interaction=nonstopmode -halt-on-error ../$<
	pdflatex -shell-escape -interaction=nonstopmode -halt-on-error ../$<
	mv $@ ..

.PHONY: clean
clean:
	rm -f *.pdf build/*
