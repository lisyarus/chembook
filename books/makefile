PDFLATEX=pdflatex -shell-escape -interaction=nonstopmode -halt-on-error

.ONESHELL:

all: *.pdf

%.pdf: %.tex images/%/*.png
	mkdir -p build
	ls build/images || ln -sv ../images build/images
	cd build
	$(PDFLATEX) ../$<
	$(PDFLATEX) ../$<
	mv $@ ..

.PHONY: clean
clean:
	rm -f *.pdf build/*
