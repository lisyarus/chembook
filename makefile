.ONESHELL: book1.pdf
book1.pdf: book1.tex
	mkdir -p build
	cd build
	pdflatex ../book1.tex
	pdflatex ../book1.tex
	mv book1.pdf ..

.PHONY: clean
clean:
	rm -f book1.pdf build/*
