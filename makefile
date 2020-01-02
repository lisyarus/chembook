.ONESHELL: book.pdf
book.pdf: book.tex clean
	cd build
	pdflatex ../book.tex
	pdflatex ../book.tex
	mv book.pdf ..
	
.PHONY: clean
clean:
	rm -f book.pdf build/*
