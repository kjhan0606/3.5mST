# Space ETC — imaging & slitless spectroscopy
# Override the interpreter if needed:  make PYTHON=/path/to/python gui
PYTHON ?= python3

DATA = etc_zodi.csv etc_qe.csv etc_throughput.csv
FIGS = etc_f5sigma.png etc_cooling.png

.PHONY: help data gui run figures validate clean

help:
	@echo "targets:"
	@echo "  make data      - build zodi/QE/throughput curves (downloads CALSPEC solar)"
	@echo "  make gui       - launch the Tk (Tcl/Tk) GUI"
	@echo "  make run       - CLI run + Roman/Euclid cross-check + cooling tradeoff"
	@echo "  make figures   - regenerate etc_f5sigma.png and etc_cooling.png"
	@echo "  make validate  - print the Roman/Euclid validation table"
	@echo "  make clean     - remove generated data and figures"
	@echo "  set PYTHON=... to choose the interpreter (needs numpy scipy matplotlib astropy)"

data: $(DATA)

$(DATA): make_etc_data.py
	$(PYTHON) make_etc_data.py

gui: $(DATA)
	$(PYTHON) etc_gui.py

run: $(DATA)
	$(PYTHON) slitless_etc.py --realistic --cooling

figures: $(DATA)
	$(PYTHON) slitless_etc.py --realistic --no-compare --cooling

validate:
	$(PYTHON) slitless_etc.py --no-compare >/dev/null; $(PYTHON) slitless_etc.py 2>/dev/null | sed -n '/CROSS-CHECK/,/deeper/p'

clean:
	rm -f $(DATA) $(FIGS)
