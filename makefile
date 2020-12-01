ODIR = build/obj
SRCDIR = src

_OBJ = hidato.o timer.o util.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_DEPS = timer.h util.h
DEPS = $(patsubst %,$(SRCDIR)/%,$(_DEPS))

default: directories hidato

hidato: $(OBJ)
	nvcc -o build/hidato $^

directories:
	mkdir -p build/obj

$(ODIR)/%.o: $(SRCDIR)/%.cu $(DEPS)
	nvcc -c -o $@ $<

clean:
	rm -rf build
