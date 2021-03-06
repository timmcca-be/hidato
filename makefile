ODIR = build/obj
SRCDIR = src
TESTDIR = test

_OBJ = hidato.o timer.o util.o board.o score.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_DEPS = timer.h util.h
DEPS = $(patsubst %,$(SRCDIR)/%,$(_DEPS))

default: directories hidato

test: directories testProgram

testProgram: $(ODIR)/util.o $(ODIR)/test.o
	nvcc -o build/test $^

hidato: $(OBJ)
	nvcc -o build/hidato $^

directories:
	mkdir -p build/obj

$(ODIR)/%.o: $(SRCDIR)/%.cu $(DEPS)
	nvcc -c -o $@ $<

$(ODIR)/%.o: $(TESTDIR)/%.cu $(DEPS)
	nvcc -c -o $@ $<

clean:
	rm -rf build
