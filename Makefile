# include and lib directories
IDIR=/usr/local/include
LDIR=/usr/local/lib
# compiler
CC=gcc
# clags
CFLAGS=-Wall -Wextra -O2

# build and object directories
TDIR=target
ODIR=$(TDIR)/obj
BDIR=$(TDIR)/bin

# libraries
LIBS=-lm

# dependencies
DEPS=cgrad.h
# object files
OBJ=$(ODIR)/cgrad.o $(ODIR)/test.o
# executable name
TARGET=$(BDIR)/test

all: prebuild build install

prebuild:
	mkdir -p $(ODIR); mkdir -p $(BDIR)

build: prebuild $(OBJ)
	$(CC) -o $(TARGET) $(OBJ) $(CFLAGS) $(LIBS)

test:
	$(TARGET)

$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm -rf $(TDIR)
