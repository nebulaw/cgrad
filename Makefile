CC ?= cc
CPPFLAGS ?= -I.
CFLAGS ?= -Wall -Wextra -O2
LDLIBS ?= -lm
RM ?= rm -rf

TDIR := target
ODIR := $(TDIR)/obj
BDIR := $(TDIR)/bin
EXAMPLES_DIR := examples

DEPS := cgrad.h
LIB_OBJ := $(ODIR)/cgrad.o
TEST_OBJ := $(ODIR)/test.o
TEST_BIN := $(BDIR)/test

EXAMPLE_SRCS := $(wildcard $(EXAMPLES_DIR)/*.c)
EXAMPLE_NAMES := $(patsubst $(EXAMPLES_DIR)/%.c,%,$(EXAMPLE_SRCS))
EXAMPLE_BINS := $(patsubst %,$(BDIR)/$(EXAMPLES_DIR)/%,$(EXAMPLE_NAMES))

all: build

build: $(TEST_BIN)

test: $(TEST_BIN)
	$(TEST_BIN)

examples: $(EXAMPLE_BINS)

list-examples:
	@printf "%s\n" $(EXAMPLE_NAMES)

ifeq ($(strip $(EXAMPLE)),)
example run-example:
	$(error EXAMPLE is required, use `make run-example EXAMPLE=xor`)
else
example: $(BDIR)/$(EXAMPLES_DIR)/$(EXAMPLE)

run-example: $(BDIR)/$(EXAMPLES_DIR)/$(EXAMPLE)
	$(BDIR)/$(EXAMPLES_DIR)/$(EXAMPLE)
endif

$(TEST_BIN): $(LIB_OBJ) $(TEST_OBJ) | $(BDIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $^ $(LDLIBS)

$(BDIR)/$(EXAMPLES_DIR)/%: $(LIB_OBJ) $(ODIR)/$(EXAMPLES_DIR)/%.o | $(BDIR)/$(EXAMPLES_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $^ $(LDLIBS)

$(ODIR)/%.o: %.c $(DEPS) | $(ODIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

$(ODIR)/$(EXAMPLES_DIR)/%.o: $(EXAMPLES_DIR)/%.c $(DEPS) | $(ODIR)/$(EXAMPLES_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

$(ODIR) $(BDIR) $(ODIR)/$(EXAMPLES_DIR) $(BDIR)/$(EXAMPLES_DIR):
	mkdir -p $@

.PHONY: all build test examples list-examples example run-example clean

clean:
	$(RM) $(TDIR)
