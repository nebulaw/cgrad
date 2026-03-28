# cgrad

cgrad is a small autograd engine for c/c++. it is designed to stay readable,
easy to integrate, and easy to experiment with. the engine works on scalar
values, supports a small set of primitive operations, and builds a computation
graph for automatic differentiation. it has no external dependencies and is
meant to be a compact educational codebase rather than a full framework.

## build/test

to build and test the engine, you can use the following commands:

```sh
# in case of make
make build
make test

# using gcc
gcc -o test test.c cgrad.c -lm && ./test

# or clang
clang -o test test.c cgrad.c -lm && ./test
```

## examples

examples are in `examples/` and use a generic make interface:

```sh
# list all available examples
make list-examples

# build every example under examples/
make examples

# build a single example
make example EXAMPLE=xor

# build and run a single example
make run-example EXAMPLE=xor
```

or directly:

`gcc -I. -o example examples/<name>.c cgrad.c -lm && ./example`
