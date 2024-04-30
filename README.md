# cgrad

cgrad is simple yet fast autograd engine for c/c++. it supposed to be
educational and easy to understand. cgrad engine is based on small set of
operations(+,\*,^) and builds an internal graph for automatic differentiations.
it contains no external dependencies and can be easily integrated into any
project. Ye, Karpathy and his micrograd is the inspiration of the project :3.

## build/test

To build and test the engine, you can use the following commands:

```sh
# in case of make
make build
make test

# using gcc
gcc -o test test.c cgrad.c -lm && ./test

# or clang
clang -o test test.c cgrad.c -lm && ./test
```
