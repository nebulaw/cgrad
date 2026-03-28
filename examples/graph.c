#include "cgrad.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "target/cgrad.dot";

  Value *a = createvalue(2.0f, NULL, NULL, FN_NONE, 1);
  Value *b = createvalue(-3.0f, NULL, NULL, FN_NONE, 1);
  Value *c = createvalue(10.0f, NULL, NULL, FN_NONE, 1);
  Value *d = fwmul(a, b);
  Value *e = fwadd(d, c);
  Value *f = fwpow(e, createconst(2.0f));

  backward(f, 1);

  FILE *out = fopen(path, "w");
  if (!out) {
    perror("fopen");
    deletechain(f);
    return EXIT_FAILURE;
  }

  dumpdot(f, out);
  fclose(out);

  printf("wrote graph to %s\n", path);

  deletechain(f);
  return EXIT_SUCCESS;
}
