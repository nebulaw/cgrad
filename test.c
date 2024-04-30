#include "cgrad.h"
#include <stdio.h>

int main(void)
{
  Value a = createvalue(5, NULL, NULL, FN_NONE, 1); // 1
  Value b = createvalue(2, NULL, NULL, FN_NONE, 1); // 2
  Value c = fwmul(&a, &b);
  c.requires_grad = 1;
  Value d = createvalue(0.8f, NULL, NULL, FN_NONE, 1); // 90
  Value e = fwadd(&c, &d);
  e.requires_grad = 1;
  Value f = createvalue(2.0f, NULL, NULL, FN_NONE, 1); // 3
  Value g = fwpow(&e, &f);
  backward(&g, 1);

  printf("a.grad: %f\n", a.grad);
  printf("b.grad: %f\n", b.grad);
  printf("c.grad: %f\n", c.grad);
  printf("d.grad: %f\n", d.grad);
  printf("e.grad: %f\n", e.grad);
  printf("f.grad: %f\n", f.grad);
  printf("g.grad: %f\n", g.grad);
}

