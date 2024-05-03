#include "cgrad.h"
#include <stdio.h>


int main(void)
{
  Value *a = createvalue(5, NULL, NULL, FN_NONE, 1); // 1
  Value *b = createvalue(2, NULL, NULL, FN_NONE, 1); // 2
  Value *c = fwmul(a, b);
  c->requires_grad = 1;
  Value *d = createvalue(0.8f, NULL, NULL, FN_NONE, 1); // 90
  Value *e = fwadd(c, d);
  e->requires_grad = 1;
  Value *f = createvalue(2.0f, NULL, NULL, FN_NONE, 1); // 3
  Value *g = fwpow(e, f);
  g->requires_grad = 1;
  Value *h = createvalue(8.0f, NULL, NULL, FN_NONE, 1); // 1
  Value *i = fwsub(g, h);
  i->requires_grad = 1;
  Value *j = createvalue(2.0f, NULL, NULL, FN_NONE, 1); // 2
  Value *k = fwmul(i, j);
  k->requires_grad = 1;
  Value *l = createvalue(3.0f, NULL, NULL, FN_NONE, 1); // 3
  Value *m = fwdiv(k, l);
  backward(m, 1);

  printf("a.grad: %f\n", a->grad);
  printf("b.grad: %f\n", b->grad);
  printf("c.grad: %f\n", c->grad);
  printf("d.grad: %f\n", d->grad);
  printf("e.grad: %f\n", e->grad);
  printf("f.grad: %f\n", f->grad);
  printf("g.grad: %f\n", g->grad);
  printf("h.grad: %f\n", h->grad);
  printf("i.grad: %f\n", i->grad);
  printf("j.grad: %f\n", j->grad);
  printf("k.grad: %f\n", k->grad);
  printf("l.grad: %f\n", l->grad);
  printf("m.grad: %f\n", m->grad);

  deletechain(m);
}

