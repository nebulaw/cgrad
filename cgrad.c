#include "cgrad.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

Value *createconst(float float_value) {
  // plain scalar leaf: no parents, no local backward rule.
  Value *value = malloc(sizeof(Value));
  value->value = float_value;
  value->grad = 0.0f;
  value->requires_grad = 0;
  value->fn_op = FN_NONE;
  value->fn_type = FN_NONE;
  value->visited = 0;
  value->pa = NULL;
  value->pb = NULL;
  value->backward = bwnone;
  return value;
}

Value *createvalue(float value, Value *pa, Value *pb, int fn_op,
                   int requires_grad) {
  // op node: store result, wire parents, plug in backward rule later.
  Value *v = malloc(sizeof(Value));
  v->value = value;
  v->grad = 0.0f;
  v->fn_op = fn_op;
  v->requires_grad = requires_grad;
  v->fn_type = FN_NONE | fn_op;
  v->visited = 0;
  v->pa = pa;
  v->pb = pb;
  v->backward = bwnone;
  return v;
}

Value *createempty(void) { return createconst(0.0f); }

Value *copy(Value *value) {
  Value *v = malloc(sizeof(Value));
  v->value = value->value;
  v->grad = value->grad;
  v->fn_op = value->fn_op;
  v->requires_grad = value->requires_grad;
  v->fn_type = value->fn_type;
  v->visited = 0;
  v->pa = value->pa;
  v->pb = value->pb;
  v->backward = value->backward;
  return v;
}

void deletevalue(Value *value) { free(value); }

void deletevalues(int n, ...) {
  va_list ap;
  va_start(ap, n);
  for (int i = 0; i < n; i++) {
    free(va_arg(ap, Value *));
  }
  va_end(ap);
}

// dfs the graph, parents first, then stash the node for reverse-time.
static void buildtopo(Value *v, Value ***topo, int *count, int *capacity) {
  if (!v || v->visited)
    return;
  v->visited = 1;
  buildtopo(v->pa, topo, count, capacity);
  buildtopo(v->pb, topo, count, capacity);
  if (*count == *capacity) {
    *capacity = (*capacity == 0) ? 16 : (*capacity * 2);
    *topo = realloc(*topo, *capacity * sizeof(Value *));
  }
  (*topo)[(*count)++] = v;
}

// collect the whole graph once, then free each node exactly once.
void deletechain(Value *value) {
  if (!value)
    return;
  Value **topo = NULL;
  int count = 0, capacity = 0;
  buildtopo(value, &topo, &count, &capacity);
  for (int i = 0; i < count; i++) {
    free(topo[i]);
  }
  free(topo);
}

Value *fwadd(Value *a, Value *b) {
  assert(a != NULL && b != NULL);
  // forward pass: add now, remember how to split grad later.
  Value *v = createvalue(a->value + b->value, a, b, FN_ADD, 0);
  v->backward = bwadd;
  return v;
}

Value *fwmul(Value *a, Value *b) {
  assert(a != NULL && b != NULL);
  // multiply on the way forward, use the other input on the way back.
  Value *v = createvalue(a->value * b->value, a, b, FN_MUL, 0);
  v->backward = bwmul;
  return v;
}

Value *fwpow(Value *a, Value *b) {
  assert(a != NULL && b != NULL);
  // power op is just another node in the graph, same deal.
  Value *v = createvalue(powf(a->value, b->value), a, b, FN_POW, 0);
  v->backward = bwpow;
  return v;
}

Value *fwsub(Value *a, Value *b) {
  assert(a != NULL && b != NULL);
  // subtraction is add with a negated rhs. keep primitives small.
  Value *v = fwadd(a, fwmul(b, createconst(-1.0f)));
  return v;
}

Value *fwdiv(Value *a, Value *b) {
  assert(a != NULL && b != NULL);
  // division is multiply by b^-1, which lets pow do the heavy lifting.
  Value *v = fwmul(a, fwpow(b, createconst(-1.0f)));
  v->pb->requires_grad = b->requires_grad;
  return v;
}

void bwnone(Value *v) { (void)v; }

void bwadd(Value *value) {
  // z = x + y, so incoming grad flows straight to both parents.
  if (value->pa)
    value->pa->grad += value->grad;
  if (value->pb)
    value->pb->grad += value->grad;
}

void bwmul(Value *value) {
  // z = x * y, so each parent gets the other parent's value times dz.
  if (value->pa)
    value->pa->grad += value->pb->value * value->grad;
  if (value->pb)
    value->pb->grad += value->pa->value * value->grad;
}

// z = a^b. one grad for the base, one for the exponent when log is valid.
void bwpow(Value *value) {
  if (value->pa) {
    value->pa->grad +=
        (value->pb->value * powf(value->pa->value, value->pb->value - 1.0f)) *
        value->grad;
  }
  if (value->pb && value->pa->value > 0.0f) {
    value->pb->grad += (value->value * logf(value->pa->value)) * value->grad;
  }
}

// build a topo order, seed the output grad, then walk backward through time.
void backward(Value *value, int init_grad) {
  if (!value)
    return;
  Value **topo = NULL;
  int count = 0, capacity = 0;
  buildtopo(value, &topo, &count, &capacity);

  if (init_grad) {
    value->grad = 1.0f;
  }

  for (int i = count - 1; i >= 0; i--) {
    if (topo[i]->backward) {
      topo[i]->backward(topo[i]);
    }
    topo[i]->visited = 0;
  }
  free(topo);
}
