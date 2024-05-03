#include "cgrad.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>


Value *createconst(float float_value)
{
  Value *value = malloc(sizeof(Value));
  value->value = float_value;
  value->grad = 0.0f;
  value->value = float_value;
  value->grad = 0.0f;
  value->out_grad = 0.0f;
  value->requires_grad = 0;
  value->fn_op = FN_NONE;
  value->fn_type = FN_NONE;
  value->pa = NULL;
  value->pb = NULL;
  value->backward = bwnone;
  return value;
}

Value *createvalue(float value, Value *pa, Value *pb, int fn_op, int requires_grad)
{
  Value *v = malloc(sizeof(Value));
  v->value = value;
  v->grad = 0.0f;
  v->out_grad = 0.0f;
  v->fn_op = fn_op;
  v->requires_grad = requires_grad;
  v->pa = pa;
  v->pb = pb;
  v->fn_type = FN_NONE | fn_op;
  v->backward = bwnone;
  return v;
}

Value *createempty(void)
{
  return createconst(0.0f);
}

Value *copy(Value *value)
{
  Value *v = malloc(sizeof(Value));
  v->value = value->value;
  v->grad = value->grad;
  v->out_grad = value->out_grad;
  v->fn_op = value->fn_op;
  v->requires_grad = value->requires_grad;
  v->pa = value->pa;
  v->pb = value->pb;
  v->fn_type = value->fn_type;
  v->backward = value->backward;
  return v;
}

void deletevalue(Value *value)
{
  free(value);
}

void deletevalues(int n, ...)
{
  va_list ap;
  va_start(ap, n);
  for (int i = 0; i < n; i++) {
    free(va_arg(ap, Value *));
  }
  va_end(ap);
}

void deletechain(Value *value)
{
  if (!value) {
    return;
  }
  if (value->pa) {
    deletechain(value->pa);
  }
  if (value->pb) {
    deletechain(value->pb);
  }
  deletevalue(value);
}

Value *fwadd(Value *a, Value *b)
{
  assert(a != NULL && b != NULL);
  Value *v = createvalue(a->value + b->value, a, b, FN_ADD, 0);
  v->backward = bwadd;
  return v;
}

Value *fwmul(Value *a, Value *b)
{
  assert(a != NULL && b != NULL);
  Value *v = createvalue(a->value * b->value, a, b, FN_MUL, 0);
  v->backward = bwmul;
  return v;
}

Value *fwpow(Value *a, Value *b)
{
  assert(a != NULL && b != NULL);
  Value *v = createvalue(powf(a->value, b->value), a, b, FN_POW, 0);
  v->backward = bwpow;
  return v;
}

Value *fwsub(Value *a, Value *b)
{
  assert(a != NULL && b != NULL);
  Value *v = fwadd(a, fwmul(b, createconst(-1.0f)));
  v->backward = bwadd;
  return v;
}

Value *fwdiv(Value *a, Value *b)
{
  assert(a != NULL && b != NULL);
  Value *v = fwmul(a, fwpow(b, createconst(-1.0f)));
  v->pb->requires_grad = b->requires_grad;
  return v;
}

void bwnone(Value *)
{
  // This does nothing
}

void bwadd(Value *value)
{
  if (value->pa && value->pa->requires_grad) {
    if (value->requires_grad) {
      value->pa->grad += value->grad;
    } else {
      value->pa->grad += value->out_grad;
    }
  } else if (value->pa) {
    if (value->requires_grad) {
      value->pa->out_grad += value->grad;
    } else {
      value->pa->out_grad += value->out_grad;
    }
  }
  if (value->pb && value->pb->requires_grad) {
    if (value->requires_grad) {
      value->pb->grad += value->grad;
    } else {
      value->pb->grad += value->out_grad;
    }
  } else if (value->pb) {
    if (value->requires_grad) {
      value->pb->out_grad += value->grad;
    } else {
      value->pb->out_grad += value->out_grad;
    }
  }
}

void bwmul(Value *value)
{
  if (value->pa && value->pb) {
    if (value->pa->requires_grad) {
      if (value->requires_grad) {
        value->pa->grad += value->grad * value->pb->value;
      } else {
        value->pa->grad += value->out_grad * value->pb->value;
      }
    } else {
      if (value->requires_grad) {
        value->pa->out_grad += value->grad;
      } else {
        value->pa->out_grad += value->out_grad;
      }
    }
    if (value->pb->requires_grad) {
      if (value->requires_grad) {
        value->pb->grad += value->grad * value->pa->value;
      } else {
        value->pb->grad += value->out_grad * value->pa->value;
      }
    } else {
      if (value->requires_grad) {
        value->pb->out_grad += value->grad;
      } else {
        value->pb->out_grad += value->out_grad;
      }
    }
  }
}

void bwpow(Value *value)
{
  if (value) {
    if (value->pa && value->pa->requires_grad) {
      if (value->requires_grad) {
        value->pa->grad += (value->pb->value * powf(value->pa->value, value->pb->value - 1)) * value->grad;
      } else {
        value->pa->grad += (value->pb->value * powf(value->pa->value, value->pb->value - 1)) * value->out_grad;
      }
    } else if (value->pa) {
      if (value->requires_grad) {
        value->pa->out_grad += value->grad;
      } else {
        value->pa->out_grad += value->out_grad;
      }
    }
  }
}

// TODO: better not to keep recursive call chain
void backward(Value *value, int init_grad)
{
  if (!value) {
    return;
  }
  if (init_grad) {
    value->grad = 1.0f;
    if (!value->requires_grad) {
      value->out_grad = 1.0f;
    }
  }
  if (value->backward) {
    value->backward(value);
  }
  backward(value->pa, 0);
  backward(value->pb, 0);
}

