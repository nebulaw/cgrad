#include "cgrad.h"

#include <stdio.h>
#include <math.h>


Value createvalue(float value, Value *pa, Value *pb, int fn_op, int requires_grad)
{
  Value v;
  v.value = value;
  v.grad = 0.0f;
  v.out_grad = 0.0f;
  v.fn_op = fn_op;
  v.requires_grad = requires_grad;
  v.pa = pa;
  v.pb = pb;
  switch (fn_op) {
  case FN_ADD:
    v.backward = bwadd;
    v.fn_type = FN_BINARY;
    break;
  case FN_MUL:
    v.backward = bwmul;
    v.fn_type = FN_BINARY;
    break;
  case FN_POW:
    v.backward = bwpow;
    v.fn_type = FN_UNARY;
    break;
  default:
    v.backward = bwnone;
    v.fn_type = FN_NONE;
    break;
  }
  return v;
}

Value createnone(void)
{
  Value v;
  v.value = 0.0f;
  v.grad = 0.0f;
  v.out_grad = 0.0f;
  v.requires_grad = 0;
  v.fn_op = FN_NONE;
  v.fn_type = FN_NONE;
  v.pa = NULL;
  v.pb = NULL;
  v.backward = NULL;
  return v;
}

Value copy(Value *value)
{
  Value v;
  v.value = value->value;
  v.grad = value->grad;
  v.out_grad = value->grad;
  v.requires_grad = value->requires_grad;
  v.pa = value->pa;
  v.pb = value->pb;
  v.fn_type = value->fn_type;
  v.fn_op = value->fn_op;
  v.backward = value->backward;
  return v;
}

Value fwadd(Value *a, Value *b)
{
  return a && b ? createvalue(a->value + b->value, a, b, FN_ADD, 0) : createnone();
}

Value fwmul(Value *a, Value *b)
{
  return a && b ? createvalue(a->value * b->value, a, b, FN_MUL, 0) : createnone();
}

Value fwpow(Value *a, Value *b)
{
  return a && b ? createvalue(powf(a->value, b->value), a, b, FN_POW, 0) : createnone();
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

