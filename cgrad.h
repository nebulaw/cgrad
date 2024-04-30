#ifndef SMOL_AUTOGRAD
#define SMOL_AUTOGRAD

#define FN_NONE     0
#define FN_ADD      2
#define FN_MUL      8
#define FN_POW      32

#define FN_UNARY    2
#define FN_BINARY   4


typedef struct Value {
  float value;
  float grad;
  float out_grad;
  int requires_grad;
  int fn_op;
  int fn_type;
  struct Value *pa;
  struct Value *pb;
  void (*backward)(struct Value *self);
} Value;


Value createvalue(float value, Value *pa, Value *pb, int fn_op, int requires_grad);
Value createnone(void);
Value copy(Value *);

Value fwadd(Value *a, Value *b);
Value fwmul(Value *a, Value *b);
Value fwpow(Value *a, Value *b);
Value fwsub(Value *a, Value *b);
Value fwdiv(Value *a, Value *b);

void bwnone(Value *);
void bwadd(Value *);
void bwmul(Value *);
void bwpow(Value *);

void backward(Value *value, int init_grad);

#endif // SMOL_AUTOGRAD
