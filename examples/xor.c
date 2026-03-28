#include "cgrad.h"

#include <stdio.h>

typedef struct Sample {
  float x1;
  float x2;
  float y;
} Sample;

static Value *predict(Value *w1, Value *w2, Value *w3, Value *bias, Value *x1,
                      Value *x2) {
  Value *x1_term = fwmul(w1, x1);
  Value *x2_term = fwmul(w2, x2);
  Value *interaction = fwmul(x1, x2);
  Value *interaction_term = fwmul(w3, interaction);
  return fwadd(fwadd(bias, x1_term), fwadd(x2_term, interaction_term));
}

int main(void) {
  const Sample samples[] = {
      {0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 1.0f},
      {1.0f, 0.0f, 1.0f},
      {1.0f, 1.0f, 0.0f},
  };
  const int sample_count = (int)(sizeof(samples) / sizeof(samples[0]));

  float w1 = 0.20f;
  float w2 = -0.30f;
  float w3 = -0.10f;
  float bias = 0.40f;

  const float learning_rate = 0.05f;
  const int epochs = 800;

  for (int epoch = 0; epoch < epochs; epoch++) {
    Value *vw1 = createvalue(w1, NULL, NULL, FN_NONE, 1);
    Value *vw2 = createvalue(w2, NULL, NULL, FN_NONE, 1);
    Value *vw3 = createvalue(w3, NULL, NULL, FN_NONE, 1);
    Value *vbias = createvalue(bias, NULL, NULL, FN_NONE, 1);
    Value *loss = createconst(0.0f);

    for (int i = 0; i < sample_count; i++) {
      Value *x1 = createconst(samples[i].x1);
      Value *x2 = createconst(samples[i].x2);
      Value *target = createconst(samples[i].y);
      Value *pred = predict(vw1, vw2, vw3, vbias, x1, x2);
      Value *error = fwsub(pred, target);
      Value *sample_loss = fwpow(error, createconst(2.0f));
      loss = fwadd(loss, sample_loss);
    }

    loss = fwmul(loss, createconst(1.0f / (float)sample_count));

    backward(loss, 1);

    w1 -= learning_rate * vw1->grad;
    w2 -= learning_rate * vw2->grad;
    w3 -= learning_rate * vw3->grad;
    bias -= learning_rate * vbias->grad;

    if (epoch == 0 || (epoch + 1) % 200 == 0 || epoch == epochs - 1) {
      printf("epoch %3d | loss=%f | w1=% .4f w2=% .4f w3=% .4f b=% .4f\n",
             epoch + 1, loss->value, w1, w2, w3, bias);
    }

    deletechain(loss);
  }

  printf("\ntrained predictions\n");
  for (int i = 0; i < sample_count; i++) {
    float pred = bias + (w1 * samples[i].x1) + (w2 * samples[i].x2) +
                 (w3 * samples[i].x1 * samples[i].x2);
    printf("xor(%0.f, %0.f) -> %0.4f\n", samples[i].x1, samples[i].x2, pred);
  }

  return 0;
}
