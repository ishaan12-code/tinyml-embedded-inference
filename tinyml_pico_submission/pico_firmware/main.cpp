#include <cstdio>
#include <cmath>
#include "pico/stdlib.h"
#include "hardware/timer.h"

#include "model_data.h"
#include "mean_std.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

static inline int8_t q(float x, float s, int z){
    int32_t v = (int32_t)lroundf(x/s)+z;
    if(v>127) v=127; if(v<-128) v=-128;
    return (int8_t)v;
}

int main(){
    stdio_init_all();
    sleep_ms(1000);

    const tflite::Model* model = tflite::GetModel(g_model);
    static tflite::AllOpsResolver resolver;

    constexpr int ARENA = 60*1024;
    static uint8_t arena[ARENA] __attribute__((aligned(16)));

    static tflite::MicroInterpreter interp(model, resolver, arena, ARENA);
    interp.AllocateTensors();

    TfLiteTensor* in = interp.input(0);
    TfLiteTensor* out = interp.output(0);

    while(true){
        float feat[kNumFeatures] = {0};

        for(int i=0;i<kNumFeatures;i++){
            float xn=(feat[i]-g_mean[i])/(g_std[i]+1e-8f);
            in->data.int8[i]=q(xn,in->params.scale,in->params.zero_point);
        }

        absolute_time_t t0=get_absolute_time();
        interp.Invoke();
        absolute_time_t t1=get_absolute_time();

        int8_t qo=out->data.int8[0];
        float score=((int)qo-out->params.zero_point)*out->params.scale;
        printf("score=%f latency=%lld us\n",
               score, absolute_time_diff_us(t0,t1));
        sleep_ms(500);
    }
}
