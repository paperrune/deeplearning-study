## Results
### Keras CPU
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 10)                7850      
=================================================================
Total params: 7,850
Trainable params: 7,850
Non-trainable params: 0
_________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 1s 17us/step - loss: 0.4042 - acc: 0.8863 - val_loss: 0.3263 - val_acc: 0.9054
Epoch 2/20
60000/60000 [==============================] - 1s 12us/step - loss: 0.3104 - acc: 0.9125 - val_loss: 0.3079 - val_acc: 0.9115
Epoch 3/20
60000/60000 [==============================] - 1s 12us/step - loss: 0.2949 - acc: 0.9177 - val_loss: 0.3011 - val_acc: 0.9122
Epoch 4/20
60000/60000 [==============================] - 1s 12us/step - loss: 0.2866 - acc: 0.9201 - val_loss: 0.2975 - val_acc: 0.9125
Epoch 5/20
60000/60000 [==============================] - 1s 12us/step - loss: 0.2810 - acc: 0.9215 - val_loss: 0.2952 - val_acc: 0.9137
Epoch 6/20
60000/60000 [==============================] - 1s 12us/step - loss: 0.2769 - acc: 0.9225 - val_loss: 0.2935 - val_acc: 0.9150
Epoch 7/20
60000/60000 [==============================] - 1s 12us/step - loss: 0.2737 - acc: 0.9237 - val_loss: 0.2922 - val_acc: 0.9159
Epoch 8/20
60000/60000 [==============================] - 1s 11us/step - loss: 0.2711 - acc: 0.9245 - val_loss: 0.2912 - val_acc: 0.9155
Epoch 9/20
60000/60000 [==============================] - 1s 11us/step - loss: 0.2688 - acc: 0.9253 - val_loss: 0.2904 - val_acc: 0.9166
Epoch 10/20
60000/60000 [==============================] - 1s 11us/step - loss: 0.2669 - acc: 0.9259 - val_loss: 0.2895 - val_acc: 0.9169
Epoch 11/20
60000/60000 [==============================] - 1s 11us/step - loss: 0.2653 - acc: 0.9264 - val_loss: 0.2889 - val_acc: 0.9168
Epoch 12/20
60000/60000 [==============================] - 1s 11us/step - loss: 0.2638 - acc: 0.9270 - val_loss: 0.2884 - val_acc: 0.9168
Epoch 13/20
60000/60000 [==============================] - 1s 11us/step - loss: 0.2625 - acc: 0.9275 - val_loss: 0.2880 - val_acc: 0.9169
Epoch 14/20
60000/60000 [==============================] - 1s 13us/step - loss: 0.2613 - acc: 0.9279 - val_loss: 0.2876 - val_acc: 0.9174
Epoch 15/20
60000/60000 [==============================] - 1s 13us/step - loss: 0.2602 - acc: 0.9282 - val_loss: 0.2873 - val_acc: 0.9176
Epoch 16/20
60000/60000 [==============================] - 1s 12us/step - loss: 0.2592 - acc: 0.9285 - val_loss: 0.2870 - val_acc: 0.9178
Epoch 17/20
60000/60000 [==============================] - 1s 11us/step - loss: 0.2583 - acc: 0.9288 - val_loss: 0.2867 - val_acc: 0.9181
Epoch 18/20
60000/60000 [==============================] - 1s 12us/step - loss: 0.2574 - acc: 0.9290 - val_loss: 0.2865 - val_acc: 0.9182
Epoch 19/20
60000/60000 [==============================] - 1s 11us/step - loss: 0.2566 - acc: 0.9293 - val_loss: 0.2864 - val_acc: 0.9185
Epoch 20/20
60000/60000 [==============================] - 1s 11us/step - loss: 0.2559 - acc: 0.9296 - val_loss: 0.2862 - val_acc: 0.9187
```

### Tensorflow GPU
```
loss: 0.4042 / 0.3264    accuracy: 0.8862 / 0.9050    step 1  1.22 sec
loss: 0.3104 / 0.3080    accuracy: 0.9127 / 0.9111    step 2  2.20 sec
loss: 0.2950 / 0.3012    accuracy: 0.9177 / 0.9121    step 3  3.13 sec
loss: 0.2867 / 0.2976    accuracy: 0.9202 / 0.9124    step 4  4.07 sec
loss: 0.2811 / 0.2952    accuracy: 0.9215 / 0.9137    step 5  4.89 sec
loss: 0.2771 / 0.2935    accuracy: 0.9226 / 0.9150    step 6  5.70 sec
loss: 0.2739 / 0.2922    accuracy: 0.9237 / 0.9154    step 7  6.51 sec
loss: 0.2713 / 0.2912    accuracy: 0.9244 / 0.9154    step 8  7.34 sec
loss: 0.2691 / 0.2903    accuracy: 0.9252 / 0.9159    step 9  8.18 sec
loss: 0.2672 / 0.2896    accuracy: 0.9257 / 0.9167    step 10  9.03 sec
loss: 0.2655 / 0.2890    accuracy: 0.9262 / 0.9165    step 11  9.93 sec
loss: 0.2641 / 0.2885    accuracy: 0.9267 / 0.9166    step 12  10.85 sec
loss: 0.2628 / 0.2881    accuracy: 0.9271 / 0.9164    step 13  11.76 sec
loss: 0.2616 / 0.2877    accuracy: 0.9276 / 0.9168    step 14  12.64 sec
loss: 0.2605 / 0.2874    accuracy: 0.9280 / 0.9172    step 15  13.57 sec
loss: 0.2595 / 0.2871    accuracy: 0.9283 / 0.9177    step 16  14.44 sec
loss: 0.2586 / 0.2869    accuracy: 0.9286 / 0.9179    step 17  15.39 sec
loss: 0.2578 / 0.2867    accuracy: 0.9290 / 0.9179    step 18  16.26 sec
loss: 0.2570 / 0.2866    accuracy: 0.9291 / 0.9179    step 19  17.15 sec
loss: 0.2563 / 0.2864    accuracy: 0.9293 / 0.9186    step 20  18.07 sec
```
