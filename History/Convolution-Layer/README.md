## Results
### Keras CPU
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 12, 12, 24)        624       
_________________________________________________________________
flatten_1 (Flatten)          (None, 3456)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               1769984   
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130      
=================================================================
Total params: 1,775,738
Trainable params: 1,775,738
Non-trainable params: 0
_________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/30
60000/60000 [==============================] - 15s 246us/step - loss: 0.2727 - acc: 0.9162 - val_loss: 0.0856 - val_acc: 0.9711
Epoch 2/30
60000/60000 [==============================] - 15s 246us/step - loss: 0.0605 - acc: 0.9818 - val_loss: 0.0624 - val_acc: 0.9807
Epoch 3/30
60000/60000 [==============================] - 15s 245us/step - loss: 0.0317 - acc: 0.9904 - val_loss: 0.0601 - val_acc: 0.9823
Epoch 4/30
60000/60000 [==============================] - 15s 242us/step - loss: 0.0184 - acc: 0.9946 - val_loss: 0.0568 - val_acc: 0.9839
Epoch 5/30
60000/60000 [==============================] - 14s 240us/step - loss: 0.0126 - acc: 0.9960 - val_loss: 0.0538 - val_acc: 0.9851
Epoch 6/30
60000/60000 [==============================] - 14s 241us/step - loss: 0.0096 - acc: 0.9968 - val_loss: 0.0553 - val_acc: 0.9862
Epoch 7/30
60000/60000 [==============================] - 14s 241us/step - loss: 0.0075 - acc: 0.9975 - val_loss: 0.0594 - val_acc: 0.9858
Epoch 8/30
60000/60000 [==============================] - 14s 241us/step - loss: 0.0042 - acc: 0.9989 - val_loss: 0.0598 - val_acc: 0.9871
Epoch 9/30
60000/60000 [==============================] - 14s 241us/step - loss: 0.0037 - acc: 0.9989 - val_loss: 0.0551 - val_acc: 0.9875
Epoch 10/30
60000/60000 [==============================] - 14s 240us/step - loss: 0.0032 - acc: 0.9991 - val_loss: 0.0517 - val_acc: 0.9871
Epoch 11/30
60000/60000 [==============================] - 14s 240us/step - loss: 0.0020 - acc: 0.9995 - val_loss: 0.0577 - val_acc: 0.9875
Epoch 12/30
60000/60000 [==============================] - 14s 240us/step - loss: 9.3753e-04 - acc: 0.9999 - val_loss: 0.0534 - val_acc: 0.9885
Epoch 13/30
60000/60000 [==============================] - 15s 244us/step - loss: 5.7628e-04 - acc: 0.9999 - val_loss: 0.0518 - val_acc: 0.9888
Epoch 14/30
60000/60000 [==============================] - 15s 242us/step - loss: 4.2011e-04 - acc: 1.0000 - val_loss: 0.0518 - val_acc: 0.9885
Epoch 15/30
60000/60000 [==============================] - 14s 241us/step - loss: 4.1042e-04 - acc: 0.9999 - val_loss: 0.0525 - val_acc: 0.9889
Epoch 16/30
60000/60000 [==============================] - 14s 239us/step - loss: 3.4204e-04 - acc: 1.0000 - val_loss: 0.0531 - val_acc: 0.9888
Epoch 17/30
60000/60000 [==============================] - 14s 241us/step - loss: 3.2257e-04 - acc: 1.0000 - val_loss: 0.0534 - val_acc: 0.9888
Epoch 18/30
60000/60000 [==============================] - 15s 248us/step - loss: 3.1579e-04 - acc: 1.0000 - val_loss: 0.0538 - val_acc: 0.9888
Epoch 19/30
60000/60000 [==============================] - 15s 244us/step - loss: 3.1073e-04 - acc: 1.0000 - val_loss: 0.0541 - val_acc: 0.9888
Epoch 20/30
60000/60000 [==============================] - 15s 254us/step - loss: 3.0685e-04 - acc: 1.0000 - val_loss: 0.0544 - val_acc: 0.9887
Epoch 21/30
60000/60000 [==============================] - 14s 240us/step - loss: 3.0370e-04 - acc: 1.0000 - val_loss: 0.0547 - val_acc: 0.9887
Epoch 22/30
60000/60000 [==============================] - 14s 241us/step - loss: 3.0113e-04 - acc: 1.0000 - val_loss: 0.0549 - val_acc: 0.9888
Epoch 23/30
60000/60000 [==============================] - 15s 242us/step - loss: 2.9893e-04 - acc: 1.0000 - val_loss: 0.0552 - val_acc: 0.9890
Epoch 24/30
60000/60000 [==============================] - 15s 245us/step - loss: 2.9702e-04 - acc: 1.0000 - val_loss: 0.0554 - val_acc: 0.9890
Epoch 25/30
60000/60000 [==============================] - 15s 246us/step - loss: 2.9534e-04 - acc: 1.0000 - val_loss: 0.0556 - val_acc: 0.9889
Epoch 26/30
60000/60000 [==============================] - 15s 253us/step - loss: 2.9387e-04 - acc: 1.0000 - val_loss: 0.0558 - val_acc: 0.9889
Epoch 27/30
60000/60000 [==============================] - 16s 261us/step - loss: 2.9256e-04 - acc: 1.0000 - val_loss: 0.0560 - val_acc: 0.9889
Epoch 28/30
60000/60000 [==============================] - 15s 256us/step - loss: 2.9138e-04 - acc: 1.0000 - val_loss: 0.0562 - val_acc: 0.9889
Epoch 29/30
60000/60000 [==============================] - 15s 245us/step - loss: 2.9034e-04 - acc: 1.0000 - val_loss: 0.0563 - val_acc: 0.9889
Epoch 30/30
60000/60000 [==============================] - 14s 241us/step - loss: 2.8938e-04 - acc: 1.0000 - val_loss: 0.0565 - val_acc: 0.9888
```

### Tensorflow GPU
```
loss: 0.2794 / 0.0868    accuracy: 0.9127 / 0.9708    step 1  3.35 sec
loss: 0.0618 / 0.0712    accuracy: 0.9816 / 0.9770    step 2  5.09 sec
loss: 0.0346 / 0.0611    accuracy: 0.9896 / 0.9799    step 3  6.81 sec
loss: 0.0198 / 0.0564    accuracy: 0.9942 / 0.9830    step 4  8.49 sec
loss: 0.0150 / 0.0567    accuracy: 0.9952 / 0.9852    step 5  10.24 sec
loss: 0.0099 / 0.0624    accuracy: 0.9969 / 0.9837    step 6  12.05 sec
loss: 0.0086 / 0.0635    accuracy: 0.9974 / 0.9836    step 7  13.87 sec
loss: 0.0052 / 0.0554    accuracy: 0.9984 / 0.9855    step 8  15.55 sec
loss: 0.0031 / 0.0533    accuracy: 0.9991 / 0.9864    step 9  17.26 sec
loss: 0.0034 / 0.0573    accuracy: 0.9991 / 0.9869    step 10  19.00 sec
loss: 0.0021 / 0.0556    accuracy: 0.9994 / 0.9876    step 11  20.82 sec
loss: 0.0013 / 0.0549    accuracy: 0.9997 / 0.9884    step 12  22.65 sec
loss: 0.0004 / 0.0507    accuracy: 1.0000 / 0.9893    step 13  24.47 sec
loss: 0.0001 / 0.0504    accuracy: 1.0000 / 0.9892    step 14  26.29 sec
loss: 0.0001 / 0.0512    accuracy: 1.0000 / 0.9894    step 15  28.17 sec
loss: 0.0001 / 0.0518    accuracy: 1.0000 / 0.9894    step 16  30.03 sec
loss: 0.0001 / 0.0524    accuracy: 1.0000 / 0.9894    step 17  31.78 sec
loss: 0.0000 / 0.0528    accuracy: 1.0000 / 0.9894    step 18  33.62 sec
loss: 0.0000 / 0.0533    accuracy: 1.0000 / 0.9893    step 19  35.49 sec
loss: 0.0000 / 0.0537    accuracy: 1.0000 / 0.9892    step 20  37.35 sec
loss: 0.0000 / 0.0540    accuracy: 1.0000 / 0.9892    step 21  39.26 sec
loss: 0.0000 / 0.0544    accuracy: 1.0000 / 0.9891    step 22  41.26 sec
loss: 0.0000 / 0.0547    accuracy: 1.0000 / 0.9891    step 23  43.21 sec
loss: 0.0000 / 0.0550    accuracy: 1.0000 / 0.9890    step 24  45.12 sec
loss: 0.0000 / 0.0553    accuracy: 1.0000 / 0.9890    step 25  46.99 sec
loss: 0.0000 / 0.0555    accuracy: 1.0000 / 0.9890    step 26  48.88 sec
loss: 0.0000 / 0.0558    accuracy: 1.0000 / 0.9890    step 27  50.74 sec
loss: 0.0000 / 0.0560    accuracy: 1.0000 / 0.9890    step 28  52.57 sec
loss: 0.0000 / 0.0562    accuracy: 1.0000 / 0.9890    step 29  54.37 sec
loss: 0.0000 / 0.0564    accuracy: 1.0000 / 0.9890    step 30  56.18 sec
```

### Neural_Networks GPU
```
loss: 0.2843 / 0.1030	accuracy: 0.9676 / 0.9667	step 1  40.17 sec
loss: 0.0604 / 0.0583	accuracy: 0.9854 / 0.9806	step 2  80.71 sec
loss: 0.0331 / 0.0503	accuracy: 0.9911 / 0.9845	step 3  121.52 sec
loss: 0.0198 / 0.0647	accuracy: 0.9905 / 0.9813	step 4  162.46 sec
loss: 0.0120 / 0.0498	accuracy: 0.9962 / 0.9861	step 5  203.49 sec
loss: 0.0093 / 0.0495	accuracy: 0.9972 / 0.9864	step 6  244.56 sec
loss: 0.0053 / 0.0560	accuracy: 0.9970 / 0.9850	step 7  285.60 sec
loss: 0.0029 / 0.0529	accuracy: 0.9984 / 0.9864	step 8  326.65 sec
loss: 0.0027 / 0.0583	accuracy: 0.9983 / 0.9865	step 9  367.68 sec
loss: 0.0030 / 0.0551	accuracy: 0.9988 / 0.9868	step 10  408.72 sec
loss: 0.0021 / 0.0534	accuracy: 0.9994 / 0.9873	step 11  449.75 sec
loss: 0.0008 / 0.0515	accuracy: 0.9999 / 0.9883	step 12  490.80 sec
loss: 0.0003 / 0.0500	accuracy: 1.0000 / 0.9887	step 13  531.85 sec
loss: 0.0001 / 0.0509	accuracy: 1.0000 / 0.9888	step 14  572.93 sec
loss: 0.0001 / 0.0515	accuracy: 1.0000 / 0.9888	step 15  614.01 sec
loss: 0.0001 / 0.0520	accuracy: 1.0000 / 0.9890	step 16  655.09 sec
loss: 0.0001 / 0.0525	accuracy: 1.0000 / 0.9891	step 17  696.15 sec
loss: 0.0001 / 0.0528	accuracy: 1.0000 / 0.9890	step 18  737.23 sec
loss: 0.0001 / 0.0531	accuracy: 1.0000 / 0.9890	step 19  778.31 sec
loss: 0.0000 / 0.0534	accuracy: 1.0000 / 0.9890	step 20  819.35 sec
loss: 0.0000 / 0.0537	accuracy: 1.0000 / 0.9891	step 21  860.39 sec
loss: 0.0000 / 0.0539	accuracy: 1.0000 / 0.9892	step 22  901.45 sec
loss: 0.0000 / 0.0541	accuracy: 1.0000 / 0.9891	step 23  942.50 sec
loss: 0.0000 / 0.0543	accuracy: 1.0000 / 0.9892	step 24  983.56 sec
loss: 0.0000 / 0.0545	accuracy: 1.0000 / 0.9893	step 25  1024.63 sec
loss: 0.0000 / 0.0547	accuracy: 1.0000 / 0.9892	step 26  1065.69 sec
loss: 0.0000 / 0.0549	accuracy: 1.0000 / 0.9892	step 27  1106.77 sec
loss: 0.0000 / 0.0551	accuracy: 1.0000 / 0.9892	step 28  1147.82 sec
loss: 0.0000 / 0.0552	accuracy: 1.0000 / 0.9892	step 29  1188.88 sec
loss: 0.0000 / 0.0554	accuracy: 1.0000 / 0.9892	step 30  1229.95 sec

```