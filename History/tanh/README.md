## Results
### Keras CPU
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_4 (Dense)              (None, 512)               401920    
_________________________________________________________________
dense_5 (Dense)              (None, 512)               262656    
_________________________________________________________________
dense_6 (Dense)              (None, 10)                5130      
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
_________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/30
60000/60000 [==============================] - 4s 68us/step - loss: 0.6533 - acc: 0.7870 - val_loss: 0.3445 - val_acc: 0.8945
Epoch 2/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.2477 - acc: 0.9252 - val_loss: 0.2253 - val_acc: 0.9312
Epoch 3/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.1708 - acc: 0.9481 - val_loss: 0.1859 - val_acc: 0.9413
Epoch 4/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.1275 - acc: 0.9607 - val_loss: 0.1671 - val_acc: 0.9460
Epoch 5/30
60000/60000 [==============================] - 4s 69us/step - loss: 0.1002 - acc: 0.9702 - val_loss: 0.1509 - val_acc: 0.9513
Epoch 6/30
60000/60000 [==============================] - 4s 69us/step - loss: 0.0815 - acc: 0.9762 - val_loss: 0.1338 - val_acc: 0.9578
Epoch 7/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0668 - acc: 0.9802 - val_loss: 0.1189 - val_acc: 0.9626
Epoch 8/30
60000/60000 [==============================] - 4s 66us/step - loss: 0.0545 - acc: 0.9837 - val_loss: 0.1063 - val_acc: 0.9673
Epoch 9/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0440 - acc: 0.9871 - val_loss: 0.1012 - val_acc: 0.9696
Epoch 10/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0352 - acc: 0.9902 - val_loss: 0.1036 - val_acc: 0.9695
Epoch 11/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0281 - acc: 0.9926 - val_loss: 0.0986 - val_acc: 0.9723
Epoch 12/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0222 - acc: 0.9946 - val_loss: 0.0927 - val_acc: 0.9735
Epoch 13/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0173 - acc: 0.9965 - val_loss: 0.0937 - val_acc: 0.9738
Epoch 14/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0557 - acc: 0.9891 - val_loss: 0.1028 - val_acc: 0.9700
Epoch 15/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0222 - acc: 0.9930 - val_loss: 0.0893 - val_acc: 0.9754
Epoch 16/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0122 - acc: 0.9974 - val_loss: 0.0870 - val_acc: 0.9754
Epoch 17/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0086 - acc: 0.9987 - val_loss: 0.0876 - val_acc: 0.9759
Epoch 18/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0063 - acc: 0.9993 - val_loss: 0.0859 - val_acc: 0.9763
Epoch 19/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0049 - acc: 0.9996 - val_loss: 0.0854 - val_acc: 0.9774
Epoch 20/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0039 - acc: 0.9997 - val_loss: 0.0852 - val_acc: 0.9782
Epoch 21/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0032 - acc: 0.9999 - val_loss: 0.0854 - val_acc: 0.9786
Epoch 22/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0026 - acc: 0.9999 - val_loss: 0.0859 - val_acc: 0.9791
Epoch 23/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0022 - acc: 1.0000 - val_loss: 0.0864 - val_acc: 0.9788
Epoch 24/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0019 - acc: 1.0000 - val_loss: 0.0869 - val_acc: 0.9784
Epoch 25/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0017 - acc: 1.0000 - val_loss: 0.0874 - val_acc: 0.9786
Epoch 26/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0015 - acc: 1.0000 - val_loss: 0.0880 - val_acc: 0.9788
Epoch 27/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0013 - acc: 1.0000 - val_loss: 0.0885 - val_acc: 0.9788
Epoch 28/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0012 - acc: 1.0000 - val_loss: 0.0890 - val_acc: 0.9787
Epoch 29/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0011 - acc: 1.0000 - val_loss: 0.0894 - val_acc: 0.9789
Epoch 30/30
60000/60000 [==============================] - 4s 65us/step - loss: 9.8050e-04 - acc: 1.0000 - val_loss: 0.0899 - val_acc: 0.9788
```

### Tensorflow GPU
```
loss: 0.6547 / 0.3442    accuracy: 0.7893 / 0.8951    step 1  1.42 sec
loss: 0.2503 / 0.2276    accuracy: 0.9242 / 0.9284    step 2  2.53 sec
loss: 0.1701 / 0.1769    accuracy: 0.9488 / 0.9438    step 3  3.63 sec
loss: 0.1274 / 0.1430    accuracy: 0.9615 / 0.9558    step 4  4.74 sec
loss: 0.0996 / 0.1238    accuracy: 0.9699 / 0.9625    step 5  5.83 sec
loss: 0.0806 / 0.1131    accuracy: 0.9755 / 0.9652    step 6  6.95 sec
loss: 0.0661 / 0.1108    accuracy: 0.9800 / 0.9668    step 7  8.06 sec
loss: 0.0543 / 0.1091    accuracy: 0.9839 / 0.9679    step 8  9.16 sec
loss: 0.0438 / 0.1081    accuracy: 0.9877 / 0.9686    step 9  10.26 sec
loss: 0.0352 / 0.1094    accuracy: 0.9906 / 0.9694    step 10  11.37 sec
loss: 0.0280 / 0.1064    accuracy: 0.9928 / 0.9696    step 11  12.49 sec
loss: 0.0220 / 0.1040    accuracy: 0.9948 / 0.9706    step 12  13.58 sec
loss: 0.0174 / 0.1037    accuracy: 0.9960 / 0.9717    step 13  14.69 sec
loss: 0.0136 / 0.1127    accuracy: 0.9974 / 0.9697    step 14  15.79 sec
loss: 0.0107 / 0.0971    accuracy: 0.9984 / 0.9738    step 15  16.92 sec
loss: 0.0084 / 0.0955    accuracy: 0.9987 / 0.9750    step 16  18.02 sec
loss: 0.0065 / 0.0940    accuracy: 0.9991 / 0.9758    step 17  19.13 sec
loss: 0.0050 / 0.0922    accuracy: 0.9996 / 0.9764    step 18  20.23 sec
loss: 0.0040 / 0.0910    accuracy: 0.9998 / 0.9773    step 19  21.34 sec
loss: 0.0032 / 0.0907    accuracy: 0.9998 / 0.9783    step 20  22.44 sec
loss: 0.0026 / 0.0903    accuracy: 0.9999 / 0.9779    step 21  23.55 sec
loss: 0.0022 / 0.0903    accuracy: 1.0000 / 0.9785    step 22  24.67 sec
loss: 0.0018 / 0.0912    accuracy: 1.0000 / 0.9787    step 23  25.76 sec
loss: 0.0016 / 0.0913    accuracy: 1.0000 / 0.9784    step 24  26.89 sec
loss: 0.0014 / 0.0915    accuracy: 1.0000 / 0.9788    step 25  28.00 sec
loss: 0.0013 / 0.0918    accuracy: 1.0000 / 0.9785    step 26  29.12 sec
loss: 0.0011 / 0.0921    accuracy: 1.0000 / 0.9787    step 27  30.22 sec
loss: 0.0010 / 0.0923    accuracy: 1.0000 / 0.9791    step 28  31.35 sec
loss: 0.0009 / 0.0926    accuracy: 1.0000 / 0.9792    step 29  32.47 sec
loss: 0.0009 / 0.0930    accuracy: 1.0000 / 0.9793    step 30  33.57 sec
```

### Neural_Networks CPU
```
loss: 0.6415 / 0.3664	accuracy: 0.8833 / 0.8849	step 1  86.43 sec
loss: 0.2548 / 0.2345	accuracy: 0.9250 / 0.9280	step 2  173.23 sec
loss: 0.1711 / 0.1945	accuracy: 0.9385 / 0.9380	step 3  259.04 sec
loss: 0.1268 / 0.1867	accuracy: 0.9424 / 0.9396	step 4  345.43 sec
loss: 0.0994 / 0.1723	accuracy: 0.9493 / 0.9450	step 5  431.85 sec
loss: 0.0802 / 0.1351	accuracy: 0.9634 / 0.9570	step 6  518.88 sec
loss: 0.0653 / 0.1083	accuracy: 0.9743 / 0.9670	step 7  605.52 sec
loss: 0.0531 / 0.0963	accuracy: 0.9802 / 0.9714	step 8  692.22 sec
loss: 0.0431 / 0.0920	accuracy: 0.9830 / 0.9725	step 9  779.11 sec
loss: 0.0347 / 0.0903	accuracy: 0.9850 / 0.9729	step 10  866.22 sec
loss: 0.0277 / 0.0891	accuracy: 0.9863 / 0.9735	step 11  953.00 sec
loss: 0.0220 / 0.0857	accuracy: 0.9887 / 0.9751	step 12  1039.82 sec
loss: 0.0174 / 0.0833	accuracy: 0.9908 / 0.9760	step 13  1127.01 sec
loss: 0.0136 / 0.0849	accuracy: 0.9917 / 0.9756	step 14  1213.99 sec
loss: 0.0107 / 0.0850	accuracy: 0.9930 / 0.9762	step 15  1301.14 sec
loss: 0.0086 / 0.0875	accuracy: 0.9938 / 0.9761	step 16  1388.56 sec
loss: 0.1575 / 0.1250	accuracy: 0.9762 / 0.9668	step 17  1475.78 sec
loss: 0.0405 / 0.0855	accuracy: 0.9904 / 0.9748	step 18  1562.65 sec
loss: 0.0140 / 0.0763	accuracy: 0.9954 / 0.9782	step 19  1649.51 sec
loss: 0.0081 / 0.0754	accuracy: 0.9972 / 0.9790	step 20  1736.38 sec
loss: 0.0055 / 0.0747	accuracy: 0.9980 / 0.9796	step 21  1823.11 sec
loss: 0.0041 / 0.0749	accuracy: 0.9989 / 0.9797	step 22  1910.29 sec
loss: 0.0032 / 0.0752	accuracy: 0.9992 / 0.9801	step 23  1997.77 sec
loss: 0.0026 / 0.0754	accuracy: 0.9995 / 0.9803	step 24  2084.75 sec
loss: 0.0022 / 0.0758	accuracy: 0.9997 / 0.9803	step 25  2171.93 sec
loss: 0.0018 / 0.0761	accuracy: 0.9998 / 0.9805	step 26  2259.26 sec
loss: 0.0016 / 0.0763	accuracy: 0.9998 / 0.9805	step 27  2346.68 sec
loss: 0.0014 / 0.0766	accuracy: 0.9999 / 0.9806	step 28  2432.04 sec
loss: 0.0013 / 0.0768	accuracy: 0.9999 / 0.9809	step 29  2517.15 sec
loss: 0.0011 / 0.0771	accuracy: 0.9999 / 0.9809	step 30  2605.09 sec

```
