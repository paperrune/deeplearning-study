## Results
### Keras CPU
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional_1 (Bidirection (None, 256)               160768    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                2570      
=================================================================
Total params: 163,338
Trainable params: 163,338
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:Variable *= will be deprecated. Use variable.assign_mul if you want assignment to the variable value or 'x = x * y' if you want a new python Tensor object.
Train on 60000 samples, validate on 10000 samples
Epoch 1/30
60000/60000 [==============================] - 35s 588us/step - loss: 0.4976 - acc: 0.8368 - val_loss: 0.1888 - val_acc: 0.9403
Epoch 2/30
60000/60000 [==============================] - 36s 598us/step - loss: 0.1125 - acc: 0.9658 - val_loss: 0.1010 - val_acc: 0.9701
Epoch 3/30
60000/60000 [==============================] - 36s 603us/step - loss: 0.0754 - acc: 0.9770 - val_loss: 0.0649 - val_acc: 0.9783
Epoch 4/30
60000/60000 [==============================] - 36s 598us/step - loss: 0.0571 - acc: 0.9820 - val_loss: 0.0573 - val_acc: 0.9811
Epoch 5/30
60000/60000 [==============================] - 38s 628us/step - loss: 0.0441 - acc: 0.9865 - val_loss: 0.0483 - val_acc: 0.9845
Epoch 6/30
60000/60000 [==============================] - 39s 658us/step - loss: 0.0399 - acc: 0.9871 - val_loss: 0.0606 - val_acc: 0.9821
Epoch 7/30
60000/60000 [==============================] - 38s 631us/step - loss: 0.0338 - acc: 0.9893 - val_loss: 0.0466 - val_acc: 0.9854
Epoch 8/30
60000/60000 [==============================] - 39s 642us/step - loss: 0.0278 - acc: 0.9911 - val_loss: 0.0445 - val_acc: 0.9858
Epoch 9/30
60000/60000 [==============================] - 37s 612us/step - loss: 0.0237 - acc: 0.9923 - val_loss: 0.0427 - val_acc: 0.9870
Epoch 10/30
60000/60000 [==============================] - 38s 626us/step - loss: 0.0203 - acc: 0.9934 - val_loss: 0.0475 - val_acc: 0.9853
Epoch 11/30
60000/60000 [==============================] - 36s 603us/step - loss: 0.0178 - acc: 0.9940 - val_loss: 0.0449 - val_acc: 0.9855
Epoch 12/30
60000/60000 [==============================] - 36s 600us/step - loss: 0.0151 - acc: 0.9950 - val_loss: 0.0398 - val_acc: 0.9890
Epoch 13/30
60000/60000 [==============================] - 36s 596us/step - loss: 0.0130 - acc: 0.9957 - val_loss: 0.0442 - val_acc: 0.9875
Epoch 14/30
60000/60000 [==============================] - 36s 593us/step - loss: 0.0122 - acc: 0.9962 - val_loss: 0.0479 - val_acc: 0.9863
Epoch 15/30
60000/60000 [==============================] - 35s 592us/step - loss: 0.0109 - acc: 0.9965 - val_loss: 0.0429 - val_acc: 0.9883
Epoch 16/30
60000/60000 [==============================] - 36s 603us/step - loss: 0.0068 - acc: 0.9980 - val_loss: 0.0395 - val_acc: 0.9887
Epoch 17/30
60000/60000 [==============================] - 37s 610us/step - loss: 0.0083 - acc: 0.9973 - val_loss: 0.0385 - val_acc: 0.9898
Epoch 18/30
60000/60000 [==============================] - 37s 610us/step - loss: 0.0072 - acc: 0.9980 - val_loss: 0.0448 - val_acc: 0.9883
Epoch 19/30
60000/60000 [==============================] - 36s 602us/step - loss: 0.0072 - acc: 0.9979 - val_loss: 0.0459 - val_acc: 0.9878
Epoch 20/30
60000/60000 [==============================] - 36s 608us/step - loss: 0.0054 - acc: 0.9984 - val_loss: 0.0455 - val_acc: 0.9884
Epoch 21/30
60000/60000 [==============================] - 36s 603us/step - loss: 0.0091 - acc: 0.9970 - val_loss: 0.0415 - val_acc: 0.9893
Epoch 22/30
60000/60000 [==============================] - 36s 607us/step - loss: 0.0072 - acc: 0.9980 - val_loss: 0.0420 - val_acc: 0.9896
Epoch 23/30
60000/60000 [==============================] - 36s 604us/step - loss: 0.0047 - acc: 0.9986 - val_loss: 0.0420 - val_acc: 0.9898
Epoch 24/30
60000/60000 [==============================] - 36s 599us/step - loss: 0.0020 - acc: 0.9997 - val_loss: 0.0412 - val_acc: 0.9899
Epoch 25/30
60000/60000 [==============================] - 36s 597us/step - loss: 0.0019 - acc: 0.9996 - val_loss: 0.0424 - val_acc: 0.9895
Epoch 26/30
60000/60000 [==============================] - 36s 601us/step - loss: 0.0014 - acc: 0.9998 - val_loss: 0.0444 - val_acc: 0.9898
Epoch 27/30
60000/60000 [==============================] - 36s 599us/step - loss: 0.0012 - acc: 0.9998 - val_loss: 0.0405 - val_acc: 0.9910
Epoch 28/30
60000/60000 [==============================] - 36s 594us/step - loss: 6.9200e-04 - acc: 0.9999 - val_loss: 0.0403 - val_acc: 0.9910
Epoch 29/30
60000/60000 [==============================] - 36s 599us/step - loss: 6.9272e-04 - acc: 0.9999 - val_loss: 0.0417 - val_acc: 0.9911
Epoch 30/30
60000/60000 [==============================] - 36s 601us/step - loss: 5.0653e-04 - acc: 1.0000 - val_loss: 0.0415 - val_acc: 0.9907
```

### Tensorflow GPU
```
loss: 0.4027 / 0.1053	accuracy: 0.8662 / 0.9649	step 1  34.75 sec
loss: 0.0938 / 0.0766	accuracy: 0.9710 / 0.9750	step 2  68.80 sec
loss: 0.0654 / 0.0634	accuracy: 0.9792 / 0.9810	step 3  102.87 sec
loss: 0.0498 / 0.0534	accuracy: 0.9838 / 0.9829	step 4  136.95 sec
loss: 0.0440 / 0.0579	accuracy: 0.9862 / 0.9817	step 5  171.26 sec
loss: 0.0371 / 0.0461	accuracy: 0.9884 / 0.9866	step 6  205.50 sec
loss: 0.0293 / 0.0415	accuracy: 0.9908 / 0.9866	step 7  239.74 sec
loss: 0.0256 / 0.0500	accuracy: 0.9918 / 0.9854	step 8  274.05 sec
loss: 0.0229 / 0.0391	accuracy: 0.9925 / 0.9888	step 9  308.15 sec
loss: 0.0183 / 0.0443	accuracy: 0.9942 / 0.9882	step 10  342.52 sec
loss: 0.0135 / 0.0409	accuracy: 0.9959 / 0.9877	step 11  376.18 sec
loss: 0.0118 / 0.0532	accuracy: 0.9963 / 0.9851	step 12  410.57 sec
loss: 0.0118 / 0.0386	accuracy: 0.9960 / 0.9896	step 13  445.84 sec
loss: 0.0115 / 0.0393	accuracy: 0.9965 / 0.9889	step 14  483.20 sec
loss: 0.0113 / 0.0391	accuracy: 0.9965 / 0.9904	step 15  521.36 sec
loss: 0.0080 / 0.0388	accuracy: 0.9976 / 0.9902	step 16  556.32 sec
loss: 0.0053 / 0.0434	accuracy: 0.9987 / 0.9901	step 17  590.42 sec
loss: 0.0041 / 0.0462	accuracy: 0.9990 / 0.9893	step 18  626.31 sec
loss: 0.0070 / 0.0431	accuracy: 0.9980 / 0.9895	step 19  666.96 sec
loss: 0.0049 / 0.0446	accuracy: 0.9985 / 0.9901	step 20  700.67 sec
loss: 0.0073 / 0.0468	accuracy: 0.9977 / 0.9890	step 21  735.05 sec
loss: 0.0044 / 0.0422	accuracy: 0.9987 / 0.9895	step 22  769.50 sec
loss: 0.0030 / 0.0390	accuracy: 0.9993 / 0.9909	step 23  803.99 sec
loss: 0.0020 / 0.0464	accuracy: 0.9996 / 0.9896	step 24  838.49 sec
loss: 0.0012 / 0.0427	accuracy: 0.9998 / 0.9892	step 25  872.70 sec
loss: 0.0015 / 0.0454	accuracy: 0.9997 / 0.9896	step 26  917.39 sec
loss: 0.0007 / 0.0439	accuracy: 0.9999 / 0.9902	step 27  959.82 sec
loss: 0.0004 / 0.0425	accuracy: 1.0000 / 0.9902	step 28  1002.04 sec
loss: 0.0006 / 0.0433	accuracy: 0.9999 / 0.9908	step 29  1043.83 sec
loss: 0.0003 / 0.0441	accuracy: 1.0000 / 0.9905	step 30  1086.34 sec
```

### Neural_Networks GPU
```
loss: 0.3487 / 0.1152	accuracy: 0.9659 / 0.9618	step 1  107.81 sec
loss: 0.0903 / 0.0687	accuracy: 0.9808 / 0.9769	step 2  216.81 sec
loss: 0.0625 / 0.0741	accuracy: 0.9820 / 0.9771	step 3  326.02 sec
loss: 0.0478 / 0.0511	accuracy: 0.9883 / 0.9846	step 4  435.20 sec
loss: 0.0397 / 0.0498	accuracy: 0.9908 / 0.9848	step 5  544.38 sec
loss: 0.0339 / 0.0524	accuracy: 0.9913 / 0.9835	step 6  653.57 sec
loss: 0.0273 / 0.0390	accuracy: 0.9935 / 0.9877	step 7  762.77 sec
loss: 0.0213 / 0.0429	accuracy: 0.9951 / 0.9867	step 8  871.95 sec
loss: 0.0186 / 0.0399	accuracy: 0.9953 / 0.9874	step 9  981.13 sec
loss: 0.0173 / 0.0422	accuracy: 0.9962 / 0.9871	step 10  1090.33 sec
loss: 0.0154 / 0.0433	accuracy: 0.9948 / 0.9864	step 11  1199.54 sec
loss: 0.0124 / 0.0404	accuracy: 0.9973 / 0.9881	step 12  1308.73 sec
loss: 0.0095 / 0.0372	accuracy: 0.9976 / 0.9890	step 13  1417.90 sec
loss: 0.0086 / 0.0448	accuracy: 0.9981 / 0.9875	step 14  1527.08 sec
loss: 0.0074 / 0.0389	accuracy: 0.9981 / 0.9896	step 15  1636.25 sec
loss: 0.0059 / 0.0381	accuracy: 0.9991 / 0.9895	step 16  1745.48 sec
loss: 0.0047 / 0.0392	accuracy: 0.9989 / 0.9890	step 17  1854.84 sec
loss: 0.0051 / 0.0405	accuracy: 0.9979 / 0.9892	step 18  1964.23 sec
loss: 0.0034 / 0.0426	accuracy: 0.9988 / 0.9891	step 19  2073.57 sec
loss: 0.0024 / 0.0361	accuracy: 0.9993 / 0.9896	step 20  2182.92 sec
loss: 0.0021 / 0.0379	accuracy: 0.9999 / 0.9892	step 21  2292.13 sec
loss: 0.0008 / 0.0375	accuracy: 0.9999 / 0.9894	step 22  2401.34 sec
loss: 0.0007 / 0.0372	accuracy: 1.0000 / 0.9902	step 23  2510.52 sec
loss: 0.0005 / 0.0372	accuracy: 1.0000 / 0.9899	step 24  2619.70 sec
loss: 0.0004 / 0.0373	accuracy: 1.0000 / 0.9903	step 25  2728.89 sec
loss: 0.0003 / 0.0371	accuracy: 1.0000 / 0.9899	step 26  2838.09 sec
loss: 0.0003 / 0.0372	accuracy: 1.0000 / 0.9904	step 27  2947.28 sec
loss: 0.0003 / 0.0378	accuracy: 1.0000 / 0.9905	step 28  3056.50 sec
loss: 0.0003 / 0.0378	accuracy: 1.0000 / 0.9905	step 29  3165.70 sec
loss: 0.0002 / 0.0382	accuracy: 1.0000 / 0.9905	step 30  3274.96 sec
```
