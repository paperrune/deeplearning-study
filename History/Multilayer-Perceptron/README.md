## Results
### Keras CPU
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_50 (Dense)             (None, 512)               401920    
_________________________________________________________________
dense_51 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_52 (Dense)             (None, 10)                5130      
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
_________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/30
60000/60000 [==============================] - 5s 87us/step - loss: 1.9496 - acc: 0.2606 - val_loss: 1.6834 - val_acc: 0.3212
Epoch 2/30
60000/60000 [==============================] - 4s 74us/step - loss: 1.5501 - acc: 0.3977 - val_loss: 1.2489 - val_acc: 0.5185
Epoch 3/30
60000/60000 [==============================] - 4s 73us/step - loss: 0.8494 - acc: 0.7082 - val_loss: 0.6360 - val_acc: 0.7964
Epoch 4/30
60000/60000 [==============================] - 4s 73us/step - loss: 0.4273 - acc: 0.8711 - val_loss: 0.4476 - val_acc: 0.8599
Epoch 5/30
60000/60000 [==============================] - 4s 73us/step - loss: 0.3210 - acc: 0.9051 - val_loss: 0.3581 - val_acc: 0.8893
Epoch 6/30
60000/60000 [==============================] - 4s 73us/step - loss: 0.2706 - acc: 0.9193 - val_loss: 0.2830 - val_acc: 0.9143
Epoch 7/30
60000/60000 [==============================] - 4s 73us/step - loss: 0.2324 - acc: 0.9308 - val_loss: 0.2337 - val_acc: 0.9296
Epoch 8/30
60000/60000 [==============================] - 4s 74us/step - loss: 0.2022 - acc: 0.9390 - val_loss: 0.2029 - val_acc: 0.9401
Epoch 9/30
60000/60000 [==============================] - 4s 74us/step - loss: 0.1777 - acc: 0.9467 - val_loss: 0.1803 - val_acc: 0.9464
Epoch 10/30
60000/60000 [==============================] - 4s 74us/step - loss: 0.1575 - acc: 0.9526 - val_loss: 0.1638 - val_acc: 0.9517
Epoch 11/30
60000/60000 [==============================] - 4s 74us/step - loss: 0.1409 - acc: 0.9571 - val_loss: 0.1516 - val_acc: 0.9541
Epoch 12/30
60000/60000 [==============================] - 5s 78us/step - loss: 0.1270 - acc: 0.9619 - val_loss: 0.1410 - val_acc: 0.9567
Epoch 13/30
60000/60000 [==============================] - 4s 75us/step - loss: 0.1150 - acc: 0.9654 - val_loss: 0.1318 - val_acc: 0.9596
Epoch 14/30
60000/60000 [==============================] - 5s 76us/step - loss: 0.1044 - acc: 0.9688 - val_loss: 0.1237 - val_acc: 0.9623
Epoch 15/30
60000/60000 [==============================] - 4s 73us/step - loss: 0.0951 - acc: 0.9717 - val_loss: 0.1161 - val_acc: 0.9644
Epoch 16/30
60000/60000 [==============================] - 4s 73us/step - loss: 0.0868 - acc: 0.9745 - val_loss: 0.1092 - val_acc: 0.9669
Epoch 17/30
60000/60000 [==============================] - 4s 72us/step - loss: 0.0793 - acc: 0.9772 - val_loss: 0.1030 - val_acc: 0.9686
Epoch 18/30
60000/60000 [==============================] - 4s 72us/step - loss: 0.0725 - acc: 0.9789 - val_loss: 0.0977 - val_acc: 0.9702
Epoch 19/30
60000/60000 [==============================] - 4s 72us/step - loss: 0.0664 - acc: 0.9810 - val_loss: 0.0931 - val_acc: 0.9715
Epoch 20/30
60000/60000 [==============================] - 4s 72us/step - loss: 0.0608 - acc: 0.9827 - val_loss: 0.0891 - val_acc: 0.9727
Epoch 21/30
60000/60000 [==============================] - 4s 72us/step - loss: 0.0557 - acc: 0.9844 - val_loss: 0.0860 - val_acc: 0.9734
Epoch 22/30
60000/60000 [==============================] - 4s 73us/step - loss: 0.0510 - acc: 0.9859 - val_loss: 0.0836 - val_acc: 0.9744
Epoch 23/30
60000/60000 [==============================] - 4s 73us/step - loss: 0.0467 - acc: 0.9868 - val_loss: 0.0819 - val_acc: 0.9750
Epoch 24/30
60000/60000 [==============================] - 4s 73us/step - loss: 0.0427 - acc: 0.9882 - val_loss: 0.0805 - val_acc: 0.9761
Epoch 25/30
60000/60000 [==============================] - 4s 73us/step - loss: 0.0391 - acc: 0.9895 - val_loss: 0.0794 - val_acc: 0.9763
Epoch 26/30
60000/60000 [==============================] - 4s 74us/step - loss: 0.0358 - acc: 0.9905 - val_loss: 0.0785 - val_acc: 0.9763
Epoch 27/30
60000/60000 [==============================] - 4s 73us/step - loss: 0.0327 - acc: 0.9918 - val_loss: 0.0778 - val_acc: 0.9764
Epoch 28/30
60000/60000 [==============================] - 4s 74us/step - loss: 0.0299 - acc: 0.9928 - val_loss: 0.0771 - val_acc: 0.9768
Epoch 29/30
60000/60000 [==============================] - 4s 73us/step - loss: 0.0272 - acc: 0.9937 - val_loss: 0.0767 - val_acc: 0.9773
Epoch 30/30
60000/60000 [==============================] - 4s 74us/step - loss: 0.0248 - acc: 0.9943 - val_loss: 0.0762 - val_acc: 0.9776
```

### Tensorflow GPU
```
loss: 1.8986 / 1.6792    accuracy: 0.2745 / 0.3353    step 1  1.39 sec
loss: 1.4814 / 1.2983    accuracy: 0.4318 / 0.5176    step 2  2.45 sec
loss: 0.7566 / 0.5802    accuracy: 0.7462 / 0.8142    step 3  3.57 sec
loss: 0.4093 / 0.4692    accuracy: 0.8773 / 0.8560    step 4  4.65 sec
loss: 0.3149 / 0.3906    accuracy: 0.9065 / 0.8843    step 5  5.77 sec
loss: 0.2684 / 0.3342    accuracy: 0.9201 / 0.8994    step 6  6.86 sec
loss: 0.2338 / 0.2891    accuracy: 0.9303 / 0.9133    step 7  7.96 sec
loss: 0.2049 / 0.2401    accuracy: 0.9382 / 0.9278    step 8  9.10 sec
loss: 0.1804 / 0.2073    accuracy: 0.9457 / 0.9369    step 9  10.17 sec
loss: 0.1597 / 0.1850    accuracy: 0.9518 / 0.9442    step 10  11.28 sec
loss: 0.1421 / 0.1686    accuracy: 0.9576 / 0.9491    step 11  12.40 sec
loss: 0.1272 / 0.1565    accuracy: 0.9617 / 0.9529    step 12  13.51 sec
loss: 0.1144 / 0.1444    accuracy: 0.9656 / 0.9566    step 13  14.60 sec
loss: 0.1036 / 0.1283    accuracy: 0.9692 / 0.9612    step 14  15.73 sec
loss: 0.0940 / 0.1155    accuracy: 0.9725 / 0.9662    step 15  16.85 sec
loss: 0.0855 / 0.1070    accuracy: 0.9750 / 0.9684    step 16  17.94 sec
loss: 0.0778 / 0.1013    accuracy: 0.9775 / 0.9697    step 17  19.04 sec
loss: 0.0710 / 0.0973    accuracy: 0.9795 / 0.9709    step 18  20.16 sec
loss: 0.0650 / 0.0942    accuracy: 0.9811 / 0.9713    step 19  21.26 sec
loss: 0.0595 / 0.0917    accuracy: 0.9830 / 0.9718    step 20  22.34 sec
loss: 0.0543 / 0.0893    accuracy: 0.9847 / 0.9722    step 21  23.47 sec
loss: 0.0496 / 0.0871    accuracy: 0.9860 / 0.9730    step 22  24.58 sec
loss: 0.0452 / 0.0851    accuracy: 0.9876 / 0.9737    step 23  25.67 sec
loss: 0.0412 / 0.0834    accuracy: 0.9886 / 0.9744    step 24  26.81 sec
loss: 0.0376 / 0.0818    accuracy: 0.9896 / 0.9745    step 25  27.89 sec
loss: 0.0343 / 0.0804    accuracy: 0.9909 / 0.9753    step 26  29.01 sec
loss: 0.0312 / 0.0792    accuracy: 0.9921 / 0.9754    step 27  30.10 sec
loss: 0.0283 / 0.0785    accuracy: 0.9931 / 0.9763    step 28  31.23 sec
loss: 0.0258 / 0.0780    accuracy: 0.9940 / 0.9764    step 29  32.36 sec
loss: 0.0234 / 0.0775    accuracy: 0.9949 / 0.9768    step 30  33.45 sec
```

### Neural_Networks CPU
```
loss: 1.8883 / 1.6803	accuracy: 0.3328 / 0.3364	step 1  85.80 sec
loss: 1.4749 / 1.0843	accuracy: 0.5963 / 0.6043	step 2  171.81 sec
loss: 0.7497 / 0.5980	accuracy: 0.7979 / 0.8085	step 3  258.02 sec
loss: 0.4089 / 0.5257	accuracy: 0.8348 / 0.8379	step 4  344.72 sec
loss: 0.3173 / 0.3906	accuracy: 0.8761 / 0.8819	step 5  431.34 sec
loss: 0.2675 / 0.3146	accuracy: 0.9005 / 0.9052	step 6  517.83 sec
loss: 0.2311 / 0.2567	accuracy: 0.9197 / 0.9205	step 7  604.55 sec
loss: 0.2017 / 0.2211	accuracy: 0.9313 / 0.9324	step 8  691.22 sec
loss: 0.1773 / 0.2000	accuracy: 0.9384 / 0.9398	step 9  778.10 sec
loss: 0.1571 / 0.1849	accuracy: 0.9434 / 0.9443	step 10  865.02 sec
loss: 0.1404 / 0.1731	accuracy: 0.9472 / 0.9466	step 11  951.90 sec
loss: 0.1262 / 0.1588	accuracy: 0.9527 / 0.9499	step 12  1037.63 sec
loss: 0.1139 / 0.1425	accuracy: 0.9593 / 0.9554	step 13  1128.77 sec
loss: 0.1032 / 0.1290	accuracy: 0.9638 / 0.9593	step 14  1215.62 sec
loss: 0.0939 / 0.1190	accuracy: 0.9683 / 0.9620	step 15  1302.26 sec
loss: 0.0858 / 0.1116	accuracy: 0.9714 / 0.9643	step 16  1389.50 sec
loss: 0.0785 / 0.1059	accuracy: 0.9739 / 0.9656	step 17  1476.34 sec
loss: 0.0720 / 0.1012	accuracy: 0.9761 / 0.9676	step 18  1563.21 sec
loss: 0.0660 / 0.0972	accuracy: 0.9780 / 0.9685	step 19  1650.30 sec
loss: 0.0604 / 0.0936	accuracy: 0.9799 / 0.9699	step 20  1736.66 sec
loss: 0.0554 / 0.0905	accuracy: 0.9819 / 0.9705	step 21  1823.54 sec
loss: 0.0507 / 0.0878	accuracy: 0.9832 / 0.9724	step 22  1910.51 sec
loss: 0.0465 / 0.0855	accuracy: 0.9848 / 0.9727	step 23  1997.12 sec
loss: 0.0425 / 0.0834	accuracy: 0.9861 / 0.9738	step 24  2084.43 sec
loss: 0.0389 / 0.0816	accuracy: 0.9875 / 0.9747	step 25  2171.50 sec
loss: 0.0355 / 0.0800	accuracy: 0.9886 / 0.9752	step 26  2258.50 sec
loss: 0.0325 / 0.0787	accuracy: 0.9893 / 0.9760	step 27  2345.57 sec
loss: 0.0297 / 0.0778	accuracy: 0.9899 / 0.9761	step 28  2432.21 sec
loss: 0.0271 / 0.0773	accuracy: 0.9905 / 0.9766	step 29  2519.02 sec
loss: 0.0248 / 0.0771	accuracy: 0.9912 / 0.9774	step 30  2606.17 sec
```