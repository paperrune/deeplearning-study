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
60000/60000 [==============================] - 1s 10us/step - loss: 0.0541 - acc: 0.7679 - val_loss: 0.0370 - val_acc: 0.8547
Epoch 2/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0338 - acc: 0.8555 - val_loss: 0.0301 - val_acc: 0.8702
Epoch 3/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0294 - acc: 0.8690 - val_loss: 0.0272 - val_acc: 0.8813
Epoch 4/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0271 - acc: 0.8758 - val_loss: 0.0254 - val_acc: 0.8871
Epoch 5/20
60000/60000 [==============================] - 1s 10us/step - loss: 0.0257 - acc: 0.8805 - val_loss: 0.0242 - val_acc: 0.8923
Epoch 6/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0247 - acc: 0.8832 - val_loss: 0.0233 - val_acc: 0.8937
Epoch 7/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0239 - acc: 0.8866 - val_loss: 0.0226 - val_acc: 0.8955
Epoch 8/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0233 - acc: 0.8888 - val_loss: 0.0220 - val_acc: 0.8979
Epoch 9/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0227 - acc: 0.8906 - val_loss: 0.0216 - val_acc: 0.8998
Epoch 10/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0223 - acc: 0.8921 - val_loss: 0.0212 - val_acc: 0.9014
Epoch 11/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0219 - acc: 0.8932 - val_loss: 0.0208 - val_acc: 0.9020
Epoch 12/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0216 - acc: 0.8942 - val_loss: 0.0205 - val_acc: 0.9032
Epoch 13/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0213 - acc: 0.8950 - val_loss: 0.0202 - val_acc: 0.9040
Epoch 14/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0210 - acc: 0.8962 - val_loss: 0.0200 - val_acc: 0.9050
Epoch 15/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0208 - acc: 0.8966 - val_loss: 0.0198 - val_acc: 0.9055
Epoch 16/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0206 - acc: 0.8974 - val_loss: 0.0196 - val_acc: 0.9068
Epoch 17/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0204 - acc: 0.8984 - val_loss: 0.0194 - val_acc: 0.9073
Epoch 18/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0202 - acc: 0.8991 - val_loss: 0.0192 - val_acc: 0.9077
Epoch 19/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0201 - acc: 0.9000 - val_loss: 0.0191 - val_acc: 0.9087
Epoch 20/20
60000/60000 [==============================] - 1s 9us/step - loss: 0.0199 - acc: 0.9005 - val_loss: 0.0189 - val_acc: 0.9088
```

### Tensorflow GPU
```
loss: 0.0541 / 0.0370    accuracy: 0.7663 / 0.8492    step 1  1.00 sec
loss: 0.0338 / 0.0301    accuracy: 0.8560 / 0.8718    step 2  1.66 sec
loss: 0.0294 / 0.0271    accuracy: 0.8682 / 0.8818    step 3  2.37 sec
loss: 0.0271 / 0.0254    accuracy: 0.8753 / 0.8878    step 4  3.06 sec
loss: 0.0257 / 0.0242    accuracy: 0.8805 / 0.8914    step 5  3.76 sec
loss: 0.0247 / 0.0233    accuracy: 0.8841 / 0.8936    step 6  4.46 sec
loss: 0.0239 / 0.0226    accuracy: 0.8865 / 0.8952    step 7  5.19 sec
loss: 0.0233 / 0.0220    accuracy: 0.8884 / 0.8967    step 8  5.88 sec
loss: 0.0227 / 0.0216    accuracy: 0.8901 / 0.8986    step 9  6.58 sec
loss: 0.0223 / 0.0212    accuracy: 0.8915 / 0.8999    step 10  7.28 sec
loss: 0.0219 / 0.0208    accuracy: 0.8926 / 0.9012    step 11  7.98 sec
loss: 0.0216 / 0.0205    accuracy: 0.8936 / 0.9021    step 12  8.68 sec
loss: 0.0213 / 0.0202    accuracy: 0.8947 / 0.9038    step 13  9.40 sec
loss: 0.0210 / 0.0200    accuracy: 0.8956 / 0.9043    step 14  10.13 sec
loss: 0.0208 / 0.0198    accuracy: 0.8964 / 0.9050    step 15  10.82 sec
loss: 0.0206 / 0.0196    accuracy: 0.8976 / 0.9057    step 16  11.54 sec
loss: 0.0204 / 0.0194    accuracy: 0.8985 / 0.9066    step 17  12.22 sec
loss: 0.0202 / 0.0192    accuracy: 0.8990 / 0.9079    step 18  12.93 sec
loss: 0.0201 / 0.0191    accuracy: 0.8998 / 0.9086    step 19  13.62 sec
loss: 0.0199 / 0.0189    accuracy: 0.9005 / 0.9089    step 20  14.36 sec
```

### Neural_Networks CPU
```
loss: 0.0541 / 0.0370	accuracy: 0.7651 / 0.8499	step 1  1.09 sec
loss: 0.0338 / 0.0301	accuracy: 0.8555 / 0.8719	step 2  2.17 sec
loss: 0.0294 / 0.0271	accuracy: 0.8685 / 0.8820	step 3  3.26 sec
loss: 0.0271 / 0.0254	accuracy: 0.8755 / 0.8876	step 4  4.34 sec
loss: 0.0257 / 0.0242	accuracy: 0.8803 / 0.8913	step 5  5.43 sec
loss: 0.0247 / 0.0233	accuracy: 0.8839 / 0.8938	step 6  6.52 sec
loss: 0.0239 / 0.0226	accuracy: 0.8864 / 0.8950	step 7  7.61 sec
loss: 0.0233 / 0.0221	accuracy: 0.8884 / 0.8967	step 8  8.70 sec
loss: 0.0227 / 0.0216	accuracy: 0.8900 / 0.8985	step 9  9.80 sec
loss: 0.0223 / 0.0212	accuracy: 0.8914 / 0.8996	step 10  10.89 sec
loss: 0.0219 / 0.0208	accuracy: 0.8927 / 0.9014	step 11  11.97 sec
loss: 0.0216 / 0.0205	accuracy: 0.8939 / 0.9022	step 12  13.05 sec
loss: 0.0213 / 0.0203	accuracy: 0.8948 / 0.9039	step 13  14.13 sec
loss: 0.0210 / 0.0200	accuracy: 0.8958 / 0.9045	step 14  15.22 sec
loss: 0.0208 / 0.0198	accuracy: 0.8966 / 0.9056	step 15  16.30 sec
loss: 0.0206 / 0.0196	accuracy: 0.8974 / 0.9063	step 16  17.39 sec
loss: 0.0204 / 0.0194	accuracy: 0.8984 / 0.9070	step 17  18.47 sec
loss: 0.0202 / 0.0192	accuracy: 0.8992 / 0.9076	step 18  19.56 sec
loss: 0.0201 / 0.0191	accuracy: 0.9000 / 0.9085	step 19  20.65 sec
loss: 0.0199 / 0.0189	accuracy: 0.9005 / 0.9087	step 20  21.75 sec
```
