## Results
### Keras
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_2 (Dense)              (None, 10)                7850      
=================================================================
Total params: 7,850
Trainable params: 7,850
Non-trainable params: 0
_________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 48s 806us/step - loss: 0.0242 - acc: 0.8810 - val_loss: 0.0186 - val_acc: 0.9055
Epoch 2/20
60000/60000 [==============================] - 49s 811us/step - loss: 0.0187 - acc: 0.9042 - val_loss: 0.0170 - val_acc: 0.9142
Epoch 3/20
60000/60000 [==============================] - 48s 796us/step - loss: 0.0176 - acc: 0.9091 - val_loss: 0.0164 - val_acc: 0.9158
Epoch 4/20
60000/60000 [==============================] - 52s 868us/step - loss: 0.0170 - acc: 0.9120 - val_loss: 0.0160 - val_acc: 0.9155
Epoch 5/20
60000/60000 [==============================] - 53s 876us/step - loss: 0.0166 - acc: 0.9138 - val_loss: 0.0158 - val_acc: 0.9167
Epoch 6/20
60000/60000 [==============================] - 48s 801us/step - loss: 0.0163 - acc: 0.9152 - val_loss: 0.0155 - val_acc: 0.9187
Epoch 7/20
60000/60000 [==============================] - 50s 837us/step - loss: 0.0161 - acc: 0.9157 - val_loss: 0.0154 - val_acc: 0.9194
Epoch 8/20
60000/60000 [==============================] - 51s 848us/step - loss: 0.0159 - acc: 0.9169 - val_loss: 0.0152 - val_acc: 0.9190
Epoch 9/20
60000/60000 [==============================] - 49s 817us/step - loss: 0.0157 - acc: 0.9183 - val_loss: 0.0151 - val_acc: 0.9207
Epoch 10/20
60000/60000 [==============================] - 46s 773us/step - loss: 0.0156 - acc: 0.9185 - val_loss: 0.0152 - val_acc: 0.9197
Epoch 11/20
60000/60000 [==============================] - 48s 798us/step - loss: 0.0155 - acc: 0.9192 - val_loss: 0.0153 - val_acc: 0.9184
Epoch 12/20
60000/60000 [==============================] - 47s 776us/step - loss: 0.0154 - acc: 0.9195 - val_loss: 0.0149 - val_acc: 0.9208
Epoch 13/20
60000/60000 [==============================] - 47s 778us/step - loss: 0.0153 - acc: 0.9197 - val_loss: 0.0152 - val_acc: 0.9184
Epoch 14/20
60000/60000 [==============================] - 46s 767us/step - loss: 0.0152 - acc: 0.9202 - val_loss: 0.0148 - val_acc: 0.9210
Epoch 15/20
60000/60000 [==============================] - 44s 733us/step - loss: 0.0151 - acc: 0.9208 - val_loss: 0.0148 - val_acc: 0.9217
Epoch 16/20
60000/60000 [==============================] - 44s 730us/step - loss: 0.0151 - acc: 0.9212 - val_loss: 0.0150 - val_acc: 0.9188
Epoch 17/20
60000/60000 [==============================] - 44s 727us/step - loss: 0.0150 - acc: 0.9213 - val_loss: 0.0149 - val_acc: 0.9218
Epoch 18/20
60000/60000 [==============================] - 44s 729us/step - loss: 0.0150 - acc: 0.9213 - val_loss: 0.0148 - val_acc: 0.9216
Epoch 19/20
60000/60000 [==============================] - 44s 727us/step - loss: 0.0149 - acc: 0.9216 - val_loss: 0.0148 - val_acc: 0.9216
Epoch 20/20
60000/60000 [==============================] - 48s 798us/step - loss: 0.0149 - acc: 0.9220 - val_loss: 0.0147 - val_acc: 0.9223
```

### Tensorflow
```
loss: 0.0240 / 0.0191    accuracy: 0.8815 / 0.9044    step 1  67.53 sec
loss: 0.0186 / 0.0174    accuracy: 0.9046 / 0.9102    step 2  133.71 sec
loss: 0.0175 / 0.0167    accuracy: 0.9101 / 0.9119    step 3  199.73 sec
loss: 0.0169 / 0.0163    accuracy: 0.9130 / 0.9138    step 4  265.84 sec
loss: 0.0165 / 0.0160    accuracy: 0.9147 / 0.9148    step 5  331.85 sec
loss: 0.0162 / 0.0158    accuracy: 0.9159 / 0.9155    step 6  398.17 sec
loss: 0.0160 / 0.0156    accuracy: 0.9167 / 0.9154    step 7  464.31 sec
loss: 0.0158 / 0.0155    accuracy: 0.9177 / 0.9160    step 8  530.52 sec
loss: 0.0156 / 0.0154    accuracy: 0.9184 / 0.9164    step 9  596.65 sec
loss: 0.0155 / 0.0153    accuracy: 0.9192 / 0.9164    step 10  662.66 sec
loss: 0.0154 / 0.0153    accuracy: 0.9198 / 0.9167    step 11  728.88 sec
loss: 0.0153 / 0.0152    accuracy: 0.9203 / 0.9171    step 12  794.91 sec
loss: 0.0152 / 0.0152    accuracy: 0.9207 / 0.9174    step 13  861.08 sec
loss: 0.0151 / 0.0151    accuracy: 0.9209 / 0.9180    step 14  926.95 sec
loss: 0.0151 / 0.0151    accuracy: 0.9212 / 0.9185    step 15  993.14 sec
loss: 0.0150 / 0.0151    accuracy: 0.9214 / 0.9188    step 16  1059.21 sec
loss: 0.0149 / 0.0150    accuracy: 0.9217 / 0.9190    step 17  1125.21 sec
loss: 0.0149 / 0.0150    accuracy: 0.9219 / 0.9195    step 18  1191.51 sec
loss: 0.0148 / 0.0150    accuracy: 0.9220 / 0.9197    step 19  1257.38 sec
loss: 0.0148 / 0.0150    accuracy: 0.9223 / 0.9197    step 20  1322.87 sec
```

### Neural_Networks.cpp
```
loss: 0.0240 / 0.0191	accuracy: 0.8946 / 0.9044	step 1  7.50 sec
loss: 0.0186 / 0.0174	accuracy: 0.9037 / 0.9099	step 2  15.54 sec
loss: 0.0175 / 0.0167	accuracy: 0.9079 / 0.9121	step 3  23.39 sec
loss: 0.0169 / 0.0163	accuracy: 0.9107 / 0.9137	step 4  31.29 sec
loss: 0.0165 / 0.0160	accuracy: 0.9124 / 0.9149	step 5  39.66 sec
loss: 0.0162 / 0.0158	accuracy: 0.9136 / 0.9154	step 6  47.38 sec
loss: 0.0160 / 0.0156	accuracy: 0.9148 / 0.9157	step 7  54.97 sec
loss: 0.0158 / 0.0155	accuracy: 0.9157 / 0.9160	step 8  62.40 sec
loss: 0.0156 / 0.0154	accuracy: 0.9166 / 0.9162	step 9  69.86 sec
loss: 0.0155 / 0.0153	accuracy: 0.9175 / 0.9163	step 10  77.15 sec
loss: 0.0154 / 0.0153	accuracy: 0.9179 / 0.9167	step 11  84.70 sec
loss: 0.0153 / 0.0152	accuracy: 0.9183 / 0.9169	step 12  92.23 sec
loss: 0.0152 / 0.0152	accuracy: 0.9188 / 0.9175	step 13  99.66 sec
loss: 0.0151 / 0.0151	accuracy: 0.9193 / 0.9179	step 14  107.03 sec
loss: 0.0151 / 0.0151	accuracy: 0.9198 / 0.9185	step 15  115.72 sec
loss: 0.0150 / 0.0151	accuracy: 0.9202 / 0.9189	step 16  123.55 sec
loss: 0.0149 / 0.0150	accuracy: 0.9205 / 0.9191	step 17  131.42 sec
loss: 0.0149 / 0.0150	accuracy: 0.9208 / 0.9196	step 18  139.47 sec
loss: 0.0148 / 0.0150	accuracy: 0.9210 / 0.9198	step 19  147.34 sec
loss: 0.0148 / 0.0150	accuracy: 0.9214 / 0.9198	step 20  154.98 sec
```
