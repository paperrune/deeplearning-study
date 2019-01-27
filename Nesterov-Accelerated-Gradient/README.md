## Results
### Keras CPU
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 512)               401920    
_________________________________________________________________
dense_2 (Dense)              (None, 512)               262656    
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5130      
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
_________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/30
60000/60000 [==============================] - 5s 78us/step - loss: 0.5125 - acc: 0.8291 - val_loss: 0.1712 - val_acc: 0.9459
Epoch 2/30
60000/60000 [==============================] - 4s 72us/step - loss: 0.1117 - acc: 0.9664 - val_loss: 0.1190 - val_acc: 0.9659
Epoch 3/30
60000/60000 [==============================] - 4s 71us/step - loss: 0.0711 - acc: 0.9781 - val_loss: 0.1304 - val_acc: 0.9631
Epoch 4/30
60000/60000 [==============================] - 4s 71us/step - loss: 0.0484 - acc: 0.9847 - val_loss: 0.1226 - val_acc: 0.9665
Epoch 5/30
60000/60000 [==============================] - 4s 71us/step - loss: 0.0377 - acc: 0.9882 - val_loss: 0.1026 - val_acc: 0.9724
Epoch 6/30
60000/60000 [==============================] - 4s 71us/step - loss: 0.0301 - acc: 0.9902 - val_loss: 0.1122 - val_acc: 0.9726
Epoch 7/30
60000/60000 [==============================] - 4s 70us/step - loss: 0.0242 - acc: 0.9920 - val_loss: 0.1281 - val_acc: 0.9691
Epoch 8/30
60000/60000 [==============================] - 4s 72us/step - loss: 0.0212 - acc: 0.9932 - val_loss: 0.0915 - val_acc: 0.9782
Epoch 9/30
60000/60000 [==============================] - 4s 71us/step - loss: 0.0166 - acc: 0.9946 - val_loss: 0.0978 - val_acc: 0.9784
Epoch 10/30
60000/60000 [==============================] - 4s 70us/step - loss: 0.0160 - acc: 0.9945 - val_loss: 0.1024 - val_acc: 0.9770
Epoch 11/30
60000/60000 [==============================] - 4s 70us/step - loss: 0.0132 - acc: 0.9954 - val_loss: 0.0873 - val_acc: 0.9803
Epoch 12/30
60000/60000 [==============================] - 4s 71us/step - loss: 0.0111 - acc: 0.9962 - val_loss: 0.0909 - val_acc: 0.9799
Epoch 13/30
60000/60000 [==============================] - 4s 71us/step - loss: 0.0084 - acc: 0.9973 - val_loss: 0.0952 - val_acc: 0.9803
Epoch 14/30
60000/60000 [==============================] - 4s 72us/step - loss: 0.0122 - acc: 0.9958 - val_loss: 0.1213 - val_acc: 0.9768
Epoch 15/30
60000/60000 [==============================] - 4s 70us/step - loss: 0.0103 - acc: 0.9965 - val_loss: 0.1006 - val_acc: 0.9802
Epoch 16/30
60000/60000 [==============================] - 4s 70us/step - loss: 0.0067 - acc: 0.9976 - val_loss: 0.0978 - val_acc: 0.9816
Epoch 17/30
60000/60000 [==============================] - 4s 70us/step - loss: 0.0055 - acc: 0.9981 - val_loss: 0.1057 - val_acc: 0.9804
Epoch 18/30
60000/60000 [==============================] - 4s 70us/step - loss: 0.0061 - acc: 0.9982 - val_loss: 0.0975 - val_acc: 0.9813
Epoch 19/30
60000/60000 [==============================] - 4s 70us/step - loss: 0.0060 - acc: 0.9982 - val_loss: 0.0959 - val_acc: 0.9841
Epoch 20/30
60000/60000 [==============================] - 4s 71us/step - loss: 0.0053 - acc: 0.9983 - val_loss: 0.1007 - val_acc: 0.9823
Epoch 21/30
60000/60000 [==============================] - 4s 71us/step - loss: 0.0067 - acc: 0.9976 - val_loss: 0.0939 - val_acc: 0.9833
Epoch 22/30
60000/60000 [==============================] - 4s 70us/step - loss: 0.0054 - acc: 0.9983 - val_loss: 0.0946 - val_acc: 0.9825
Epoch 23/30
60000/60000 [==============================] - 4s 70us/step - loss: 0.0035 - acc: 0.9989 - val_loss: 0.0980 - val_acc: 0.9830
Epoch 24/30
60000/60000 [==============================] - 4s 71us/step - loss: 0.0064 - acc: 0.9982 - val_loss: 0.0998 - val_acc: 0.9826
Epoch 25/30
60000/60000 [==============================] - 4s 70us/step - loss: 0.0031 - acc: 0.9991 - val_loss: 0.0950 - val_acc: 0.9840
Epoch 26/30
60000/60000 [==============================] - 4s 70us/step - loss: 0.0015 - acc: 0.9996 - val_loss: 0.0909 - val_acc: 0.9858
Epoch 27/30
60000/60000 [==============================] - 4s 71us/step - loss: 0.0011 - acc: 0.9997 - val_loss: 0.0898 - val_acc: 0.9865
Epoch 28/30
60000/60000 [==============================] - 4s 72us/step - loss: 7.5989e-04 - acc: 0.9999 - val_loss: 0.0908 - val_acc: 0.9862
Epoch 29/30
60000/60000 [==============================] - 4s 72us/step - loss: 4.8823e-04 - acc: 0.9999 - val_loss: 0.0900 - val_acc: 0.9866
Epoch 30/30
60000/60000 [==============================] - 4s 71us/step - loss: 3.2892e-04 - acc: 1.0000 - val_loss: 0.0897 - val_acc: 0.9870
```

### Tensorflow GPU
```
loss: 0.5087 / 0.1736	accuracy: 0.8296 / 0.9451	step 1  1.46 sec
loss: 0.1128 / 0.1076	accuracy: 0.9661 / 0.9674	step 2  2.58 sec
loss: 0.0724 / 0.1218	accuracy: 0.9782 / 0.9644	step 3  3.68 sec
loss: 0.0507 / 0.1093	accuracy: 0.9844 / 0.9687	step 4  4.80 sec
loss: 0.0374 / 0.1501	accuracy: 0.9883 / 0.9615	step 5  5.93 sec
loss: 0.0309 / 0.1471	accuracy: 0.9905 / 0.9634	step 6  7.04 sec
loss: 0.0229 / 0.0939	accuracy: 0.9925 / 0.9763	step 7  8.13 sec
loss: 0.0192 / 0.1146	accuracy: 0.9937 / 0.9729	step 8  9.24 sec
loss: 0.0144 / 0.1201	accuracy: 0.9950 / 0.9734	step 9  10.40 sec
loss: 0.0155 / 0.1023	accuracy: 0.9950 / 0.9773	step 10  11.57 sec
loss: 0.0112 / 0.1097	accuracy: 0.9963 / 0.9771	step 11  12.70 sec
loss: 0.0122 / 0.1021	accuracy: 0.9958 / 0.9802	step 12  13.83 sec
loss: 0.0100 / 0.0941	accuracy: 0.9965 / 0.9823	step 13  14.95 sec
loss: 0.0102 / 0.1268	accuracy: 0.9965 / 0.9768	step 14  16.09 sec
loss: 0.0070 / 0.0980	accuracy: 0.9975 / 0.9814	step 15  17.22 sec
loss: 0.0068 / 0.1095	accuracy: 0.9978 / 0.9789	step 16  18.34 sec
loss: 0.0084 / 0.1222	accuracy: 0.9972 / 0.9767	step 17  19.48 sec
loss: 0.0099 / 0.0973	accuracy: 0.9967 / 0.9809	step 18  20.64 sec
loss: 0.0077 / 0.1138	accuracy: 0.9974 / 0.9804	step 19  21.86 sec
loss: 0.0078 / 0.0982	accuracy: 0.9975 / 0.9822	step 20  23.04 sec
loss: 0.0035 / 0.0903	accuracy: 0.9988 / 0.9845	step 21  24.24 sec
loss: 0.0016 / 0.0922	accuracy: 0.9996 / 0.9843	step 22  25.37 sec
loss: 0.0013 / 0.0944	accuracy: 0.9996 / 0.9851	step 23  26.51 sec
loss: 0.0010 / 0.0889	accuracy: 0.9997 / 0.9855	step 24  27.64 sec
loss: 0.0003 / 0.0885	accuracy: 1.0000 / 0.9859	step 25  28.80 sec
loss: 0.0001 / 0.0891	accuracy: 1.0000 / 0.9860	step 26  30.01 sec
loss: 0.0001 / 0.0912	accuracy: 1.0000 / 0.9860	step 27  31.25 sec
loss: 0.0000 / 0.0922	accuracy: 1.0000 / 0.9860	step 28  32.40 sec
loss: 0.0000 / 0.0929	accuracy: 1.0000 / 0.9860	step 29  33.53 sec
loss: 0.0000 / 0.0935	accuracy: 1.0000 / 0.9860	step 30  34.70 sec
```

### Neural_Networks CPU
```
loss: 0.4964 / 0.1862	accuracy: 0.8325 / 0.9383	step 1  15.36 sec
loss: 0.1164 / 0.1132	accuracy: 0.9644 / 0.9643	step 2  30.68 sec
loss: 0.0758 / 0.1139	accuracy: 0.9767 / 0.9647	step 3  45.97 sec
loss: 0.0518 / 0.0899	accuracy: 0.9842 / 0.9738	step 4  61.29 sec
loss: 0.0366 / 0.0909	accuracy: 0.9887 / 0.9749	step 5  76.60 sec
loss: 0.0301 / 0.1010	accuracy: 0.9900 / 0.9733	step 6  91.92 sec
loss: 0.0251 / 0.0975	accuracy: 0.9918 / 0.9759	step 7  107.20 sec
loss: 0.0196 / 0.1017	accuracy: 0.9938 / 0.9744	step 8  122.51 sec
loss: 0.0187 / 0.0849	accuracy: 0.9936 / 0.9783	step 9  137.86 sec
loss: 0.0134 / 0.0808	accuracy: 0.9955 / 0.9808	step 10  153.19 sec
loss: 0.0091 / 0.0823	accuracy: 0.9969 / 0.9818	step 11  168.52 sec
loss: 0.0108 / 0.0803	accuracy: 0.9961 / 0.9818	step 12  183.85 sec
loss: 0.0080 / 0.0963	accuracy: 0.9972 / 0.9812	step 13  199.17 sec
loss: 0.0084 / 0.0941	accuracy: 0.9969 / 0.9796	step 14  214.50 sec
loss: 0.0131 / 0.0921	accuracy: 0.9961 / 0.9810	step 15  229.91 sec
loss: 0.0058 / 0.0803	accuracy: 0.9981 / 0.9831	step 16  245.23 sec
loss: 0.0047 / 0.0917	accuracy: 0.9988 / 0.9809	step 17  260.55 sec
loss: 0.0067 / 0.0934	accuracy: 0.9981 / 0.9800	step 18  275.93 sec
loss: 0.0032 / 0.0958	accuracy: 0.9990 / 0.9815	step 19  291.25 sec
loss: 0.0022 / 0.0880	accuracy: 0.9994 / 0.9825	step 20  306.60 sec
loss: 0.0014 / 0.0835	accuracy: 0.9996 / 0.9843	step 21  321.96 sec
loss: 0.0004 / 0.0790	accuracy: 0.9999 / 0.9859	step 22  337.30 sec
loss: 0.0001 / 0.0806	accuracy: 1.0000 / 0.9851	step 23  352.62 sec
loss: 0.0001 / 0.0813	accuracy: 1.0000 / 0.9851	step 24  367.88 sec
loss: 0.0001 / 0.0818	accuracy: 1.0000 / 0.9852	step 25  383.19 sec
loss: 0.0001 / 0.0823	accuracy: 1.0000 / 0.9852	step 26  398.47 sec
loss: 0.0000 / 0.0827	accuracy: 1.0000 / 0.9855	step 27  413.78 sec
loss: 0.0000 / 0.0831	accuracy: 1.0000 / 0.9855	step 28  429.07 sec
loss: 0.0000 / 0.0835	accuracy: 1.0000 / 0.9857	step 29  444.41 sec
loss: 0.0000 / 0.0839	accuracy: 1.0000 / 0.9856	step 30  459.74 sec
```
