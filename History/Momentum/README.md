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
60000/60000 [==============================] - 4s 68us/step - loss: 0.5778 - acc: 0.8074 - val_loss: 0.1548 - val_acc: 0.9508
Epoch 2/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.1298 - acc: 0.9608 - val_loss: 0.1213 - val_acc: 0.9629
Epoch 3/30
60000/60000 [==============================] - 4s 71us/step - loss: 0.0825 - acc: 0.9745 - val_loss: 0.1678 - val_acc: 0.9529
Epoch 4/30
60000/60000 [==============================] - 4s 70us/step - loss: 0.0570 - acc: 0.9816 - val_loss: 0.1279 - val_acc: 0.9642
Epoch 5/30
60000/60000 [==============================] - 4s 67us/step - loss: 0.0427 - acc: 0.9866 - val_loss: 0.0955 - val_acc: 0.9742
Epoch 6/30
60000/60000 [==============================] - 4s 72us/step - loss: 0.0312 - acc: 0.9900 - val_loss: 0.0907 - val_acc: 0.9762
Epoch 7/30
60000/60000 [==============================] - 5s 80us/step - loss: 0.0280 - acc: 0.9907 - val_loss: 0.0896 - val_acc: 0.9772
Epoch 8/30
60000/60000 [==============================] - 5s 77us/step - loss: 0.0223 - acc: 0.9928 - val_loss: 0.0839 - val_acc: 0.9787
Epoch 9/30
60000/60000 [==============================] - 5s 76us/step - loss: 0.0157 - acc: 0.9950 - val_loss: 0.0844 - val_acc: 0.9804
Epoch 10/30
60000/60000 [==============================] - 5s 78us/step - loss: 0.0139 - acc: 0.9951 - val_loss: 0.0941 - val_acc: 0.9799
Epoch 11/30
60000/60000 [==============================] - 4s 69us/step - loss: 0.0129 - acc: 0.9958 - val_loss: 0.0979 - val_acc: 0.9790
Epoch 12/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0134 - acc: 0.9957 - val_loss: 0.0996 - val_acc: 0.9793
Epoch 13/30
60000/60000 [==============================] - 4s 66us/step - loss: 0.0103 - acc: 0.9965 - val_loss: 0.0959 - val_acc: 0.9784
Epoch 14/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0086 - acc: 0.9972 - val_loss: 0.1061 - val_acc: 0.9785
Epoch 15/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0094 - acc: 0.9970 - val_loss: 0.0930 - val_acc: 0.9814
Epoch 16/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0065 - acc: 0.9978 - val_loss: 0.0862 - val_acc: 0.9820
Epoch 17/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0061 - acc: 0.9979 - val_loss: 0.0858 - val_acc: 0.9813
Epoch 18/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0048 - acc: 0.9987 - val_loss: 0.0823 - val_acc: 0.9822
Epoch 19/30
60000/60000 [==============================] - 4s 64us/step - loss: 0.0025 - acc: 0.9994 - val_loss: 0.0921 - val_acc: 0.9822
Epoch 20/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0027 - acc: 0.9993 - val_loss: 0.0755 - val_acc: 0.9857
Epoch 21/30
60000/60000 [==============================] - 4s 65us/step - loss: 0.0013 - acc: 0.9997 - val_loss: 0.0795 - val_acc: 0.9854
Epoch 22/30
60000/60000 [==============================] - 4s 68us/step - loss: 0.0051 - acc: 0.9985 - val_loss: 0.1151 - val_acc: 0.9790
Epoch 23/30
60000/60000 [==============================] - 4s 69us/step - loss: 0.0041 - acc: 0.9988 - val_loss: 0.0948 - val_acc: 0.9827
Epoch 24/30
60000/60000 [==============================] - 4s 66us/step - loss: 0.0027 - acc: 0.9994 - val_loss: 0.0780 - val_acc: 0.9846
Epoch 25/30
60000/60000 [==============================] - 4s 70us/step - loss: 6.5097e-04 - acc: 0.9999 - val_loss: 0.0763 - val_acc: 0.9859
Epoch 26/30
60000/60000 [==============================] - 4s 66us/step - loss: 4.0309e-04 - acc: 1.0000 - val_loss: 0.0761 - val_acc: 0.9862
Epoch 27/30
60000/60000 [==============================] - 4s 65us/step - loss: 3.3992e-04 - acc: 1.0000 - val_loss: 0.0765 - val_acc: 0.9865
Epoch 28/30
60000/60000 [==============================] - 4s 65us/step - loss: 3.2731e-04 - acc: 1.0000 - val_loss: 0.0769 - val_acc: 0.9864
Epoch 29/30
60000/60000 [==============================] - 4s 65us/step - loss: 3.1969e-04 - acc: 1.0000 - val_loss: 0.0773 - val_acc: 0.9864
Epoch 30/30
60000/60000 [==============================] - 4s 65us/step - loss: 3.1418e-04 - acc: 1.0000 - val_loss: 0.0777 - val_acc: 0.9866
```

### Tensorflow GPU
```
loss: 0.5564 / 0.1754	accuracy: 0.8143 / 0.9453	step 1  1.43 sec
loss: 0.1281 / 0.1149	accuracy: 0.9606 / 0.9647	step 2  2.57 sec
loss: 0.0816 / 0.1171	accuracy: 0.9748 / 0.9659	step 3  3.73 sec
loss: 0.0579 / 0.1603	accuracy: 0.9810 / 0.9555	step 4  4.88 sec
loss: 0.0420 / 0.1128	accuracy: 0.9867 / 0.9690	step 5  6.10 sec
loss: 0.0336 / 0.0932	accuracy: 0.9891 / 0.9766	step 6  7.35 sec
loss: 0.0265 / 0.0907	accuracy: 0.9911 / 0.9773	step 7  8.52 sec
loss: 0.0197 / 0.0805	accuracy: 0.9937 / 0.9800	step 8  9.67 sec
loss: 0.0143 / 0.0767	accuracy: 0.9954 / 0.9822	step 9  10.79 sec
loss: 0.0164 / 0.0934	accuracy: 0.9948 / 0.9789	step 10  11.96 sec
loss: 0.0116 / 0.0855	accuracy: 0.9958 / 0.9797	step 11  13.21 sec
loss: 0.0122 / 0.0928	accuracy: 0.9957 / 0.9816	step 12  14.40 sec
loss: 0.0099 / 0.0979	accuracy: 0.9965 / 0.9801	step 13  15.52 sec
loss: 0.0070 / 0.0879	accuracy: 0.9977 / 0.9812	step 14  16.63 sec
loss: 0.0058 / 0.0959	accuracy: 0.9980 / 0.9806	step 15  17.79 sec
loss: 0.0027 / 0.0811	accuracy: 0.9993 / 0.9837	step 16  19.04 sec
loss: 0.0046 / 0.1040	accuracy: 0.9985 / 0.9798	step 17  20.27 sec
loss: 0.0040 / 0.0929	accuracy: 0.9988 / 0.9819	step 18  21.36 sec
loss: 0.0047 / 0.1062	accuracy: 0.9986 / 0.9802	step 19  22.56 sec
loss: 0.0033 / 0.0958	accuracy: 0.9989 / 0.9822	step 20  23.74 sec
loss: 0.0026 / 0.0759	accuracy: 0.9993 / 0.9854	step 21  24.96 sec
loss: 0.0009 / 0.0805	accuracy: 0.9998 / 0.9854	step 22  26.15 sec
loss: 0.0002 / 0.0806	accuracy: 1.0000 / 0.9855	step 23  27.37 sec
loss: 0.0001 / 0.0814	accuracy: 1.0000 / 0.9856	step 24  28.53 sec
loss: 0.0001 / 0.0821	accuracy: 1.0000 / 0.9856	step 25  29.71 sec
loss: 0.0001 / 0.0827	accuracy: 1.0000 / 0.9857	step 26  30.90 sec
loss: 0.0000 / 0.0832	accuracy: 1.0000 / 0.9860	step 27  32.05 sec
loss: 0.0000 / 0.0837	accuracy: 1.0000 / 0.9860	step 28  33.17 sec
loss: 0.0000 / 0.0841	accuracy: 1.0000 / 0.9861	step 29  34.26 sec
loss: 0.0000 / 0.0845	accuracy: 1.0000 / 0.9861	step 30  35.38 sec
```

### Neural_Networks CPU
```
loss: 0.5581 / 0.1746	accuracy: 0.9483 / 0.9449	step 1  146.93 sec
loss: 0.1276 / 0.1342	accuracy: 0.9660 / 0.9595	step 2  317.82 sec
loss: 0.0822 / 0.1445	accuracy: 0.9691 / 0.9596	step 3  468.12 sec
loss: 0.0584 / 0.1139	accuracy: 0.9797 / 0.9682	step 4  618.91 sec
loss: 0.0439 / 0.1073	accuracy: 0.9830 / 0.9727	step 5  771.05 sec
loss: 0.0316 / 0.0972	accuracy: 0.9868 / 0.9741	step 6  920.41 sec
loss: 0.0257 / 0.0811	accuracy: 0.9920 / 0.9780	step 7  1065.09 sec
loss: 0.0201 / 0.0930	accuracy: 0.9904 / 0.9769	step 8  1216.67 sec
loss: 0.0177 / 0.0934	accuracy: 0.9919 / 0.9770	step 9  1372.86 sec
loss: 0.0136 / 0.0764	accuracy: 0.9958 / 0.9821	step 10  1532.15 sec
loss: 0.0123 / 0.0809	accuracy: 0.9968 / 0.9820	step 11  1690.73 sec
loss: 0.0075 / 0.0969	accuracy: 0.9957 / 0.9797	step 12  1845.61 sec
loss: 0.0067 / 0.0892	accuracy: 0.9962 / 0.9810	step 13  1994.63 sec
loss: 0.0075 / 0.0972	accuracy: 0.9959 / 0.9798	step 14  2148.36 sec
loss: 0.0041 / 0.0944	accuracy: 0.9976 / 0.9807	step 15  2310.95 sec
loss: 0.0035 / 0.0904	accuracy: 0.9984 / 0.9826	step 16  2471.67 sec
loss: 0.0029 / 0.1139	accuracy: 0.9966 / 0.9788	step 17  2628.52 sec
loss: 0.0028 / 0.1004	accuracy: 0.9980 / 0.9817	step 18  2789.36 sec
loss: 0.0060 / 0.1103	accuracy: 0.9957 / 0.9796	step 19  2938.77 sec
loss: 0.0040 / 0.0868	accuracy: 0.9993 / 0.9841	step 20  3096.48 sec
loss: 0.0018 / 0.0868	accuracy: 0.9990 / 0.9836	step 21  3252.47 sec
loss: 0.0021 / 0.0851	accuracy: 0.9994 / 0.9848	step 22  3418.10 sec
loss: 0.0017 / 0.0869	accuracy: 0.9997 / 0.9851	step 23  3578.99 sec
loss: 0.0014 / 0.0953	accuracy: 0.9994 / 0.9833	step 24  3738.64 sec
loss: 0.0007 / 0.0825	accuracy: 1.0000 / 0.9857	step 25  3893.44 sec
loss: 0.0002 / 0.0818	accuracy: 1.0000 / 0.9856	step 26  4041.92 sec
loss: 0.0001 / 0.0820	accuracy: 1.0000 / 0.9856	step 27  4198.91 sec
loss: 0.0001 / 0.0823	accuracy: 1.0000 / 0.9857	step 28  4346.84 sec
loss: 0.0000 / 0.0825	accuracy: 1.0000 / 0.9857	step 29  4497.12 sec
loss: 0.0000 / 0.0828	accuracy: 1.0000 / 0.9858	step 30  4646.76 sec

```
