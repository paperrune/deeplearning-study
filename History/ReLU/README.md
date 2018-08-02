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
60000/60000 [==============================] - 5s 88us/step - loss: 0.6507 - acc: 0.7763 - val_loss: 0.2817 - val_acc: 0.9115
Epoch 2/30
60000/60000 [==============================] - 5s 85us/step - loss: 0.1332 - acc: 0.9600 - val_loss: 0.1419 - val_acc: 0.9552
Epoch 3/30
60000/60000 [==============================] - 5s 86us/step - loss: 0.0847 - acc: 0.9746 - val_loss: 0.1172 - val_acc: 0.9638
Epoch 4/30
60000/60000 [==============================] - 5s 87us/step - loss: 0.0579 - acc: 0.9833 - val_loss: 0.0893 - val_acc: 0.9736
Epoch 5/30
60000/60000 [==============================] - 5s 87us/step - loss: 0.0412 - acc: 0.9883 - val_loss: 0.0768 - val_acc: 0.9778
Epoch 6/30
60000/60000 [==============================] - 5s 87us/step - loss: 0.0298 - acc: 0.9917 - val_loss: 0.0761 - val_acc: 0.9790
Epoch 7/30
60000/60000 [==============================] - 5s 86us/step - loss: 0.0224 - acc: 0.9939 - val_loss: 0.0770 - val_acc: 0.9792
Epoch 8/30
60000/60000 [==============================] - 5s 86us/step - loss: 0.0166 - acc: 0.9955 - val_loss: 0.0829 - val_acc: 0.9783
Epoch 9/30
60000/60000 [==============================] - 5s 85us/step - loss: 0.0135 - acc: 0.9968 - val_loss: 0.0748 - val_acc: 0.9811
Epoch 10/30
60000/60000 [==============================] - 5s 86us/step - loss: 0.0099 - acc: 0.9974 - val_loss: 0.0773 - val_acc: 0.9817
Epoch 11/30
60000/60000 [==============================] - 5s 87us/step - loss: 0.0288 - acc: 0.9940 - val_loss: 0.1109 - val_acc: 0.9733
Epoch 12/30
60000/60000 [==============================] - 5s 85us/step - loss: 0.0136 - acc: 0.9957 - val_loss: 0.0841 - val_acc: 0.9798
Epoch 13/30
60000/60000 [==============================] - 5s 86us/step - loss: 0.0056 - acc: 0.9987 - val_loss: 0.0798 - val_acc: 0.9808
Epoch 14/30
60000/60000 [==============================] - 5s 88us/step - loss: 0.0038 - acc: 0.9993 - val_loss: 0.0761 - val_acc: 0.9828
Epoch 15/30
60000/60000 [==============================] - 5s 86us/step - loss: 0.0022 - acc: 0.9998 - val_loss: 0.0775 - val_acc: 0.9825
Epoch 16/30
60000/60000 [==============================] - 5s 86us/step - loss: 0.0015 - acc: 0.9999 - val_loss: 0.0773 - val_acc: 0.9835
Epoch 17/30
60000/60000 [==============================] - 5s 84us/step - loss: 0.0011 - acc: 1.0000 - val_loss: 0.0774 - val_acc: 0.9838
Epoch 18/30
60000/60000 [==============================] - 5s 84us/step - loss: 9.1454e-04 - acc: 1.0000 - val_loss: 0.0789 - val_acc: 0.9836
Epoch 19/30
60000/60000 [==============================] - 5s 85us/step - loss: 7.8885e-04 - acc: 1.0000 - val_loss: 0.0795 - val_acc: 0.9840
Epoch 20/30
60000/60000 [==============================] - 5s 85us/step - loss: 7.0874e-04 - acc: 1.0000 - val_loss: 0.0802 - val_acc: 0.9841
Epoch 21/30
60000/60000 [==============================] - 5s 89us/step - loss: 6.5568e-04 - acc: 1.0000 - val_loss: 0.0809 - val_acc: 0.9837
Epoch 22/30
60000/60000 [==============================] - 5s 89us/step - loss: 6.1529e-04 - acc: 1.0000 - val_loss: 0.0814 - val_acc: 0.9838
Epoch 23/30
60000/60000 [==============================] - 5s 86us/step - loss: 5.8273e-04 - acc: 1.0000 - val_loss: 0.0819 - val_acc: 0.9838
Epoch 24/30
60000/60000 [==============================] - 5s 87us/step - loss: 5.5607e-04 - acc: 1.0000 - val_loss: 0.0824 - val_acc: 0.9837
Epoch 25/30
60000/60000 [==============================] - 5s 90us/step - loss: 5.3345e-04 - acc: 1.0000 - val_loss: 0.0828 - val_acc: 0.9838
Epoch 26/30
60000/60000 [==============================] - 5s 83us/step - loss: 5.1408e-04 - acc: 1.0000 - val_loss: 0.0832 - val_acc: 0.9838
Epoch 27/30
60000/60000 [==============================] - 5s 91us/step - loss: 4.9758e-04 - acc: 1.0000 - val_loss: 0.0836 - val_acc: 0.9838
Epoch 28/30
60000/60000 [==============================] - 6s 94us/step - loss: 4.8303e-04 - acc: 1.0000 - val_loss: 0.0840 - val_acc: 0.9838
Epoch 29/30
60000/60000 [==============================] - 5s 89us/step - loss: 4.7004e-04 - acc: 1.0000 - val_loss: 0.0844 - val_acc: 0.9837
Epoch 30/30
60000/60000 [==============================] - 5s 87us/step - loss: 4.5859e-04 - acc: 1.0000 - val_loss: 0.0847 - val_acc: 0.9837
```

### Tensorflow GPU
```
loss: 0.6688 / 0.3229    accuracy: 0.7719 / 0.8970    step 1  1.75 sec
loss: 0.1352 / 0.1342    accuracy: 0.9590 / 0.9577    step 2  3.21 sec
loss: 0.0836 / 0.1013    accuracy: 0.9750 / 0.9709    step 3  4.63 sec
loss: 0.0578 / 0.0923    accuracy: 0.9835 / 0.9727    step 4  6.05 sec
loss: 0.0403 / 0.0897    accuracy: 0.9883 / 0.9736    step 5  7.49 sec
loss: 0.0291 / 0.0861    accuracy: 0.9921 / 0.9760    step 6  8.91 sec
loss: 0.0208 / 0.0894    accuracy: 0.9946 / 0.9766    step 7  10.34 sec
loss: 0.0157 / 0.0878    accuracy: 0.9962 / 0.9780    step 8  11.74 sec
loss: 0.0229 / 0.0888    accuracy: 0.9945 / 0.9759    step 9  13.13 sec
loss: 0.0122 / 0.0840    accuracy: 0.9968 / 0.9789    step 10  14.55 sec
loss: 0.0069 / 0.0832    accuracy: 0.9986 / 0.9795    step 11  15.84 sec
loss: 0.0048 / 0.0773    accuracy: 0.9992 / 0.9814    step 12  17.18 sec
loss: 0.0029 / 0.0769    accuracy: 0.9996 / 0.9824    step 13  18.55 sec
loss: 0.0020 / 0.0796    accuracy: 0.9998 / 0.9819    step 14  19.96 sec
loss: 0.0014 / 0.0801    accuracy: 0.9999 / 0.9831    step 15  21.37 sec
loss: 0.0009 / 0.0805    accuracy: 1.0000 / 0.9829    step 16  22.77 sec
loss: 0.0007 / 0.0803    accuracy: 1.0000 / 0.9834    step 17  24.13 sec
loss: 0.0006 / 0.0805    accuracy: 1.0000 / 0.9834    step 18  25.45 sec
loss: 0.0005 / 0.0808    accuracy: 1.0000 / 0.9834    step 19  26.80 sec
loss: 0.0004 / 0.0812    accuracy: 1.0000 / 0.9834    step 20  28.13 sec
loss: 0.0004 / 0.0817    accuracy: 1.0000 / 0.9835    step 21  29.50 sec
loss: 0.0004 / 0.0820    accuracy: 1.0000 / 0.9832    step 22  30.91 sec
loss: 0.0003 / 0.0825    accuracy: 1.0000 / 0.9833    step 23  32.25 sec
loss: 0.0003 / 0.0829    accuracy: 1.0000 / 0.9831    step 24  33.61 sec
loss: 0.0003 / 0.0834    accuracy: 1.0000 / 0.9832    step 25  35.00 sec
loss: 0.0003 / 0.0837    accuracy: 1.0000 / 0.9833    step 26  36.33 sec
loss: 0.0002 / 0.0842    accuracy: 1.0000 / 0.9833    step 27  37.67 sec
loss: 0.0002 / 0.0845    accuracy: 1.0000 / 0.9834    step 28  39.02 sec
loss: 0.0002 / 0.0849    accuracy: 1.0000 / 0.9834    step 29  40.48 sec
loss: 0.0002 / 0.0852    accuracy: 1.0000 / 0.9834    step 30  41.87 sec
```

### Neural_Networks CPU
```
loss: 0.6773 / 0.3326	accuracy: 0.8974 / 0.8973	step 1  88.50 sec
loss: 0.1364 / 0.1614	accuracy: 0.9539 / 0.9479	step 2  174.04 sec
loss: 0.0849 / 0.1004	accuracy: 0.9758 / 0.9696	step 3  260.28 sec
loss: 0.0584 / 0.0844	accuracy: 0.9834 / 0.9745	step 4  376.38 sec
loss: 0.0414 / 0.0820	accuracy: 0.9869 / 0.9774	step 5  476.76 sec
loss: 0.0302 / 0.0786	accuracy: 0.9892 / 0.9783	step 6  563.12 sec
loss: 0.0215 / 0.0811	accuracy: 0.9911 / 0.9774	step 7  646.92 sec
loss: 0.0156 / 0.0804	accuracy: 0.9933 / 0.9783	step 8  729.58 sec
loss: 0.0110 / 0.0823	accuracy: 0.9940 / 0.9781	step 9  811.99 sec
loss: 0.0091 / 0.0818	accuracy: 0.9956 / 0.9783	step 10  898.41 sec
loss: 0.0105 / 0.1011	accuracy: 0.9915 / 0.9768	step 11  984.41 sec
loss: 0.0111 / 0.0780	accuracy: 0.9980 / 0.9818	step 12  1070.00 sec
loss: 0.0045 / 0.0813	accuracy: 0.9982 / 0.9810	step 13  1155.26 sec
loss: 0.0028 / 0.0796	accuracy: 0.9987 / 0.9821	step 14  1240.34 sec
loss: 0.0016 / 0.0797	accuracy: 0.9996 / 0.9823	step 15  1325.93 sec
loss: 0.0010 / 0.0804	accuracy: 0.9997 / 0.9826	step 16  1411.12 sec
loss: 0.0007 / 0.0789	accuracy: 0.9999 / 0.9838	step 17  1496.05 sec
loss: 0.0006 / 0.0794	accuracy: 1.0000 / 0.9838	step 18  1588.33 sec
loss: 0.0005 / 0.0795	accuracy: 1.0000 / 0.9837	step 19  1683.13 sec
loss: 0.0004 / 0.0799	accuracy: 1.0000 / 0.9839	step 20  1778.10 sec
loss: 0.0004 / 0.0801	accuracy: 1.0000 / 0.9840	step 21  1873.28 sec
loss: 0.0003 / 0.0804	accuracy: 1.0000 / 0.9838	step 22  1967.62 sec
loss: 0.0003 / 0.0808	accuracy: 1.0000 / 0.9835	step 23  2062.47 sec
loss: 0.0003 / 0.0812	accuracy: 1.0000 / 0.9837	step 24  2157.32 sec
loss: 0.0003 / 0.0816	accuracy: 1.0000 / 0.9838	step 25  2252.21 sec
loss: 0.0002 / 0.0819	accuracy: 1.0000 / 0.9837	step 26  2347.67 sec
loss: 0.0002 / 0.0823	accuracy: 1.0000 / 0.9837	step 27  2440.31 sec
loss: 0.0002 / 0.0826	accuracy: 1.0000 / 0.9837	step 28  2525.39 sec
loss: 0.0002 / 0.0829	accuracy: 1.0000 / 0.9837	step 29  2609.84 sec
loss: 0.0002 / 0.0832	accuracy: 1.0000 / 0.9839	step 30  2694.28 sec



```
