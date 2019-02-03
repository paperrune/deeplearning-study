## Results
### Keras CPU
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
simple_rnn_1 (SimpleRNN)     (None, 128)               20096     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 21,386
Trainable params: 21,386
Non-trainable params: 0
_________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/100
60000/60000 [==============================] - 6s 105us/step - loss: 1.4169 - acc: 0.5180 - val_loss: 1.7660 - val_acc: 0.4440
Epoch 2/100
60000/60000 [==============================] - 6s 99us/step - loss: 0.6534 - acc: 0.7940 - val_loss: 0.4416 - val_acc: 0.8548
Epoch 3/100
60000/60000 [==============================] - 7s 112us/step - loss: 0.3545 - acc: 0.8933 - val_loss: 0.2806 - val_acc: 0.9178
Epoch 4/100
60000/60000 [==============================] - 6s 106us/step - loss: 0.2667 - acc: 0.9219 - val_loss: 0.2705 - val_acc: 0.9173
Epoch 5/100
60000/60000 [==============================] - 6s 106us/step - loss: 0.2017 - acc: 0.9401 - val_loss: 0.1553 - val_acc: 0.9526
Epoch 6/100
60000/60000 [==============================] - 6s 105us/step - loss: 0.1723 - acc: 0.9503 - val_loss: 0.1466 - val_acc: 0.9556
Epoch 7/100
60000/60000 [==============================] - 6s 104us/step - loss: 0.1528 - acc: 0.9555 - val_loss: 0.1518 - val_acc: 0.9558
Epoch 8/100
60000/60000 [==============================] - 6s 104us/step - loss: 0.1408 - acc: 0.9587 - val_loss: 0.1397 - val_acc: 0.9573
Epoch 9/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.1301 - acc: 0.9616 - val_loss: 0.1319 - val_acc: 0.9601
Epoch 10/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.1448 - acc: 0.9584 - val_loss: 0.1265 - val_acc: 0.9635
Epoch 11/100
60000/60000 [==============================] - 6s 100us/step - loss: 0.1082 - acc: 0.9684 - val_loss: 0.1153 - val_acc: 0.9676
Epoch 12/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.1060 - acc: 0.9690 - val_loss: 0.1047 - val_acc: 0.9706
Epoch 13/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0887 - acc: 0.9744 - val_loss: 0.0962 - val_acc: 0.9735
Epoch 14/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0847 - acc: 0.9750 - val_loss: 0.1352 - val_acc: 0.9648
Epoch 15/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0792 - acc: 0.9770 - val_loss: 0.0784 - val_acc: 0.9776
Epoch 16/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0804 - acc: 0.9766 - val_loss: 0.0846 - val_acc: 0.9765
Epoch 17/100
60000/60000 [==============================] - 6s 100us/step - loss: 0.0717 - acc: 0.9788 - val_loss: 0.0862 - val_acc: 0.9738
Epoch 18/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0709 - acc: 0.9794 - val_loss: 0.0760 - val_acc: 0.9784
Epoch 19/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0650 - acc: 0.9805 - val_loss: 0.0891 - val_acc: 0.9753
Epoch 20/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0646 - acc: 0.9808 - val_loss: 0.0805 - val_acc: 0.9760
Epoch 21/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0621 - acc: 0.9818 - val_loss: 0.0972 - val_acc: 0.9745
Epoch 22/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0564 - acc: 0.9831 - val_loss: 0.0725 - val_acc: 0.9794
Epoch 23/100
60000/60000 [==============================] - 6s 100us/step - loss: 0.0561 - acc: 0.9827 - val_loss: 0.0843 - val_acc: 0.9784
Epoch 24/100
60000/60000 [==============================] - 6s 100us/step - loss: 0.0553 - acc: 0.9837 - val_loss: 0.0821 - val_acc: 0.9767
Epoch 25/100
60000/60000 [==============================] - 6s 104us/step - loss: 0.0519 - acc: 0.9847 - val_loss: 0.0726 - val_acc: 0.9779
Epoch 26/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0535 - acc: 0.9838 - val_loss: 0.0790 - val_acc: 0.9780
Epoch 27/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0514 - acc: 0.9844 - val_loss: 0.0685 - val_acc: 0.9797
Epoch 28/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0475 - acc: 0.9857 - val_loss: 0.0734 - val_acc: 0.9793
Epoch 29/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0449 - acc: 0.9864 - val_loss: 0.0781 - val_acc: 0.9771
Epoch 30/100
60000/60000 [==============================] - 6s 100us/step - loss: 0.0470 - acc: 0.9862 - val_loss: 0.0778 - val_acc: 0.9791
Epoch 31/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0424 - acc: 0.9876 - val_loss: 0.0900 - val_acc: 0.9761
Epoch 32/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0438 - acc: 0.9870 - val_loss: 0.0727 - val_acc: 0.9810
Epoch 33/100
60000/60000 [==============================] - 6s 104us/step - loss: 0.0390 - acc: 0.9884 - val_loss: 0.0683 - val_acc: 0.9794
Epoch 34/100
60000/60000 [==============================] - 6s 104us/step - loss: 0.0402 - acc: 0.9876 - val_loss: 0.1108 - val_acc: 0.9714
Epoch 35/100
60000/60000 [==============================] - 6s 104us/step - loss: 0.0436 - acc: 0.9869 - val_loss: 0.0738 - val_acc: 0.9785
Epoch 36/100
60000/60000 [==============================] - 7s 113us/step - loss: 0.0417 - acc: 0.9872 - val_loss: 0.0797 - val_acc: 0.9774
Epoch 37/100
60000/60000 [==============================] - 6s 108us/step - loss: 0.0407 - acc: 0.9877 - val_loss: 0.0776 - val_acc: 0.9805
Epoch 38/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0398 - acc: 0.9883 - val_loss: 0.0714 - val_acc: 0.9805
Epoch 39/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0351 - acc: 0.9895 - val_loss: 0.0690 - val_acc: 0.9816
Epoch 40/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0304 - acc: 0.9906 - val_loss: 0.0779 - val_acc: 0.9772
Epoch 41/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0346 - acc: 0.9893 - val_loss: 0.0732 - val_acc: 0.9803
Epoch 42/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0372 - acc: 0.9889 - val_loss: 0.0835 - val_acc: 0.9776
Epoch 43/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.4125 - acc: 0.8728 - val_loss: 0.1826 - val_acc: 0.9482
Epoch 44/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.1123 - acc: 0.9675 - val_loss: 0.1055 - val_acc: 0.9710
Epoch 45/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0787 - acc: 0.9769 - val_loss: 0.0851 - val_acc: 0.9757
Epoch 46/100
60000/60000 [==============================] - 6s 105us/step - loss: 0.0655 - acc: 0.9808 - val_loss: 0.1026 - val_acc: 0.9707
Epoch 47/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0591 - acc: 0.9831 - val_loss: 0.0821 - val_acc: 0.9761
Epoch 48/100
60000/60000 [==============================] - 6s 105us/step - loss: 0.0541 - acc: 0.9842 - val_loss: 0.0769 - val_acc: 0.9781
Epoch 49/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0535 - acc: 0.9844 - val_loss: 0.0705 - val_acc: 0.9794
Epoch 50/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0486 - acc: 0.9850 - val_loss: 0.0645 - val_acc: 0.9808
Epoch 51/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0420 - acc: 0.9873 - val_loss: 0.0753 - val_acc: 0.9789
Epoch 52/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0440 - acc: 0.9866 - val_loss: 0.0751 - val_acc: 0.9802
Epoch 53/100
60000/60000 [==============================] - 6s 104us/step - loss: 0.0397 - acc: 0.9878 - val_loss: 0.0684 - val_acc: 0.9814
Epoch 54/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0385 - acc: 0.9888 - val_loss: 0.1090 - val_acc: 0.9711
Epoch 55/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0398 - acc: 0.9880 - val_loss: 0.0738 - val_acc: 0.9795
Epoch 56/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0361 - acc: 0.9893 - val_loss: 0.0733 - val_acc: 0.9802
Epoch 57/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0310 - acc: 0.9903 - val_loss: 0.0759 - val_acc: 0.9816
Epoch 58/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0319 - acc: 0.9907 - val_loss: 0.0709 - val_acc: 0.9817
Epoch 59/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0316 - acc: 0.9902 - val_loss: 0.0628 - val_acc: 0.9841
Epoch 60/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0306 - acc: 0.9903 - val_loss: 0.0708 - val_acc: 0.9803
Epoch 61/100
60000/60000 [==============================] - 6s 104us/step - loss: 0.0308 - acc: 0.9906 - val_loss: 0.0811 - val_acc: 0.9791
Epoch 62/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0273 - acc: 0.9913 - val_loss: 0.0733 - val_acc: 0.9815
Epoch 63/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0274 - acc: 0.9915 - val_loss: 0.0626 - val_acc: 0.9815
Epoch 64/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0298 - acc: 0.9906 - val_loss: 0.0996 - val_acc: 0.9772
Epoch 65/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0330 - acc: 0.9900 - val_loss: 0.0889 - val_acc: 0.9764
Epoch 66/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0303 - acc: 0.9910 - val_loss: 0.0692 - val_acc: 0.9822
Epoch 67/100
60000/60000 [==============================] - 6s 105us/step - loss: 0.0286 - acc: 0.9912 - val_loss: 0.0757 - val_acc: 0.9819
Epoch 68/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0302 - acc: 0.9911 - val_loss: 0.0745 - val_acc: 0.9814
Epoch 69/100
60000/60000 [==============================] - 6s 100us/step - loss: 0.0268 - acc: 0.9918 - val_loss: 0.0763 - val_acc: 0.9814
Epoch 70/100
60000/60000 [==============================] - 6s 100us/step - loss: 0.0245 - acc: 0.9925 - val_loss: 0.0698 - val_acc: 0.9798
Epoch 71/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0273 - acc: 0.9917 - val_loss: 0.0635 - val_acc: 0.9827
Epoch 72/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0290 - acc: 0.9911 - val_loss: 0.1008 - val_acc: 0.9759
Epoch 73/100
60000/60000 [==============================] - 6s 104us/step - loss: 0.0237 - acc: 0.9928 - val_loss: 0.0639 - val_acc: 0.9840
Epoch 74/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0233 - acc: 0.9929 - val_loss: 0.0645 - val_acc: 0.9830
Epoch 75/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0223 - acc: 0.9935 - val_loss: 0.0674 - val_acc: 0.9819
Epoch 76/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0246 - acc: 0.9925 - val_loss: 0.0804 - val_acc: 0.9774
Epoch 77/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0280 - acc: 0.9916 - val_loss: 0.0769 - val_acc: 0.9789
Epoch 78/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0214 - acc: 0.9935 - val_loss: 0.0798 - val_acc: 0.9806
Epoch 79/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0165 - acc: 0.9951 - val_loss: 0.0626 - val_acc: 0.9834
Epoch 80/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0250 - acc: 0.9924 - val_loss: 0.0689 - val_acc: 0.9805
Epoch 81/100
60000/60000 [==============================] - 7s 112us/step - loss: 0.0245 - acc: 0.9927 - val_loss: 0.0682 - val_acc: 0.9815
Epoch 82/100
60000/60000 [==============================] - 7s 109us/step - loss: 0.0222 - acc: 0.9930 - val_loss: 0.0989 - val_acc: 0.9807
Epoch 83/100
60000/60000 [==============================] - 6s 99us/step - loss: 0.0258 - acc: 0.9922 - val_loss: 0.0623 - val_acc: 0.9832
Epoch 84/100
60000/60000 [==============================] - 6s 100us/step - loss: 0.0195 - acc: 0.9939 - val_loss: 0.0689 - val_acc: 0.9801
Epoch 85/100
60000/60000 [==============================] - 6s 99us/step - loss: 0.0195 - acc: 0.9943 - val_loss: 0.0927 - val_acc: 0.9797
Epoch 86/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0167 - acc: 0.9949 - val_loss: 0.0818 - val_acc: 0.9797
Epoch 87/100
60000/60000 [==============================] - 6s 106us/step - loss: 0.0263 - acc: 0.9923 - val_loss: 0.1082 - val_acc: 0.9787
Epoch 88/100
60000/60000 [==============================] - 6s 105us/step - loss: 0.0225 - acc: 0.9932 - val_loss: 0.0689 - val_acc: 0.9817
Epoch 89/100
60000/60000 [==============================] - 7s 112us/step - loss: 0.0116 - acc: 0.9967 - val_loss: 0.0752 - val_acc: 0.9830
Epoch 90/100
60000/60000 [==============================] - 7s 110us/step - loss: 0.0178 - acc: 0.9946 - val_loss: 0.0662 - val_acc: 0.9820
Epoch 91/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0183 - acc: 0.9942 - val_loss: 0.0766 - val_acc: 0.9820
Epoch 92/100
60000/60000 [==============================] - 6s 104us/step - loss: 0.0179 - acc: 0.9946 - val_loss: 0.0776 - val_acc: 0.9828
Epoch 93/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0124 - acc: 0.9962 - val_loss: 0.0700 - val_acc: 0.9836
Epoch 94/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0125 - acc: 0.9960 - val_loss: 0.0797 - val_acc: 0.9823
Epoch 95/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0188 - acc: 0.9943 - val_loss: 0.0763 - val_acc: 0.9832
Epoch 96/100
60000/60000 [==============================] - 6s 102us/step - loss: 0.0133 - acc: 0.9959 - val_loss: 0.0671 - val_acc: 0.9847
Epoch 97/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0127 - acc: 0.9963 - val_loss: 0.2610 - val_acc: 0.9456
Epoch 98/100
60000/60000 [==============================] - 6s 103us/step - loss: 0.0145 - acc: 0.9955 - val_loss: 0.0673 - val_acc: 0.9814
Epoch 99/100
60000/60000 [==============================] - 6s 101us/step - loss: 0.0207 - acc: 0.9937 - val_loss: 0.1028 - val_acc: 0.9759
Epoch 100/100
60000/60000 [==============================] - 6s 104us/step - loss: 0.0179 - acc: 0.9947 - val_loss: 0.0665 - val_acc: 0.9842
```

### Tensorflow GPU
```
loss: 1.5025 / 1.1987	accuracy: 0.4855 / 0.6013	step 1  18.03 sec
loss: 0.5094 / 0.2499	accuracy: 0.8435 / 0.9240	step 2  34.73 sec
loss: 0.2658 / 0.1847	accuracy: 0.9220 / 0.9433	step 3  50.95 sec
loss: 0.2130 / 0.1818	accuracy: 0.9375 / 0.9473	step 4  67.41 sec
loss: 0.1686 / 0.1677	accuracy: 0.9513 / 0.9513	step 5  83.64 sec
loss: 0.1515 / 0.1422	accuracy: 0.9566 / 0.9581	step 6  99.85 sec
loss: 0.1335 / 0.1288	accuracy: 0.9617 / 0.9604	step 7  116.16 sec
loss: 0.1209 / 0.1154	accuracy: 0.9650 / 0.9649	step 8  132.27 sec
loss: 0.1097 / 0.1082	accuracy: 0.9680 / 0.9677	step 9  148.40 sec
loss: 0.1425 / 0.1221	accuracy: 0.9586 / 0.9626	step 10  164.70 sec
loss: 0.0998 / 0.1072	accuracy: 0.9716 / 0.9702	step 11  180.59 sec
loss: 0.0982 / 0.1170	accuracy: 0.9717 / 0.9659	step 12  196.74 sec
loss: 0.0849 / 0.1096	accuracy: 0.9756 / 0.9684	step 13  212.95 sec
loss: 0.0838 / 0.0995	accuracy: 0.9761 / 0.9711	step 14  229.06 sec
loss: 0.0765 / 0.0806	accuracy: 0.9777 / 0.9774	step 15  245.22 sec
loss: 0.0709 / 0.0993	accuracy: 0.9797 / 0.9726	step 16  261.42 sec
loss: 0.0833 / 0.0859	accuracy: 0.9762 / 0.9743	step 17  277.52 sec
loss: 0.0720 / 0.0838	accuracy: 0.9792 / 0.9751	step 18  293.64 sec
loss: 0.0641 / 0.0928	accuracy: 0.9811 / 0.9733	step 19  309.63 sec
loss: 0.0688 / 0.1125	accuracy: 0.9798 / 0.9683	step 20  325.79 sec
loss: 0.0662 / 0.0807	accuracy: 0.9809 / 0.9766	step 21  342.01 sec
loss: 0.0644 / 0.0743	accuracy: 0.9814 / 0.9779	step 22  357.93 sec
loss: 0.0547 / 0.0768	accuracy: 0.9844 / 0.9778	step 23  373.91 sec
loss: 0.0530 / 0.0848	accuracy: 0.9846 / 0.9747	step 24  390.08 sec
loss: 0.0544 / 0.0757	accuracy: 0.9835 / 0.9773	step 25  406.10 sec
loss: 0.0513 / 0.0935	accuracy: 0.9850 / 0.9722	step 26  422.19 sec
loss: 0.0497 / 0.0728	accuracy: 0.9852 / 0.9790	step 27  438.48 sec
loss: 0.0477 / 0.0740	accuracy: 0.9858 / 0.9777	step 28  454.74 sec
loss: 0.0417 / 0.0783	accuracy: 0.9879 / 0.9783	step 29  471.06 sec
loss: 0.0464 / 0.0663	accuracy: 0.9863 / 0.9811	step 30  487.83 sec
loss: 0.0411 / 0.0650	accuracy: 0.9873 / 0.9794	step 31  504.52 sec
loss: 0.0399 / 0.0781	accuracy: 0.9877 / 0.9762	step 32  521.34 sec
loss: 0.0502 / 0.0684	accuracy: 0.9851 / 0.9784	step 33  538.21 sec
loss: 0.0437 / 0.0695	accuracy: 0.9871 / 0.9825	step 34  554.78 sec
loss: 0.0439 / 0.0670	accuracy: 0.9873 / 0.9806	step 35  571.73 sec
loss: 0.0386 / 0.0612	accuracy: 0.9887 / 0.9816	step 36  588.18 sec
loss: 0.0345 / 0.0683	accuracy: 0.9896 / 0.9794	step 37  604.67 sec
loss: 0.0394 / 0.0652	accuracy: 0.9880 / 0.9822	step 38  621.54 sec
loss: 0.0373 / 0.0802	accuracy: 0.9891 / 0.9765	step 39  637.80 sec
loss: 0.0335 / 0.0722	accuracy: 0.9904 / 0.9801	step 40  656.71 sec
loss: 0.0359 / 0.0637	accuracy: 0.9892 / 0.9827	step 41  676.15 sec
loss: 0.0356 / 0.0608	accuracy: 0.9890 / 0.9825	step 42  695.09 sec
loss: 0.0388 / 0.0640	accuracy: 0.9886 / 0.9823	step 43  713.94 sec
loss: 0.0322 / 0.0603	accuracy: 0.9901 / 0.9838	step 44  733.03 sec
loss: 0.0347 / 0.0762	accuracy: 0.9893 / 0.9818	step 45  752.39 sec
loss: 0.0524 / 0.0728	accuracy: 0.9849 / 0.9799	step 46  771.41 sec
loss: 0.0366 / 0.0743	accuracy: 0.9889 / 0.9800	step 47  790.52 sec
loss: 0.0332 / 0.0622	accuracy: 0.9898 / 0.9811	step 48  809.46 sec
loss: 0.0291 / 0.0690	accuracy: 0.9913 / 0.9811	step 49  828.92 sec
loss: 0.0265 / 0.0814	accuracy: 0.9922 / 0.9778	step 50  848.68 sec
loss: 0.0260 / 0.0895	accuracy: 0.9924 / 0.9792	step 51  868.11 sec
loss: 0.0272 / 0.0664	accuracy: 0.9918 / 0.9825	step 52  887.49 sec
loss: 0.0260 / 0.0774	accuracy: 0.9923 / 0.9820	step 53  907.02 sec
loss: 0.0284 / 0.0713	accuracy: 0.9914 / 0.9822	step 54  926.64 sec
loss: 0.0251 / 0.0759	accuracy: 0.9925 / 0.9815	step 55  945.68 sec
loss: 0.0265 / 0.0858	accuracy: 0.9921 / 0.9794	step 56  965.08 sec
loss: 0.0273 / 0.0701	accuracy: 0.9919 / 0.9837	step 57  983.65 sec
loss: 0.0273 / 0.0666	accuracy: 0.9920 / 0.9826	step 58  1001.96 sec
loss: 0.0244 / 0.0702	accuracy: 0.9923 / 0.9819	step 59  1020.26 sec
loss: 0.0248 / 0.0610	accuracy: 0.9926 / 0.9838	step 60  1038.75 sec
loss: 0.0231 / 0.0740	accuracy: 0.9929 / 0.9816	step 61  1057.52 sec
loss: 0.0218 / 0.0749	accuracy: 0.9932 / 0.9832	step 62  1075.92 sec
loss: 0.0189 / 0.0641	accuracy: 0.9942 / 0.9838	step 63  1094.63 sec
loss: 0.0223 / 0.2470	accuracy: 0.9934 / 0.9466	step 64  1113.35 sec
loss: 0.0251 / 0.1024	accuracy: 0.9926 / 0.9732	step 65  1132.71 sec
loss: 0.0260 / 0.0721	accuracy: 0.9922 / 0.9803	step 66  1152.02 sec
loss: 0.0222 / 0.0662	accuracy: 0.9930 / 0.9835	step 67  1171.65 sec
loss: 0.0187 / 0.0708	accuracy: 0.9945 / 0.9828	step 68  1191.30 sec
loss: 0.0317 / 0.0664	accuracy: 0.9899 / 0.9831	step 69  1210.70 sec
loss: 0.0244 / 0.0733	accuracy: 0.9927 / 0.9798	step 70  1230.64 sec
loss: 0.0230 / 0.0769	accuracy: 0.9931 / 0.9800	step 71  1250.05 sec
loss: 0.0242 / 0.0620	accuracy: 0.9928 / 0.9829	step 72  1268.28 sec
loss: 0.0235 / 0.0810	accuracy: 0.9928 / 0.9790	step 73  1285.66 sec
loss: 0.0219 / 0.0691	accuracy: 0.9935 / 0.9827	step 74  1303.22 sec
loss: 0.0165 / 0.0810	accuracy: 0.9951 / 0.9823	step 75  1321.39 sec
loss: 0.0179 / 0.0768	accuracy: 0.9948 / 0.9820	step 76  1341.84 sec
loss: 0.0210 / 0.0653	accuracy: 0.9939 / 0.9845	step 77  1360.66 sec
loss: 0.0161 / 0.1309	accuracy: 0.9952 / 0.9748	step 78  1377.71 sec
loss: 0.0208 / 0.0785	accuracy: 0.9934 / 0.9805	step 79  1394.60 sec
loss: 0.0266 / 0.0773	accuracy: 0.9920 / 0.9803	step 80  1413.08 sec
loss: 0.0196 / 0.0829	accuracy: 0.9941 / 0.9791	step 81  1431.54 sec
loss: 0.0159 / 0.0702	accuracy: 0.9951 / 0.9833	step 82  1449.72 sec
loss: 0.0193 / 0.0776	accuracy: 0.9942 / 0.9811	step 83  1470.40 sec
loss: 0.0194 / 0.0971	accuracy: 0.9940 / 0.9755	step 84  1490.82 sec
loss: 0.0250 / 0.0883	accuracy: 0.9924 / 0.9798	step 85  1511.01 sec
loss: 0.0209 / 0.0700	accuracy: 0.9940 / 0.9831	step 86  1531.09 sec
loss: 0.0137 / 0.0685	accuracy: 0.9958 / 0.9845	step 87  1550.84 sec
loss: 0.0253 / 0.0669	accuracy: 0.9927 / 0.9816	step 88  1571.02 sec
loss: 0.0228 / 0.0731	accuracy: 0.9929 / 0.9810	step 89  1590.30 sec
loss: 0.0235 / 0.0781	accuracy: 0.9931 / 0.9825	step 90  1610.12 sec
loss: 0.0137 / 0.0677	accuracy: 0.9959 / 0.9846	step 91  1630.13 sec
loss: 0.0166 / 0.0677	accuracy: 0.9951 / 0.9839	step 92  1649.16 sec
loss: 0.0113 / 0.0739	accuracy: 0.9965 / 0.9830	step 93  1669.45 sec
loss: 0.0090 / 0.0774	accuracy: 0.9974 / 0.9837	step 94  1689.31 sec
loss: 0.0137 / 0.0827	accuracy: 0.9958 / 0.9832	step 95  1708.77 sec
loss: 0.0190 / 0.0782	accuracy: 0.9942 / 0.9825	step 96  1728.73 sec
loss: 0.0179 / 0.0650	accuracy: 0.9946 / 0.9840	step 97  1748.22 sec
loss: 0.0237 / 0.0661	accuracy: 0.9926 / 0.9843	step 98  1768.30 sec
loss: 0.0127 / 0.0788	accuracy: 0.9964 / 0.9838	step 99  1788.10 sec
loss: 0.0155 / 0.1301	accuracy: 0.9952 / 0.9753	step 100  1807.40 sec
```

### Neural_Networks GPU
```
loss: 1.3288 / 0.8707	accuracy: 0.6841 / 0.6807	step 1  17.90 sec
loss: 0.4369 / 0.3170	accuracy: 0.9030 / 0.9028	step 2  35.73 sec
loss: 0.2507 / 0.2123	accuracy: 0.9365 / 0.9403	step 3  53.64 sec
loss: 0.1900 / 0.1897	accuracy: 0.9463 / 0.9479	step 4  71.71 sec
loss: 0.1607 / 0.1747	accuracy: 0.9508 / 0.9489	step 5  89.81 sec
loss: 0.1490 / 0.1348	accuracy: 0.9605 / 0.9596	step 6  107.90 sec
loss: 0.1295 / 0.2374	accuracy: 0.9353 / 0.9336	step 7  125.96 sec
loss: 0.1144 / 0.1120	accuracy: 0.9709 / 0.9673	step 8  144.12 sec
loss: 0.1122 / 0.1055	accuracy: 0.9750 / 0.9695	step 9  162.20 sec
loss: 0.1507 / 0.1951	accuracy: 0.9484 / 0.9478	step 10  180.26 sec
loss: 0.1226 / 0.1201	accuracy: 0.9670 / 0.9658	step 11  198.33 sec
loss: 0.1075 / 0.1022	accuracy: 0.9739 / 0.9698	step 12  216.36 sec
loss: 0.0882 / 0.0940	accuracy: 0.9780 / 0.9733	step 13  234.46 sec
loss: 0.0827 / 0.1122	accuracy: 0.9691 / 0.9661	step 14  252.57 sec
loss: 0.0730 / 0.0838	accuracy: 0.9823 / 0.9763	step 15  270.69 sec
loss: 0.0698 / 0.0809	accuracy: 0.9821 / 0.9773	step 16  288.96 sec
loss: 0.0692 / 0.1061	accuracy: 0.9749 / 0.9687	step 17  307.16 sec
loss: 0.0667 / 0.0803	accuracy: 0.9839 / 0.9773	step 18  325.35 sec
loss: 0.0622 / 0.1212	accuracy: 0.9718 / 0.9669	step 19  343.66 sec
loss: 0.0596 / 0.0773	accuracy: 0.9852 / 0.9785	step 20  361.76 sec
loss: 0.0540 / 0.0832	accuracy: 0.9830 / 0.9779	step 21  379.97 sec
loss: 0.0548 / 0.0732	accuracy: 0.9883 / 0.9805	step 22  398.21 sec
loss: 0.0515 / 0.0840	accuracy: 0.9865 / 0.9769	step 23  416.36 sec
loss: 0.0521 / 0.0763	accuracy: 0.9867 / 0.9784	step 24  434.35 sec
loss: 0.0485 / 0.0844	accuracy: 0.9830 / 0.9752	step 25  452.55 sec
loss: 0.0491 / 0.0742	accuracy: 0.9863 / 0.9781	step 26  470.74 sec
loss: 0.0445 / 0.0652	accuracy: 0.9902 / 0.9819	step 27  488.81 sec
loss: 0.0431 / 0.0724	accuracy: 0.9906 / 0.9810	step 28  506.97 sec
loss: 0.0489 / 0.0787	accuracy: 0.9877 / 0.9796	step 29  525.05 sec
loss: 0.0422 / 0.0706	accuracy: 0.9910 / 0.9803	step 30  543.31 sec
loss: 0.0431 / 0.0731	accuracy: 0.9871 / 0.9800	step 31  561.44 sec
loss: 0.0447 / 0.0683	accuracy: 0.9903 / 0.9799	step 32  579.50 sec
loss: 0.0367 / 0.0645	accuracy: 0.9917 / 0.9820	step 33  597.61 sec
loss: 0.0371 / 0.1365	accuracy: 0.9735 / 0.9659	step 34  615.53 sec
loss: 0.0498 / 0.0648	accuracy: 0.9915 / 0.9834	step 35  633.28 sec
loss: 0.0362 / 0.0845	accuracy: 0.9850 / 0.9754	step 36  651.02 sec
loss: 0.0344 / 0.0776	accuracy: 0.9879 / 0.9792	step 37  668.77 sec
loss: 0.0370 / 0.0656	accuracy: 0.9926 / 0.9822	step 38  686.74 sec
loss: 0.0312 / 0.0633	accuracy: 0.9926 / 0.9820	step 39  704.97 sec
loss: 0.0335 / 0.0729	accuracy: 0.9922 / 0.9822	step 40  723.07 sec
loss: 0.0372 / 0.0807	accuracy: 0.9897 / 0.9778	step 41  741.37 sec
loss: 0.0355 / 0.0849	accuracy: 0.9879 / 0.9761	step 42  759.67 sec
loss: 0.0324 / 0.0986	accuracy: 0.9847 / 0.9745	step 43  777.82 sec
loss: 0.0317 / 0.0677	accuracy: 0.9925 / 0.9816	step 44  796.12 sec
loss: 0.0295 / 0.0842	accuracy: 0.9874 / 0.9780	step 45  814.33 sec
loss: 0.0296 / 0.0725	accuracy: 0.9923 / 0.9830	step 46  832.48 sec
loss: 0.0270 / 0.0573	accuracy: 0.9952 / 0.9847	step 47  850.79 sec
loss: 0.0276 / 0.0726	accuracy: 0.9938 / 0.9812	step 48  868.98 sec
loss: 0.0289 / 0.0708	accuracy: 0.9941 / 0.9813	step 49  887.20 sec
loss: 0.0262 / 0.0931	accuracy: 0.9829 / 0.9708	step 50  905.60 sec
loss: 0.0270 / 0.0647	accuracy: 0.9953 / 0.9819	step 51  923.85 sec
loss: 0.0283 / 0.0683	accuracy: 0.9944 / 0.9832	step 52  942.09 sec
loss: 0.0261 / 0.0671	accuracy: 0.9943 / 0.9825	step 53  960.35 sec
loss: 0.0221 / 0.0762	accuracy: 0.9929 / 0.9818	step 54  978.57 sec
loss: 0.0292 / 0.0725	accuracy: 0.9941 / 0.9818	step 55  996.94 sec
loss: 0.0231 / 0.0766	accuracy: 0.9903 / 0.9799	step 56  1015.19 sec
loss: 0.0293 / 0.0654	accuracy: 0.9938 / 0.9826	step 57  1033.37 sec
loss: 0.0284 / 0.0722	accuracy: 0.9922 / 0.9806	step 58  1051.78 sec
loss: 0.0229 / 0.0984	accuracy: 0.9845 / 0.9749	step 59  1069.90 sec
loss: 0.0239 / 0.0853	accuracy: 0.9926 / 0.9811	step 60  1088.10 sec
loss: 0.0215 / 0.0726	accuracy: 0.9930 / 0.9810	step 61  1106.22 sec
loss: 0.0221 / 0.0700	accuracy: 0.9948 / 0.9817	step 62  1124.12 sec
loss: 0.0181 / 0.0785	accuracy: 0.9954 / 0.9809	step 63  1142.15 sec
loss: 0.0221 / 0.0752	accuracy: 0.9972 / 0.9821	step 64  1160.49 sec
loss: 0.0172 / 0.2365	accuracy: 0.9567 / 0.9473	step 65  1178.81 sec
loss: 0.0250 / 0.0727	accuracy: 0.9962 / 0.9833	step 66  1197.16 sec
loss: 0.0232 / 0.0864	accuracy: 0.9934 / 0.9797	step 67  1215.33 sec
loss: 0.0205 / 0.0727	accuracy: 0.9951 / 0.9809	step 68  1233.51 sec
loss: 0.0241 / 0.0772	accuracy: 0.9933 / 0.9821	step 69  1251.79 sec
loss: 0.0284 / 0.0664	accuracy: 0.9947 / 0.9827	step 70  1270.00 sec
loss: 0.0241 / 0.0747	accuracy: 0.9938 / 0.9816	step 71  1288.12 sec
loss: 0.0209 / 0.0843	accuracy: 0.9914 / 0.9787	step 72  1306.50 sec
loss: 0.0449 / 0.0730	accuracy: 0.9920 / 0.9804	step 73  1324.67 sec
loss: 0.0261 / 0.0684	accuracy: 0.9953 / 0.9828	step 74  1342.75 sec
loss: 0.0173 / 0.0782	accuracy: 0.9950 / 0.9810	step 75  1360.70 sec
loss: 0.0154 / 0.0763	accuracy: 0.9923 / 0.9801	step 76  1378.71 sec
loss: 0.0240 / 0.0648	accuracy: 0.9968 / 0.9825	step 77  1396.56 sec
loss: 0.0146 / 0.0645	accuracy: 0.9962 / 0.9849	step 78  1414.40 sec
loss: 0.0151 / 0.0641	accuracy: 0.9968 / 0.9840	step 79  1432.22 sec
loss: 0.0242 / 0.0599	accuracy: 0.9957 / 0.9844	step 80  1450.13 sec
loss: 0.0222 / 0.0693	accuracy: 0.9948 / 0.9827	step 81  1467.98 sec
loss: 0.0195 / 0.0752	accuracy: 0.9953 / 0.9832	step 82  1485.82 sec
loss: 0.0201 / 0.0696	accuracy: 0.9949 / 0.9821	step 83  1503.68 sec
loss: 0.0186 / 0.0767	accuracy: 0.9932 / 0.9809	step 84  1521.51 sec
loss: 0.0247 / 0.0760	accuracy: 0.9925 / 0.9811	step 85  1539.36 sec
loss: 0.0239 / 0.0649	accuracy: 0.9965 / 0.9835	step 86  1557.19 sec
loss: 0.0181 / 0.1021	accuracy: 0.9922 / 0.9789	step 87  1575.03 sec
loss: 0.0183 / 0.0642	accuracy: 0.9968 / 0.9842	step 88  1592.86 sec
loss: 0.0153 / 0.0888	accuracy: 0.9891 / 0.9774	step 89  1610.70 sec
loss: 0.0190 / 0.0716	accuracy: 0.9966 / 0.9830	step 90  1628.54 sec
loss: 0.0114 / 0.0729	accuracy: 0.9970 / 0.9833	step 91  1646.35 sec
loss: 0.0141 / 0.0767	accuracy: 0.9971 / 0.9815	step 92  1664.18 sec
loss: 0.0114 / 0.0722	accuracy: 0.9967 / 0.9831	step 93  1682.01 sec
loss: 0.0214 / 0.0752	accuracy: 0.9940 / 0.9825	step 94  1699.84 sec
loss: 0.0235 / 0.0715	accuracy: 0.9958 / 0.9819	step 95  1717.69 sec
loss: 0.0136 / 0.0713	accuracy: 0.9974 / 0.9822	step 96  1735.55 sec
loss: 0.0189 / 0.0864	accuracy: 0.9907 / 0.9773	step 97  1753.39 sec
loss: 0.0164 / 0.0758	accuracy: 0.9950 / 0.9810	step 98  1771.23 sec
loss: 0.0128 / 0.0709	accuracy: 0.9980 / 0.9844	step 99  1789.07 sec
loss: 0.0107 / 0.0668	accuracy: 0.9981 / 0.9834	step 100  1806.91 sec
```
