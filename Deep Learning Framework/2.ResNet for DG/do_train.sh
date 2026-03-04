##--------------------------------------------
#Training network
#==============
#**Author**: `zhibin Li`__
#""""
#Training fushion network
#PN_AF
#==============
#**Author**: `zhibin Li`__
##--------------------------------------------


echo "Start train1......"
nohup python -u train.py --Kcross_num=1 --length_set=300 > ./result/log/train_Res1_L300.log 2>&1 &
wait
echo "Start train2......"
nohup python -u train.py --Kcross_num=1 --length_set=250 > ./result/log/train_Res2_L250.log 2>&1 &
wait
echo "Start train3......"
nohup python -u train.py --Kcross_num=1 --length_set=200 > ./result/log/train_Res3_L200.log 2>&1 &
wait
echo "Start train4......"
nohup python -u train.py --Kcross_num=1 --length_set=150 > ./result/log/train_Res4_L150.log 2>&1 &
wait
echo "Start train5......"
nohup python -u train.py --Kcross_num=1 --length_set=100 > ./result/log/train_Res5_L100.log 2>&1 &
wait
echo "Start train6......"
nohup python -u train.py --Kcross_num=1 --length_set=50 > ./result/log/train_Res6_L50.log 2>&1 &
wait
echo "Start train7......"
nohup python -u train.py --Kcross_num=1 --length_set=40 > ./result/log/train_Res7_L40.log 2>&1 &
wait
echo "Start train8......"
nohup python -u train.py --Kcross_num=1 --length_set=20 > ./result/log/train_Res8_L20.log 2>&1 &
wait
echo "Start train9......"
nohup python -u train.py --Kcross_num=1 --length_set=10 > ./result/log/train_Res9_L10.log 2>&1 &
wait

echo "Finished......"