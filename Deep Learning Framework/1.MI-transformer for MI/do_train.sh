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

#echo "Start train1......"
#nohup python -u train.py --Kcross_num=1 > ./result/log/train1.log 2>&1 &
#wait
#echo "Start train2......"
#nohup python -u train.py --Kcross_num=2 > ./result/log/train2.log 2>&1 &
#wait
#echo "Start train3......"
#nohup python -u train.py --Kcross_num=3 > ./result/log/train3.log 2>&1 &
#wait
#echo "Start train4......"
#nohup python -u train.py --Kcross_num=4 > ./result/log/train4.log 2>&1 &
#wait
#echo "Start train5......"
#nohup python -u train.py --Kcross_num=5 > ./result/log/train5.log 2>&1 &
#wait
#echo "Start train6......"
#nohup python -u train.py --Kcross_num=6 > ./result/log/train6.log 2>&1 &
#wait
#echo "Finished......"


echo "Start train1......"
nohup python -u train.py --Kcross_num=1 > ./result/log/train_NMIM1.log 2>&1 &
wait
echo "Start train2......"
nohup python -u train.py --Kcross_num=2 > ./result/log/train_NMIM2.log 2>&1 &
wait
echo "Start train3......"
nohup python -u train.py --Kcross_num=3 > ./result/log/train_NMIM3.log 2>&1 &
wait
echo "Start train4......"
nohup python -u train.py --Kcross_num=4 > ./result/log/train_NMIM4.log 2>&1 &
wait
echo "Start train5......"
nohup python -u train.py --Kcross_num=5 > ./result/log/train_NMIM5.log 2>&1 &
wait
echo "Start train6......"
nohup python -u train.py --Kcross_num=6 > ./result/log/train_NMIM6.log 2>&1 &
wait
echo "Finished......"