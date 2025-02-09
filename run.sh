RUN_NUM=$RANDOM
python train.py --model_des $RUN_NUM --dataset '4H-data-v2-3841' --settings '4H-data-v2-3841' --epochs 100
python test.py --model_des $RUN_NUM --dataset '4H-data-v2-3841' --settings '4H-data-v2-3841'