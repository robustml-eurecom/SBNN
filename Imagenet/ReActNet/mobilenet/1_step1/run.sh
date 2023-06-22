clear
mkdir log
python3 train.py --data=/datasets/imagenet --batch_size=64 --learning_rate=1.25e-3 --epochs=128 --weight_decay=1e-5 | tee -a log/training.txt