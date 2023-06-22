clear
mkdir models
cp ../1_step1/models/checkpoint.pth.tar ./models/checkpoint_ba.pth.tar
mkdir log
# EC are the expected connections
# gamma is the hyperparameter forcing SBNNs to reach the desired EC level. If gamma is 0, then the training is the one of the standard BNN
python3 train.py --data=/datasets/imagenet --batch_size=64 --learning_rate=1.25e-3 --epochs=128 --weight_decay=0 --EC=0.10 --gamma=0.02 | tee -a log/training.txt
