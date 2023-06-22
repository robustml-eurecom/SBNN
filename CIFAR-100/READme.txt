# if you want to launch experiments with the original ReActNet-18 BNN model open a terminal in this folder and run

python mainBNN.py --lr=0.001 --batch_size=512 --epochs=300 --alpha=1.0 --learn=0

# if you want to run the SBNN method run the following code changing EC (expected connections) and gamma (hyperparameter controlling the strength of the sparsity constraint). For the experiments we used the following (EC, gamma) values
# (0.50, 0.00)
# (0.05, 0.05)
# (0.04, 0.05)
# (0.03, 0.05)
# (0.02, 0.06)
# (0.01, 0.07)

python mainBNN.py --EC=0.5 --gamma=0.00 --lr=0.001 --batch_size=512 --epochs=300 --alpha=1.0 --learn=1

# if you want to run the Subbit method run the following command. To control the number of bits change bit_num.
# bit_num = 6 means Subbit 6/9 -> Subbit 0.67
# in the original work for CIFAR-10 they used bit_num in {4,5,6} and we did the same for CIFAR-100

python main.py --bit_num=6 --lr=0.001 --batch_size=512 --epochs=300 --alpha=1.0

