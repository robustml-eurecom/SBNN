# if you want to run the SBNN model run the following code changing EC (expected connections) and gamma (hyperparameter controlling the strength of the sparsity constraint). For the experiments we used the following (EC, gamma) values
# (0.50, 0.00)
# (0.25, 0.02)
# (0.10, 0.04)
# (0.05, 0.07)
# (0.04, 0.07)
# (0.03, 0.08)
# (0.02, 0.08)
# (0.01, 0.10)

python mainBNN.py --EC=0.5 --gamma=0.00 --lr=0.001 --batch_size=512 --epochs=500 --alpha=1.0 --learn=1
