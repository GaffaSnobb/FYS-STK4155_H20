import common

if __name__ == "__main__":
    q1 = common.FFNN(hidden_layer_sizes=(50, 20, 20), verbose=True)
    q1.train_neural_network(learning_rate=0.007)
    score = q1.predict(q1.X_test)
    print(score)

    # for learning_rate in np.logspace(-5, 0, 8):
    #     q1.train_neural_network_single(learning_rate)
    #     score = q1.predict_single(q1.X_test)
    #     print(f"score: {score} for learning rate: {learning_rate}")
    pass