import matplotlib.pyplot as plt
from lstm import CryptoPrediction

def plot():
    q = CryptoPrediction()
    q.train_model()

    plt.plot(q.loss, label="train")
    plt.plot(q.val_loss, label="test")
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()

    # y_predict = q.model.predict(q.scaled_price)
    y_predict = q.model.predict(q.X_test)
    y_test = q.scaler.inverse_transform(q.y_test)
    y_train = q.scaler.inverse_transform(q.y_train)
    y_predict = q.scaler.inverse_transform(y_predict)

    # plt.plot(q.price, label="price")
    plt.plot(y_test, label="y_test")
    # plt.plot(y_train, label="y_train")
    # plt.plot(np.concatenate((y_train, y_test)), label="conc")
    plt.plot(y_predict, label="Predicted Price", color='red')
    plt.title('Bitcoin price prediction')
    plt.xlabel('Time [days]')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.show()
    pass


if __name__ == "__main__":
    plot()