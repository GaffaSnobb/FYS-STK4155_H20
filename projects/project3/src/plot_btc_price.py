import datetime
import matplotlib.pyplot as plt
import pandas as pd


def plot_btc_price():
    """
    Plot the entire historical BTC/USD price data.
    """
    csv_path = "data/btc-usd-max.csv"
    df = pd.read_csv(csv_path, parse_dates=['snapped_at'])
    df = df.sort_values('snapped_at')   # Sort by date.
    price = df.price.values

    xticks = [""]*8
    xticks[1] = df.snapped_at.iloc[0].date()
    xticks[3] = df.snapped_at.iloc[1000].date()
    xticks[5] = df.snapped_at.iloc[2000].date()
    xticks[7] = df.snapped_at.iloc[2773].date()

    fig, ax = plt.subplots(figsize = (9, 7))
    ax.plot(price, color = "black")
    ax.tick_params(labelsize = 15)
    ax.set_ylabel("BTC price [USD]", fontsize = 15)
    ax.set_title("Historical BTC/USD price", fontsize = 15)
    ax.set_xticks([0, 0, 500, 1000, 1500, 2000, 2500, 2773])
    ax.set_xticklabels(xticks)
    ax.grid()
    fig.savefig(fname = "../fig/historical_price.png", dpi = 300)
    plt.show()
    pass


if __name__ == "__main__":
    plot_btc_price()