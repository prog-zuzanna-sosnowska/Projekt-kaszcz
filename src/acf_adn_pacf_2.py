from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ACVF(x, h):
    n = len(x)
    x_m = np.mean(x)
    h = abs(h)
    if h >= n:
        raise ValueError("Opóźnienie h nie może być większe lub równe długości danych")
    return (1 / n) * np.sum((x[:n - h] - x_m) * (x[h:] - x_m))


def ACF(x, h):
    return ACVF(x, h) / ACVF(x, 0)


def main():
    df = pd.read_csv(r'..\data\warsaw.csv')
    average_temperature = df['TAVG']

    # Średnia temperatura na przełomie lat 2020-2022
    df['DATE'] = pd.to_datetime(df['DATE'])
    df_2020_22 = df[df['DATE'].dt.year >= 2020]

    avg_temp_in_2020_22 = df_2020_22['TAVG']

    plot_acf(average_temperature, lags=len(df) // 2)
    plt.show()

    plot_pacf(average_temperature)
    plt.show()

    plot_acf(avg_temp_in_2020_22, lags=len(df_2020_22) // 2)
    plt.show()

    plot_pacf(avg_temp_in_2020_22)
    plt.show()


if __name__ == '__main__':
    main()
