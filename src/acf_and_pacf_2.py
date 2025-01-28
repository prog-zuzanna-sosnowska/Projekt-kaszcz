from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ACVF(x, lags):
    n = len(x)
    x_m = np.mean(x)
    y = np.zeros(len(lags))
    lags = abs(lags)
    for i, lag in enumerate(lags):
        if lag >= n:
            raise ValueError('lag must be less than n')
        y[i] = (1 / n) * np.sum((x[:n - lag] - x_m) * (x[lag:] - x_m))
    return y


def ACF(x, lags):
    return ACVF(x, lags) / ACVF(x, np.zeros_like(lags))


def raw_data(data):
    data = data.values
    n = len(data)
    lags = np.arange(0, n // 2)
    acf_values = ACF(data, lags=lags)

    plot_acf(data, lags=lags)                       # Wykres z użyciem funkcji bibliotecznej
    plt.plot(acf_values)                            # Wykres z użyciem własnej implementacji
    plt.title('Fukcja ACF dla surowych danych')
    plt.show()

    plot_pacf(data)
    plt.show()


def data_without_linear_trend(data):
    data = data.values
    n = len(data)
    t = np.arange(0, n)
    lags = np.arange(0, n // 2)
    def m(x, a, b): return a * x + b

    a, b = np.polyfit(t, data, 1)

    data_without_trend = data - m(t, a, b)

    plt.plot(data, label='surowe dane')
    plt.plot(data_without_trend, label='dane po usunięciu trendu')
    plt.title('Wykres dla danych po usunięciu trendu')
    plt.legend()
    plt.show()

    acf_values = ACF(data_without_trend, lags=lags)

    plot_acf(data_without_trend, lags=lags)
    plt.plot(acf_values)
    plt.show()

    plot_pacf(data_without_trend)
    plt.show()

    return data_without_trend


def data_without_linear_trend_ver2(data):
    data = data.values
    n = len(data)
    t = np.arange(0, n)
    lags = np.arange(0, n // 2)

    def m(x, a, b): return a * x + b

    linear_reg = LinearRegression()
    linear_reg.fit(t.reshape(-1, 1), data)
    trend_predicted = linear_reg.predict(t.reshape(-1, 1))
    data_without_linear_trend = data - trend_predicted

    plt.plot(data)
    plt.plot(data_without_linear_trend)
    plt.show()

    acf_values = ACF(data_without_linear_trend, lags=lags)

    plot_acf(data_without_linear_trend, lags=lags)
    plt.plot(acf_values)
    plt.show()

    plot_pacf(data_without_linear_trend)
    plt.show()

    return data_without_linear_trend


def main():
    df = pd.read_csv(r'..\data\warsaw.csv')
    avg_temp = df['TAVG']

    # raw_data(data=avg_temp)

    # Średnia temperatura na przełomie lat 1993-2020
    df['DATE'] = pd.to_datetime(df['DATE'])
    df_2020 = df[df['DATE'].dt.year < 2020]
    avg_temp_in_2020 = df_2020['TAVG']

    # raw_data(data=avg_temp_in_2020)

    # Usuwanie trendu z danych:

    # avg_temp_without_trend = data_without_linear_trend(data=avg_temp)
    # data_without_linear_trend(data=avg_temp_in_2020)

    # Usuwanie trendu liniowego (wersja 2)

    # avg_temp_without_trend_ver2 = data_without_linear_trend_ver2(data=avg_temp)
    # data_without_linear_trend_ver2(data=avg_temp_in_2020)


    n = len(avg_temp)
    t = np.arange(0, n)
    lags = np.arange(0, n // 2)

    def m(x, a, b): return a * x + b

    a, b = np.polyfit(t, avg_temp, 1)

    avg_temp_without_linear_trend = avg_temp - m(t, a, b)

    # Usuwanie funkcji okresowej

    def sine_func2(x, A, omega, phi):
        return A * np.sin(omega * x + phi)

    def sine_func(x, A, omega):
        return A * np.sin(omega * x)

    n = len(avg_temp)
    t = np.arange(0, n)
    lags = np.arange(0, n // 2)

    p0 = [(max(avg_temp_without_linear_trend) - min(avg_temp_without_linear_trend))/2, 2 * np.pi / 365, np.pi / 2]
    params, _ = curve_fit(sine_func2, t, avg_temp_without_linear_trend, p0=p0)
    fitted_sine_wave = sine_func2(t, *params)

    avg_temp_without_seasonal = avg_temp_without_linear_trend - fitted_sine_wave

    # print((max(avg_temp_without_linear_trend) - min(avg_temp_without_linear_trend))/2)
    #
    # plt.plot(avg_temp_without_linear_trend)
    # plt.plot(avg_temp_without_seasonal)
    # plt.show()
    #
    # plot_acf(avg_temp_without_linear_trend, lags=lags)
    # plt.show()
    #
    # plot_acf(avg_temp_without_seasonal, lags=lags)
    # plt.show()

    # Periodogram

    periodogram = np.abs(np.fft.fft(avg_temp_without_linear_trend)) ** 2 / len(
        avg_temp_without_linear_trend)
    frequencies = np.fft.fftfreq(len(avg_temp_without_linear_trend))

    plt.figure(figsize=(10, 5))
    plt.plot(frequencies[:len(avg_temp_without_linear_trend) // 2],
             periodogram[:len(avg_temp_without_linear_trend) // 2])
    plt.title('Periodogram')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.grid(True)
    plt.show()

    peak_indices = np.argsort(periodogram)[::-1][:6]
    peaks_frequency = frequencies[peak_indices]
    peaks_period = 1 / peaks_frequency

    print("Top 5 Peaks (Frequency, Period):")
    for i in range(len(peaks_frequency)):
        print(f"Peak {i + 1}: {peaks_frequency[i]:.4f}, {peaks_period[i]:.2f}")


if __name__ == '__main__':
    main()
