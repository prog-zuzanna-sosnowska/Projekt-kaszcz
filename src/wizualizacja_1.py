import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv(r'..\data\warsaw.csv')

    # Informacje o brakach w danych
    print(df.isnull().sum())

    date = df['DATE']
    average_temperature = df['TAVG']

    # max_temperature = df['TMAX']
    # min_temperature = df['TMIN']

    plt.figure(figsize=(9, 4))

    plt.plot(average_temperature)
    plt.title('Średnia temperatura w Warszawie na przełomie lat 1993-2022')
    plt.xlabel('Czas (w dniach)')
    plt.ylabel(r'Temperatura ($\degree$C)')
    plt.tight_layout()
    plt.savefig("..\\img\\wizualizacja_danych.png")
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
