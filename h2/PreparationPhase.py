import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def preprocess(df: pd.DataFrame, corr_threshold: float = 0.81) -> pd.DataFrame:
    # числовые столбцы
    num_df = df.select_dtypes("number")

    # кореляционная матрица
    corr = num_df.corr().abs()

    # оставляем столбец, если в его строке нет корреляций
    keep_mask = ~(corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                      .gt(corr_threshold).any())
    num_df = num_df.loc[:, keep_mask]

    # удаляем строки с пропущенными значениями
    num_df = num_df.dropna()

    # масштабируем
    scaler = StandardScaler()
    num_df = num_df.astype(float)
    num_df.loc[:, :] = scaler.fit_transform(num_df)

    print(num_df)
    return num_df


def plot_pca_3d(df: pd.DataFrame, target_col: str) -> None:
    X = df.drop(columns=target_col)
    y = df[target_col]

    # проецируем в двумерное пространство
    X_red = PCA(n_components=2).fit_transform(X)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(X_red[:, 0], X_red[:, 1], y, c=y, cmap="viridis", s=20)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel(target_col)
    plt.show()
