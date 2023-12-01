# Decopon

## 某スイカゲームモドキのためのAI
カゴの中の座標の高い20個の座標を取得，球を落とすべき座標を予測する．

Getting Started
===============

poetryをインストールします。
[poetry](https://python-poetry.org/docs/)のページを参考に，自分の環境に合った方法でインストールしてください．

```
curl -sSL https://install.python-poetry.org | python -
```

decoponを使えるようにするため，installします．

```
poetry install
```

installできたら，ゲームを起動しましょう．デフォルトではAIモデルがゲームをプレイします．

```
poetry run python src/main.py
```

モデルを学習させる場合

```
poetry run python src/train.py
```

モデルのテストはゲームの起動と同様です．```test.py```は使用しません．
