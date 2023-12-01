# Decopon

某スイカゲームモドキのためのAI

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

installできたら，ゲームを起動しましょう．

```
poetry run python src/main.py
```

モデルを学習させる場合

```
poetry run python src/train.py
```
