# DynamicBackgroundEstimation

某天体画像処理ソフトウェアのオマージュ的処理プログラム。
天体画像の背景カブリを推定し減算処理することで除去します。

# DEMO

処理前（レベル補正済み）
![処理前](https://github.com/takashi-154/DynamicBackgroundExtraction/blob/main/sample/sample2.jpg)

ポイント指定画面
![ポイント指定画面](https://github.com/takashi-154/DynamicBackgroundExtraction/blob/main/sample/Figure_1.jpg)

処理後（レベル補正済み）
![処理後](https://github.com/takashi-154/DynamicBackgroundExtraction/blob/main/sample/sample1.jpg)

# Features

某ソフトのようにそこそこ背景カブリを除くことができます。
本プログラムでは、あらかじめ指定したポイントの背景値を使用し、4次関数の曲面フィッティングにより背景カブリを推定します。
画像のポイント指定もmatplotlibのGUI操作により行うことが可能です。
入力画像は現状**tiff,fits形式**に対応しています。
出力画像はデフォルトで**実数32bitのtiff,fits形式**です（16bit整数なども対応できるが未テスト）。

# Requirement

* astropy 4.1
* matplotlib 3.3.2
* numpy 1.19.2
* scipy 1.5.2
* tifffile 2020.9.3

# Installation

```bash
pip install -r requirements.txt
```

# Usage

jupyter notebookで動かすことを基本とします。
[sample_code.ipynb](https://github.com/takashi-154/DynamicBackgroundExtraction/blob/main/sample_code.ipynb)参照
処理したい画像のパスに変更して、上から順に進めていけば、問題なく処理できると思います。

# Note

一応ベータ版。今後の方針は**Future plan**に。

# Future plan

* 必要に応じて更新・拡張
* jupyter notebookから切り離して完全アプリ化
* フィッティング関数の選択
* 出力形式の汎用化

# Author

* takashi-154
* Twitter: [@Mazic_tell_Arts](https://twitter.com/Mazic_tell_Arts)

# License

"DynamicBackgroundEstimation" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
