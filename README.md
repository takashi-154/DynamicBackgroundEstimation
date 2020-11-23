# DynamicBackgroundEstimation

某天体画像処理ソフトウェアのオマージュ的処理プログラム。
天体画像の背景カブリを推定し減算処理することで除去します。

# DEMO

処理前（レベル補正済み）

ポイント指定画面

処理後（レベル補正済み）


# Features

某ソフトのようにそこそこ背景カブリを除くことができます。
本プログラムでは、あらかじめ指定したポイントの背景値を使用し、4次関数の曲面フィッティングにより背景カブリを推定します。
画像のポイント指定もmatplotlibのGUI操作により行うことが可能です。
入力画像は現状[tiff,fits]画像に対応しています。
出力画像はデフォルトで実数32bitの[tiff,fits]画像です（16bit整数なども対応できるが未テスト）。

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

jupyter notebookで動かすことを基本とする
`sample_code.ipynb`参照

# Note

一応ベータ版。必要に応じて更新・拡張。

# Author

* takashi-154
* Twitter: @Mazic_tell_Arts

# License

"DynamicBackgroundEstimation" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
