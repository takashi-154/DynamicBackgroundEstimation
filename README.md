# DynamicBackgroundEstimation

某天体画像処理ソフトウェアのオマージュ的処理プログラム。  
天体画像の背景カブリを推定し減算処理することで除去します。

# New!!

（2021/02/24）Version 0.4.0 が公開されました。[簡単な説明書](https://github.com/takashi-154/DynamicBackgroundExtraction/blob/main/instructions_for_gui.md)も記載しました。

# DEMO

処理前（レベル補正済み）  
![処理前](https://github.com/takashi-154/DynamicBackgroundExtraction/blob/main/sample_img/sample2.jpg)

GUI（画像読み込み時）  
![GUI（画像読み込み時）](https://github.com/takashi-154/DynamicBackgroundEstimation/blob/main/sample_img/screenshot1.png)

GUI（ポイント指定時）  
![GUI（ポイント指定時）](https://github.com/takashi-154/DynamicBackgroundEstimation/blob/main/sample_img/screenshot2.png)

処理後（レベル補正済み）  
![処理後](https://github.com/takashi-154/DynamicBackgroundExtraction/blob/main/sample_img/sample1.jpg)

# Features

某ソフトのようにそこそこ背景カブリを除くことができます。  
本プログラムでは、あらかじめ指定したポイントの背景値を使用し、n次関数の曲面フィッティングにより背景カブリを推定します。  
画像のポイント指定もGUIアプリにより行うことが可能です。
入力画像は**tiff,fits形式**に対応しています。  
出力画像はデフォルトで**実数32bitのtiff,fits形式**です（16bit整数なども対応）。

# Requirement

* astropy 4.1
* matplotlib 3.3.2
* numpy 1.19.2
* scipy 1.5.2
* tifffile 2020.9.3
* scikit-image 0.17.2

# Installation

```bash
pip install -r requirements.txt
```

# Usage

~~jupyter notebookで動かすことを基本とします。~~  
~~[sample_code.ipynb](https://github.com/takashi-154/DynamicBackgroundExtraction/blob/main/sample_code.ipynb)参照。~~  
~~処理したい画像のパスに変更して、上から順に進めていけば、問題なく処理できると思います。~~  
GUIアプリを作成しました。かなり直感的に操作できます。説明書は[こちら](https://github.com/takashi-154/DynamicBackgroundExtraction/blob/main/instructions_for_gui.md)。

# Note

今後の方針は**Future plan**に。

# Future plan

* 必要に応じて更新・拡張

# Author

* takashi-154
* Twitter: [@Mazic_tell_Arts](https://twitter.com/Mazic_tell_Arts)

# License

"DynamicBackgroundEstimation" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
