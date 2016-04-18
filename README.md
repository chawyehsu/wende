Wende
---
Wende(问得) Chinese Question Answering, is a small and simple factoid chinese question answering system written in Python. You ask a chinese natural language question and Wende will try to give a certain answer of the question. This is still a work in progress, currently the system can only judge the type of the question that the user ask, which means, only the question classification module has been implemented.

## Run on Windows
**Your PC should be 64-bit and at least has 4G RAM!**

1. 安装 [Anaconda]。Anaconda 是一个集成的 Python 科学计算环境，可以免去在 Windows 下安装诸如 numpy、scipy 等科学计算类库的烦恼与痛苦。  
   请选择 **Python2.7 64-bit** 的版本进行安装，安装过程全部默认。
   ![conda_win_installer](https://cloud.githubusercontent.com/assets/5764917/14608348/e763f1f0-05b7-11e6-8507-f8a2fe0375ba.png)

2. 安装 [Microsoft Visual C++ Compiler for Python 2.7]  
   由于需要编译安装 [pyltp]，所以需要安装编译器 VC for Python

3. 安装 [Git-Bash]，安装过程全部默认。然后打开 Git-Bash，本地新建一个目录并克隆本仓库的源码
   ``` sh
   $ mkdir -p /c/workspace && cd /c/workspace
   $ git clone https://github.com/h404bi/wende
   $ cd wende/
   $ git submodule update --init --recursive
   ```
   稍等片刻克隆完毕后，可以在 Git-Bash 下看看刚才安装的 Anaconda 是否能正常使用。
   ``` sh
   $ conda -V
   $ winpty python --version
   $ conda list
   ```

4. 配置项目依赖环境  
   接着上面的 Git-Bash 窗口，安装项目依赖。

   先更新 conda 自带的类库
   ``` sh
   $ conda update --all --yes
   ```
   然后安装 requirements.txt 中的依赖
   ``` sh
   $ conda install --file requirements_conda.txt --yes
   $ pip install -r requirements_pip.txt
   ```
   之后编译安装 pyltp
   ``` sh
   $ cd pyltp/ && winpty python setup.py install
   ```
   查看已安装列表
   ``` sh
   $ cd .. && conda list
   ```

5. 下载相关模型（LTP/Wende/Word2Vec）  
   LTP（哈工大语言技术平台）：[百度云]，选择模型版本 3.3.0  
   Wende & Word2Vec：请查看 [releases]，其中也可以自行训练 Wende 的模型，具体请见 `wende/classification/model.py` 文件  
   所有模型文件请放到 `data/models/` 下对应的目录

6. 测试与运行  
   直接运行 Web App
   ``` sh
   $ winpty python -m app
   ```
   <kbd>Ctrl</kbd>+<kbd>C</kbd> 退出

   运行分类器评估测试
   ``` sh
   $ winpty python -m evaluation.evaluate_models
   ```
   运行特征提取评估测试
   ``` sh
   $ winpty python -m evaluation.evaluate_features
   ```

## Run on Linux / Mac
Coming soon...

## License
Code: MIT. Copyright (c) 2016 by Chawye Hsu
Data: [CC BY-NC-SA 4.0], except for the [ltp model] use its original license.


[Anaconda]: https://www.continuum.io/downloads
[Microsoft Visual C++ Compiler for Python 2.7]: https://www.microsoft.com/en-us/download/details.aspx?id=44266
[pyltp]: https://github.com/HIT-SCIR/pyltp
[Git-Bash]: https://git-for-windows.github.io/
[百度云]: http://pan.baidu.com/share/link?shareid=1988562907&uk=2738088569
[releases]: https://github.com/h404bi/wende/releases
[CC BY-NC-SA 4.0]: https://creativecommons.org/licenses/by-nc-sa/4.0/
[ltp model]: https://github.com/HIT-SCIR/ltp#模型