## Hlir builder sample

Hlir builder sample

本repo包含了多个常见op和神经网络结构在hlir的实现,用于展示如何基于tops sdk搭建常见op或网络结构的hlir

## 安装

本项目运行前请确保安装了tops sdk和驱动

```sh
$ dpkg -i tops-sdk.deb
```

## 编译

```
mkdir build && cd build
cmake .. -DOPType=Add  //构建需要编译的op类型，具体见operator文件夹内
make //编译
```

目前已实现的op类型：

| Add           | Add_fp16      | Add_fp16_without_convert    | AveragePool    | BatchNorm   |
|:-------------:|:-------------:|:---------------------------:|:--------------:|:-----------:|
| **Conv2D**    | **Convert**   | **Copy**                    | **DotGeneral** | **Dropout** |
| **Less**      | **MaxPool**   | **GlobalAveragePool**       | **Mul**        | **OneHot**  |
| **Relu**      | **Reshape**   | **ReduceMean**              | **Squeeze**    | **Resize**  |
| **Softmax**   | **Split**     | **MaxPool_by_ReduceWindow** | **Sqrt**       | **Slice**   |
| **Conv**      | **Gemm**      | **Gather**                  |                | **Tanh**    |
| **Transpose** | **Unsqueeze** | **Sigmoid**                 | **Sub**        | **TopK**    |
| **Concat**    | **Gather**    | **Clip**                    | **Flatten**    | **Pow**     |
| **Shape**     | **Tile**      |                             |                |             |

目前已实现的神经网络结构类型：

| ResidualBlock | ResidualBlock_fp16_without_convert |
| ------------- | ---------------------------------- |

or，执行以下脚本可自动验证所有op

```
./test.sh
```

## TODO

- [ ] 完整模型的搭建实例（mnist等）

- [ ] 更多网络结构的实例
