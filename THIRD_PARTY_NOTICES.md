# 第三方来源与许可

- HDSF方法参考：https://github.com/daifeizhai/PowerData_Prediction 。通过固定提交Git子模块引用，未将其源码或历史复制进本仓库；上游没有在所检查提交中提供独立LICENSE文件，不对其额外授予许可。
- Moirai2/uni2ts：https://github.com/SalesforceAIResearch/uni2ts 。通过固定提交引用，其上游许可证由原项目提供。
- Granite/PatchTST-FM/TinyTimeMixer：https://github.com/ibm-granite/granite-tsfm 。固定源码通过`third_party/granite-tsfm`子模块取得，提交见`docs/UPSTREAM_SOURCES.json`。
- Chronos-2、TiRex-2、TimesFM 2.5/3、Toto2及Moirai2的代码、软件包和模型权重各按原发布方条款使用。没有分发它们的权重。TimesFM3仅用于允许的非商业、非生产研究；其[官方许可](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/LICENSE)另禁止转发模型与衍生权重，因此微调检查点由使用者在本地依固定步骤重新训练。
- Google数据与说明的CC BY 4.0要求及NLR官方数据使用声明分别保留，不将一套数据许可扩展到所有数据或模型。

本次发布未替项目作者指定新的统一开源许可证；仓库公开性不替代各部分原有许可。数据、模型、第三方代码和本项目研究代码的权利范围需分别理解。
