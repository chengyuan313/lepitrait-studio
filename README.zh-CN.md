# EuroLepi ID

EuroLepi ID是一个只做欧洲蝴蝶图像物种鉴定的训练与推理框架。旧版LEPY、OCR、颜色、
形态测量和气候匹配模块已经移除。

## 现在能做什么

- 检查未来欧洲数据集的CSV清单；
- 防止同一标本进入训练集和测试集；
- 强制博物馆训练图片移除标签、二维码、馆藏号和比例尺；
- 按物种进行标本级训练/验证/测试划分；
- 使用MaxViT-T进行类别平衡微调；
- 输出Top-5候选；
- 低于阈值时返回“未知或需要专家复核”；
- 分别记录博物馆标准照、野外标准照和自然状态野外照。

## 拿到数据后的操作

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[ml]"

eurolepi validate data/manifest_unsplit.csv --before-split
eurolepi split data/manifest_unsplit.csv --output data/manifest.csv
eurolepi validate data/manifest.csv
eurolepi train configs/maxvit_tiny.yaml
eurolepi evaluate models/eurolepi_maxvit_tiny/best.pt data/manifest.csv
```

训练完成后启动GUI：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

清单格式见`data/manifest.example.csv`。详细要求见`docs/`。
