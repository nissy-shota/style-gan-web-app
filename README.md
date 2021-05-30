# style-gan-web-app

## styleGAN

```shell
.
├── config.yaml
├── data
│  └── images
├── dataset.py
├── main.py
├── results
│   └── style_GAN_result.jpg
├── styleGAN.py
└── utils.py
```
- config.yaml  
  config file  
- main.py  
  ここで，Styleと対象ファイルを選択（yamlファイルを書き換える必要あり）  
- styleGAN.py  
  styleGANのモデル本体
- utils.py  
  表示と保存  
- dataset.py  
  擬似的なデータローダ  
- data  
  サンプルデータ  
- results  
  サンプル結果  