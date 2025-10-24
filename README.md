# BMW Intern - CASE 1: Case -> Item

把Makefile中的month换成对应的月份
```bash
make


            acc   | f1_macro | hit@1 | hit@3 | hit@5 | hit@10
1-6>>>7     0.714 | 0.226    | 0.714 | 0.784 | 0.814 | 0.849
1-7>>>8     0.733 | 0.234    | 0.733 | 0.807 | 0.832 | 0.864
1-8>>>9     0.705 | 0.237    | 0.705 | 0.780 | 0.808 | 0.839
1-9>>>10    0.415 | 0.099    | 0.415 | 0.598 | 0.653 | 0.708

fix item_title
            acc   | f1_macro | hit@1 | hit@3 | hit@5 | hit@10
1-6>>>7     0.428 | 0.059    | 0.428 | 0.533 | 0.567 | 0.617
1-7>>>8 


(base) szy@NJL5CG35065JF:~/bmw/BMW-item$ uv run src/predict.py 

[预测结果] 样本 #13799

Text: DSCI故障 老师您好:
1.车辆提示：小心驾驶 制动系统故障。
2.电脑检测存有故障：DSC 单元：内部，线性制动器位置传感器，电气故障。
3.ABL提示删除故障，删除后故障再次出现ABL提示更换DSCI。
4.登录AIR检查车辆无召回信息.
5.烦请老师给与技术支持。
6.联系人：李旭 13078884333 TSARA, CN_ G07 LCI_ Brake assistant power  ...
True label: 35645  (in_train=True) | item_title: TSARA, CN_ G07 LCI_ Brake assistant power changed with linear actuator DTC in IB_G70_4808BC, 4808B9, 480A06, 480A7D

Top-10 predictions:
35645           0.1598 ✅
44710           0.0006
45163           0.0005
32030           0.0005
34375           0.0005
16161           0.0005
44447           0.0005
45986           0.0005
45861           0.0004
45801           0.0004

Not-in-train probability: 0.0048 (prob_thr=0.500)
Final: 35645（Known）

(base) szy@NJL5CG35065JF:~/bmw/BMW-item$ uv run src/predict.py 

[预测结果] 样本 #10311

Text: 座椅皮子开裂 1.试车确认故障存在，主驾驶座椅座套外侧内部开裂。请见附件视频及照片； 
2.检查外观未见外力及人为损坏现象，无拆装痕迹。 
3.客户不接受；
请老师给予技术支持，谢谢

 TSARA_CN_G18 front seat cover crack ...
True label: 42062  (in_train=True) | item_title: TSARA_CN_G18 front seat cover crack

Top-10 predictions:
42062           0.0902 ✅
19919           0.0008
32030           0.0008
38988           0.0008
24743           0.0007
42118           0.0007
39587           0.0007
41522           0.0006
44450           0.0006
45163           0.0006

Not-in-train probability: 0.0072 (prob_thr=0.500)
Final: 42062（Known）
```
