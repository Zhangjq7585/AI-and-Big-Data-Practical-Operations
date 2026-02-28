（二）自然语言处理任务
目标：对一批文本评论进行情感分析，判断其正负面倾向，并提取关键主题词。
任务：
1．文本清洗与分词；
2．构建词向量模型；
3．训练情感分类模型；
4．输出情感分布可视化图表；
5．提取高频关键词与主题。

IMDB数据集文件夹存放路径：
E:\Download\aclImdb

代码运行结果：
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 200, 100)          2000000   
                                                                 
 bidirectional (Bidirectiona  (None, 128)              84480     
 l)                                                              
                                                                 
 dense (Dense)               (None, 64)                8256      
                                                                 
 dropout (Dropout)           (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 1)                 65        
                                                                 
=================================================================
Total params: 2,092,801
Trainable params: 2,092,801
Non-trainable params: 0
_________________________________________________________________

Epoch 4/10
157/157 [==============================] - 8s 53ms/step - loss: 0.1177 - accuracy: 0.9611 - val_loss: 0.7158 - val_accuracy: 0.7854
Epoch 5/10
157/157 [==============================] - 8s 53ms/step - loss: 0.0722 - accuracy: 0.9765 - val_loss: 0.8795 - val_accuracy: 0.7484
测试集准确率：0.8337
782/782 [==============================] - 13s 16ms/step

正面评论高频词：
the: 168983
<OOV>: 112597
and: 88541
a: 82778
of: 76497
to: 66320
is: 57003
in: 49290
it: 38052
i: 34099
that: 33975
this: 33711
as: 25850
with: 23055
for: 22151
was: 21824
but: 20034
film: 19606
movie: 18144
his: 17126

负面评论高频词：
the: 159145
<OOV>: 107452
a: 78536
and: 73035
of: 68669
to: 68502
is: 49799
in: 42897
this: 39475
i: 38378
it: 38261
that: 35224
was: 26164
movie: 23682
for: 21550
but: 20965
with: 20669
as: 20209
film: 17883
on: 16760

正面评论TF-IDF主题词：
film: 0.0752
movie: 0.0731
great: 0.0386
like: 0.0367
good: 0.0364
story: 0.0344
just: 0.0310
time: 0.0299
really: 0.0285
love: 0.0281
best: 0.0280
people: 0.0248
films: 0.0246
life: 0.0239
movies: 0.0233
think: 0.0223
watch: 0.0221
way: 0.0219
seen: 0.0219
characters: 0.0214

负面评论TF-IDF主题词：
movie: 0.0915
film: 0.0682
just: 0.0457
like: 0.0454
bad: 0.0444
good: 0.0350
really: 0.0324
time: 0.0297
dont: 0.0296
story: 0.0267
plot: 0.0266
acting: 0.0264
movies: 0.0263
make: 0.0261
people: 0.0259
watch: 0.0227
worst: 0.0225
better: 0.0221
think: 0.0221
characters: 0.0220
