sample4:大数据处理与分布式计算
	目标：使用Spark对大规模日志数据进行聚合分析，统计各类事件的频率与分布。
	任务：
		1．读取分布式存储数据；
		2．编写Spark SQL或DataFrame操作；
		3．进行分组聚合与排序；
		4．输出统计结果与可视化图表。

注意！！本教程使用纯Python的数据处理库，用Pandas代替Spark，不需要安装Java，分析模拟日志数据。
读者可以自行使用spark和java环境来分析。

以下是运行结果。
________________________________________________________
纯Python日志分析系统 - 无需Spark/Java
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
正在生成 2000 条模拟日志...
✓ 模拟日志已生成并保存到: sample_logs.json
正在读取日志文件: sample_logs.json
✓ 成功加载 2000 条日志记录
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
数据探索
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
前5条日志记录:
      event           timestamp  user_id            ip  value     session_id
0     error 2026-01-23 10:29:03     1877    10.0.1.126    NaN  session_54766
1     share 2026-01-08 03:34:01     1836  192.168.1.73  32.60  session_67755
2  purchase 2026-01-17 21:43:26     1224     10.0.0.20  51.83  session_72075
3     error 2026-01-22 04:22:43     1209    10.0.0.164  75.76  session_29462
4     login 2026-01-27 02:22:42     1701  192.168.2.98   2.11  session_20856
数据基本信息:
总记录数: 2000
列名: ['event', 'timestamp', 'user_id', 'ip', 'value', 'session_id']
数据类型:
event                 object
timestamp     datetime64[ns]
user_id                int64
ip                    object
value                float64
session_id            object
dtype: object
缺失值统计:
event           0
timestamp       0
user_id         0
ip              0
value         574
session_id      0
dtype: int64
数值列统计信息:
           user_id        value
count  2000.000000  1426.000000
mean   1499.435000    50.153471
std     288.737572    29.319464
min    1001.000000     0.110000
25%    1253.750000    24.787500
50%    1489.500000    49.470000
75%    1756.250000    75.757500
max    1998.000000    99.900000
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
事件频率分析
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
各事件类型频率:
           event  count
           error    214
        purchase    210
          search    210
          logout    207
remove_from_cart    206
           login    200
           share    196
            view    193
           click    188
     add_to_cart    176
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
高级分析
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
每小时事件分布（前5行）:
event  add_to_cart  click  error  login  ...  remove_from_cart  search  share  view
hour                                     ...                                       
0                5      9     11      4  ...                 8      19     11     3
1                7      6      6     11  ...                 8       8      3     6
2                8      6     11      6  ...                10       4      8     9
3                4      7      7      9  ...                 2      12      6     5
4                8      9      9      8  ...                 6       8     12     8
[5 rows x 10 columns]
用户活跃度Top 10:
         total_actions  avg_value
user_id                          
1825                 9  40.043333
1370                 7  35.045000
1378                 7  56.338000
1438                 7  74.657500
1459                 7        NaN
1418                 6  56.942500
1483                 6  52.520000
1561                 6  18.155000
1644                 6  37.790000
1665                 6  56.293333
事件类型百分比分布:
error               10.70
purchase            10.50
search              10.50
logout              10.35
remove_from_cart    10.30
login               10.00
share                9.80
view                 9.65
click                9.40
add_to_cart          8.80
Name: event, dtype: float64
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
生成可视化图表
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
✓ 可视化图表已保存到: log_analysis_results.png
✓ 事件频率统计已保存到: analysis_output/event_frequencies.csv
✓ 小时分布统计已保存到: analysis_output/hourly_distribution.csv
✓ 分析报告已保存到: analysis_output/analysis_report.txt
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
✓ 分析完成！所有结果已保存。
