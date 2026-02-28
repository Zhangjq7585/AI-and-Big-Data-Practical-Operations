"""
纯Python版本日志分析 - 使用Pandas替代Spark
不需要Java环境，适合本地学习和快速原型开发
"""
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os

# 1. 生成模拟日志数据（与Spark版本相同）
def generate_log_data(num_records=1000, output_file="sample_logs.json"):
    """生成模拟的日志数据并保存为JSON文件"""

    # 定义可能的事件类型
    event_types = ["click", "view", "purchase", "login", "logout", "error",
                   "search", "add_to_cart", "remove_from_cart", "share"]
    # 定义用户ID范围
    user_ids = list(range(1001, 2001))  # 1000个用户
    # 定义IP地址段
    ip_prefixes = ["192.168.1", "10.0.0", "172.16.0", "192.168.2", "10.0.1"]
    # 生成日志数据
    logs = []
    start_time = datetime(2026, 1, 1)
    print(f"正在生成 {num_records} 条模拟日志...")

    for i in range(num_records):
        # 随机生成时间戳（在30天范围内）
        time_offset = timedelta(
            days=random.randint(0, 29),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        # 构建日志条目
        log_entry = {
            "event": random.choice(event_types),
            "timestamp": (start_time + time_offset).isoformat(),
            "user_id": random.choice(user_ids),
            "ip": f"{random.choice(ip_prefixes)}.{random.randint(1, 255)}",
            "value": round(random.uniform(0.1, 100.0), 2) if random.random() > 0.3 else None,  # 30%可能为null
            "session_id": f"session_{random.randint(10000, 99999)}"
        }
        logs.append(log_entry)

    # 保存为JSON文件，每行一个JSON对象（JSON Lines格式）
    with open(output_file, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"✓ 模拟日志已生成并保存到: {output_file}")
    return output_file


# 2. 读取JSON文件并转换为Pandas DataFrame
def load_log_data(file_path="sample_logs.json"):
    """读取JSON日志文件并返回Pandas DataFrame"""

    print(f"正在读取日志文件: {file_path}")

    # 读取JSON Lines文件
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # 转换为Pandas DataFrame
    df = pd.DataFrame(data)

    # 将timestamp列转换为datetime类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"✓ 成功加载 {len(df)} 条日志记录")
    return df


# 3. 数据探索和清洗
def explore_data(df):
    """初步探索数据的基本信息"""
    print("\n" + "= " * 50)
    print("数据探索")
    print("= " * 50)

    # 查看前几行
    print("\n前5条日志记录:")
    print(df.head())

    # 数据基本信息
    print("\n数据基本信息:")
    print(f"总记录数: {len(df)}")
    print(f"列名: {list(df.columns)}")

    # 数据类型
    print("\n数据类型:")
    print(df.dtypes)

    # 缺失值统计
    print("\n缺失值统计:")
    print(df.isnull().sum())

    # 基本统计信息
    print("\n数值列统计信息:")
    print(df.describe())


# 4. 分组聚合分析
def analyze_events(df):
    """分析事件频率"""

    print("\n" + "= " * 50)
    print("事件频率分析")
    print("= " * 50)

    # 按事件类型分组计数
    event_counts = df['event'].value_counts().reset_index()
    event_counts.columns = ['event', 'count']

    # 按计数降序排序
    event_counts = event_counts.sort_values('count', ascending=False)
    print("\n各事件类型频率:")
    print(event_counts.to_string(index=False))

    return event_counts


# 5. 高级分析
def advanced_analysis(df):
    """进行更深入的分析"""
    print("\n" + "= " * 50)
    print("高级分析")
    print("= " * 50)

    # 按小时分析事件分布
    df['hour'] = df['timestamp'].dt.hour
    hourly_stats = df.groupby(['hour', 'event']).size().unstack(fill_value=0)

    print("\n每小时事件分布（前5行）:")
    print(hourly_stats.head())

    # 用户活跃度分析
    user_activity = df.groupby('user_id').agg({
        'event': 'count',
        'value': 'mean'
    }).rename(columns={'event': 'total_actions', 'value': 'avg_value'})

    print("\n用户活跃度Top 10:")
    print(user_activity.nlargest(10, 'total_actions'))

    # 计算事件的百分比分布
    event_percentages = (df['event'].value_counts(normalize=True) * 100).round(2)
    print("\n事件类型百分比分布:")
    print(event_percentages)

    return hourly_stats, user_activity


# 6. 数据可视化
def create_visualizations(event_counts, hourly_stats, df):
    """创建各种可视化图表"""
    print("\n" + "= " * 50)
    print("生成可视化图表")
    print("= " * 50)

    # 设置中文显示（如果需要）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 事件频率柱状图
    ax1 = axes[0, 0]
    bars = ax1.bar(event_counts['event'], event_counts['count'], color='skyblue')
    ax1.set_xlabel('事件类型')
    ax1.set_ylabel('出现次数')
    ax1.set_title('事件频率分布')
    ax1.tick_params(axis='x', rotation=45)
    # 在柱子上添加数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}', ha='center', va='bottom')

    # 2. 事件占比饼图
    ax2 = axes[0, 1]
    # 只显示前6个事件，其他合并为"其他"
    top_events = event_counts.head(6)
    others_count = event_counts.iloc[6:]['count'].sum() if len(event_counts) > 6 else 0
    if others_count > 0:
        plot_data = pd.concat([top_events, pd.DataFrame([{'event': '其他', 'count': others_count}])])
    else:
        plot_data = top_events

    ax2.pie(plot_data['count'], labels=plot_data['event'], autopct='%1.1f%%')
    ax2.set_title('事件类型占比')

    # 3. 小时活动热力图
    ax3 = axes[1, 0]
    # 计算每小时的事件数量
    hourly_total = df.groupby('hour').size()
    ax3.bar(hourly_total.index, hourly_total.values, color='lightcoral')
    ax3.set_xlabel('小时')
    ax3.set_ylabel('事件数量')
    ax3.set_title('每小时事件分布')
    ax3.set_xticks(range(0, 24, 2))

    # 4. 事件随时间变化趋势
    ax4 = axes[1, 1]
    # 按天聚合
    daily_counts = df.groupby(df['timestamp'].dt.date).size()
    ax4.plot(range(len(daily_counts)), daily_counts.values, marker='o', linestyle='-', color='green')
    ax4.set_xlabel('天数')
    ax4.set_ylabel('事件数量')
    ax4.set_title('每日事件趋势')

    plt.tight_layout()

    # 保存图表
    chart_file = "log_analysis_results.png"
    plt.savefig(chart_file, dpi=100, bbox_inches='tight')
    print(f"✓ 可视化图表已保存到: {chart_file}")

    # 如果是在Jupyter Notebook或支持GUI的环境，可以显示
    plt.show()
    return fig

# 7. 保存分析结果
def save_results(df,event_counts, hourly_stats, output_dir="analysis_output"):
    """将分析结果保存为CSV文件"""

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存事件频率统计
    event_counts.to_csv(f"{output_dir}/event_frequencies.csv", index=False)
    print(f"✓ 事件频率统计已保存到: {output_dir}/event_frequencies.csv")

    # 保存小时统计
    hourly_stats.to_csv(f"{output_dir}/hourly_distribution.csv")
    print(f"✓ 小时分布统计已保存到: {output_dir}/hourly_distribution.csv")

    # 生成摘要报告
    with open(f"{output_dir}/analysis_report.txt", "w", encoding="utf-8") as f:
        f.write("日志分析报告\n")
        f.write("= " * 50 + "\n\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总日志记录数: {len(df)}\n")
        f.write(f"唯一事件类型数: {len(event_counts)}\n")
        f.write(f"唯一用户数: {df['user_id'].nunique()}\n")
        f.write(f"时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}\n\n")

        f.write("事件频率排名:\n")
        for idx, row in event_counts.iterrows():
            f.write(f"  {row['event']}: {row['count']} 次\n")

    print(f"✓ 分析报告已保存到: {output_dir}/analysis_report.txt")


# 8. 主函数 - 整合所有步骤
def main():
    """主函数：执行完整的分析流程"""
    print("\n" + "= " * 60)
    print("纯Python日志分析系统 - 无需Spark/Java")
    print("= " * 60)

    # 步骤1: 生成模拟数据
    data_file = generate_log_data(num_records=2000)  # 生成2000条日志

    # 步骤2: 加载数据
    df = load_log_data(data_file)

    # 步骤3: 探索数据
    explore_data(df)

    # 步骤4: 基本分析
    event_counts = analyze_events(df)

    # 步骤5: 高级分析
    hourly_stats, user_activity = advanced_analysis(df)

    # 步骤6: 可视化
    create_visualizations(event_counts, hourly_stats, df)

    # 步骤7: 保存结果
    save_results(df,event_counts, hourly_stats)

    print("\n" + "= " * 60)
    print("✓ 分析完成！所有结果已保存。")
    print("= " * 60)

# 运行主程序
if __name__ == "__main__":
    main()
