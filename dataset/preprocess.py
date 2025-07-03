import pandas as pd
# 明确列名（如 test.csv 无表头）
column_names = [
    'DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage',
    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
    'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
]

def clean_csv(input_path, output_path, has_header=True):
    if has_header:
        df = pd.read_csv(
            input_path,
            parse_dates=['DateTime'],
            na_values=['?', ' ', '', 'NA']  # 明确将这些值作为 NaN
        )
    else:
        df = pd.read_csv(
            input_path,
            header=None,
            names=column_names,
            parse_dates=['DateTime'],
            na_values=['?', ' ', '', 'NA']
        )

    # 删除任意一列含 NaN 的整行
    df.dropna(inplace=True)

    # 确保输出路径存在

    df.to_csv(output_path, index=False)
    print(f"[✓] Cleaned data saved to: {output_path}")
    print(f"     Rows kept: {len(df)}")

# 使用方法
clean_csv('train.csv', 'train_cleaned.csv', has_header=True)
clean_csv('test.csv', 'test_cleaned.csv', has_header=False)  # 若 test.csv 无表头
