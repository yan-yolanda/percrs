import json
import re


def filter_movie_recommendation_json(input_file, output_file=None):
    """
    从包含多个 JSON 对象的 txt 文件中筛选出 goal 包含 "Movie recommendation" 的 JSON

    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径（可选，不指定则只返回结果）

    返回:
        筛选后的 JSON 对象列表
    """
    movie_recommendation_data = []

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # 逐行读取文件
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                try:
                    # 解析 JSON
                    data = json.loads(line)

                    # 检查 goal 字段是否包含 "Movie recommendation"
                    if "goal" in data:
                        goal_text = data["goal"]
                        # 使用正则表达式匹配 "Movie recommendation"，忽略大小写和括号
                        if re.search(r'Movie\s*recommendation', goal_text, re.IGNORECASE):
                            movie_recommendation_data.append(data)

                except json.JSONDecodeError as e:
                    print(f"JSON 解析错误（跳过该行）: {e}")
                    print(f"问题行内容: {line[:100]}...")  # 只打印前100个字符
                    continue

        print(f"总共找到 {len(movie_recommendation_data)} 个包含 'Movie recommendation' 的 JSON 对象")

        # 如果指定了输出文件，将结果写入
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for data in movie_recommendation_data:
                    json.dump(data, f, ensure_ascii=False)
                    f.write('\n')
            print(f"结果已保存到: {output_file}")

        return movie_recommendation_data

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return []
    except Exception as e:
        print(f"发生错误: {e}")
        return []


# 使用方法示例
if __name__ == "__main__":
    # 设置输入输出文件路径
    input_txt_file = "/data/yantingting/crs/PerCRS/data/en_dev.txt"  # 替换为你的输入文件路径
    output_txt_file = "/data/yantingting/crs/PerCRS/data/movie_recommendation_data.txt"  # 输出文件路径（可选）

    # 执行筛选
    filtered_data = filter_movie_recommendation_json(
        input_file=input_txt_file,
        output_file=output_txt_file  # 如果不需要保存文件，可以设置为 None
    )

    # 打印一些统计信息
    if filtered_data:
        # print("\n=== 筛选结果示例（第一个符合条件的 JSON）===")
        # print(json.dumps(filtered_data[0], indent=2, ensure_ascii=False))

        # 统计每个符合条件的 goal 字段
        print("\n=== 符合条件的 goal 字段 ===")
        for i, data in enumerate(filtered_data, 1):
            goal_text = data.get("goal", "")
            # 提取简短的 goal 描述
            short_goal = goal_text[:100] + "..." if len(goal_text) > 100 else goal_text
            print(f"{i}. {short_goal}")