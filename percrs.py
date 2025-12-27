from UserAgent import UserAgent
from CHATCRS import CHATCRS
import json
from UserAgent import UserProfile
import os


def read_jsonl_file(filename):
    """
    读取JSONL格式文件（每行一个JSON对象）

    Args:
        filename: 文件名

    Returns:
        list: 包含所有JSON对象的列表
    """
    json_objects = []

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        json_obj = json.loads(line)
                        if not "Greetings" in json_obj["goal"]:
                            json_objects.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"第 {line_num} 行JSON解析错误: {e}")
                        print(f"问题内容前100字符: {line[:100]}")

    except FileNotFoundError:
        print(f"文件 {filename} 不存在")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

    return json_objects


def extract_specific_fields(json_objects):
    """
    从每个JSON对象中提取指定字段

    Args:
        json_objects: JSON对象列表

    Returns:
        list: 提取的数据列表
    """
    extracted_data = []

    for i, obj in enumerate(json_objects):
        # 提取需要的字段
        data = {
            '序号': i + 1,
            'goal': obj.get('goal', ''),
            '用户名': obj.get('user_profile', {}).get('Name', ''),
            '年龄范围': obj.get('user_profile', {}).get('Age Range', ''),
            '性别': obj.get('user_profile', {}).get('Gender', ''),
            '职业': obj.get('user_profile', {}).get('Occupation', ''),
            '对话轮数': len(obj.get('conversation', [])),
            '情境': obj.get('situation', '')
        }
        extracted_data.append(data)

    return extracted_data

def simulate_with_chatcrs(user_profile: UserProfile):
    """使用ChatCRS进行模拟对话"""

    print("创建ChatCRS...")
    chatcrs = CHATCRS(
        seed=42,  # 设置随机种子
        debug=False,  # 调试模式
        kg_dataset="opendialkg"  # 替换为您的知识图谱数据集名称
    )

    # 创建LLM-US模拟器
    print("创建UserAgent模拟器...")
    useragent = UserAgent(user_profile)

    print("开始对话模拟")

    conversation_history = []
    max_turns = 10

    # 第一轮：用户开始对话
    user_message = user_profile.query
    conversation_history.append(("user", user_message))
    print(f"USER: {user_message}")

    for turn in range(max_turns):
        print(f"\n第 {turn + 1} 轮")

        # 构建对话字典格式（符合CHATGPT类的输入格式）
        conv_dict = {
            "context": [msg for _, msg in conversation_history if _ == "user" or _ == "system"],
            "rec": []  # 推荐列表，初始为空
        }

        # CHATGPT 生成回复
        try:
            # 使用get_conv方法生成对话回复
            gen_inputs, system_reply = chatcrs.get_conv(conv_dict)

            if not system_reply:
                print("系统返回空回复")
                break

            conversation_history.append(("system", system_reply))
            print(f"SYSTEM: {system_reply}")

        except Exception as e:
            print(f"CHATCRS生成回复错误: {e}")
            break

        # 用户模拟器回复
        user_message = useragent.generate_response(system_reply)
        conversation_history.append(("user", user_message))
        print(f"USER: {user_message}")

        # 检查用户是否终止对话
        if useragent.is_conversation_ended(user_message):
            print(f"用户终止对话")
            break

    print(f"\n对话结束，共 {len(conversation_history)} 轮")

    return conversation_history, useragent.get_conversation_summary()


if __name__ == '__main__':
    file_name = "/data/yantingting/crs/PerCRS/data/movie_recommendation_data.txt"
    json_objects = read_jsonl_file(file_name)
    sample_num = 100
    for i in range(sample_num):
        object = json_objects[i]
        user_profile = UserProfile(
            name=object["user_profile"]["Name"],
            gender=object["user_profile"]["Gender"],
            age_range=object["user_profile"]["Age Range"],
            residence=object["user_profile"]["Residence"],
            liked_movies=object["user_profile"]["Accepted movies"],
            liked_celebrities=object["user_profile"]["Accepted celebrities"],
            disliked_movies=object["user_profile"]["Rejected movies"],
            query=object["conversation"][0].split('] ', 1)[1]
        )
        conversation_history, conversation_summary = simulate_with_chatcrs(user_profile)

        #保存对话数据和user信息
        result_data = {
            "user_profile": {
            "name": user_profile.name,
            "gender": user_profile.gender,
            "age_range": user_profile.age_range,
            "residence": user_profile.residence,
            "liked_movies": user_profile.liked_movies,
            "liked_celebrities": user_profile.liked_celebrities,
            "disliked_movies": user_profile.disliked_movies,
            "query": user_profile.query
        },
            "conversation_summary": conversation_summary,
            "conversation_history": conversation_history
        }

        output_dir = "/data/yantingting/crs/PerCRS/data/output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"simulation_{i + 1:03d}_{user_profile.name}.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        print(f"✓ 结果已保存到: {output_file}")


