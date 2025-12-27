import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import openai  # 或其他LLM API
from enum import Enum
from openai import OpenAI
import random

client = OpenAI(
        api_key="sk-zYW5O9J45Cq0dXHjipJlYJHKz63GiSUOGqsoxETlnFyRrsqk",  # 使用环境变量
        base_url="https://yunwu.ai/v1"
)


class PersonalityPolarity(Enum):
    """人格极性枚举"""
    POSITIVE = "+"
    NEGATIVE = "-"


@dataclass
class PersonalityProfile:
    """人格特质配置"""
    openness: PersonalityPolarity
    conscientiousness: PersonalityPolarity
    extraversion: PersonalityPolarity
    agreeableness: PersonalityPolarity
    neuroticism: PersonalityPolarity

    @classmethod
    def random(cls):
        """生成随机人格特征"""
        random.seed()
        return cls(
            openness=random.choice(list(PersonalityPolarity)),
            conscientiousness=random.choice(list(PersonalityPolarity)),
            extraversion=random.choice(list(PersonalityPolarity)),
            agreeableness=random.choice(list(PersonalityPolarity)),
            neuroticism=random.choice(list(PersonalityPolarity))
        )

    def to_vector(self) -> List[int]:
        """转换为向量表示 [-1, +1]"""
        return [
            1 if self.openness == PersonalityPolarity.POSITIVE else -1,
            1 if self.conscientiousness == PersonalityPolarity.POSITIVE else -1,
            1 if self.extraversion == PersonalityPolarity.POSITIVE else -1,
            1 if self.agreeableness == PersonalityPolarity.POSITIVE else -1,
            1 if self.neuroticism == PersonalityPolarity.POSITIVE else -1
        ]

    def get_description(self) -> str:
        """获取人格特质的自然语言描述"""
        descriptions = []

        # 开放性
        if self.openness == PersonalityPolarity.POSITIVE:
            descriptions.append(
                "highly open to new experiences, curious about unfamiliar topics, and enjoy deep conversations")
        else:
            descriptions.append("prefer familiar content, resistant to change, and lack curiosity")

        # 尽责性
        if self.conscientiousness == PersonalityPolarity.POSITIVE:
            descriptions.append("goal-oriented, organized, thoughtful, and provide useful feedback")
        else:
            descriptions.append("lack focus, easily distracted, and rarely provide detailed feedback")

        # 外向性
        if self.extraversion == PersonalityPolarity.POSITIVE:
            descriptions.append(
                "extroverted, actively participate in conversations, enjoy engagement, and interested in communication")
        else:
            descriptions.append(
                "introverted, avoid social interactions, hesitant to express yourself, and uninterested in socializing")

        # 宜人性
        if self.agreeableness == PersonalityPolarity.POSITIVE:
            descriptions.append("very agreeable, empathetic, cooperative, trusting, polite, and appreciative")
        else:
            descriptions.append("indifferent to others, uncooperative, and sometimes use rude language")

        # 神经质
        if self.neuroticism == PersonalityPolarity.POSITIVE:
            descriptions.append("emotionally fluctuating, lack confidence, and easily discouraged")
        else:
            descriptions.append("emotionally stable, confident in responses, and handle challenges well")

        return "You are " + "; ".join(descriptions) + "."


@dataclass
class UserProfile:
    """用户基本信息"""
    name: str
    gender: str
    age_range: str
    residence: str
    liked_movies: List[str]
    liked_celebrities: List[str]
    disliked_movies: List[str]
    query: str

class UserAgent:
    """基于LLM的人格化用户代理"""

    def __init__(
            self,
            user_profile: UserProfile,
            max_response_length: int = 50  # 最大回复长度（词数）
    ):
        self.user_profile = user_profile
        self.personality_profile = PersonalityProfile.random()
        self.max_response_length = max_response_length


        # 构建系统提示词
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """构建系统提示词（角色设定）"""
        # 用户基本信息
        profile_info = f"""You are {self.user_profile.name}, a {self.user_profile.gender} 
        in the age range of {self.user_profile.age_range}, living in {self.user_profile.residence}. 
        You enjoy movies like {', '.join(self.user_profile.liked_movies[:3])} 
        and celebrities like {', '.join(self.user_profile.liked_celebrities[:3])}, 
        but dislike movies such as {', '.join(self.user_profile.disliked_movies[:3])}."""

        # 人格特质描述
        personality_desc = self.personality_profile.get_description()

        # 行为规则
        behavior_rules = """You must follow these instructions during the conversation:
        1. Pretend you have limited knowledge about the recommended movies, and the only information source is the recommender.
        2. You don't need to introduce yourself or recommend anything, but feel free to share personal interests and reflect on your personality.
        3. When mentioning movie titles, put them in quotation marks (e.g., "Inception").
        4. You may end the conversation if you're satisfied with the recommendation or lose interest (e.g., by saying "thank you" or "no more questions").
        5. Keep your responses brief, ideally within 20 words. Be natural and conversational.
        6. Respond based on your personality traits described above."""

        # 组合完整提示词
        system_prompt = f"""{profile_info}

Your personality: {personality_desc}

{behavior_rules}

Now, let's start the conversation."""

        return system_prompt


    def generate_response(self, system_message: str) -> str:
        """
        生成用户回复

        Args:
            system_message: 系统（推荐系统）的消息

        Returns:
            用户的回复文本
        """
        # 构建消息列表
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "assistant", "content": system_message},
        ]

        # 调用LLM生成回复
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        user_response = response.choices[0].message.content.strip()

        return user_response


    def is_conversation_ended(self, user_response: str) -> bool:
        """判断用户是否想结束对话"""
        end_phrases = [
            "thank you", "thanks", "no more questions", "that's all",
            "goodbye", "bye", "see you", "I'm done", "no thanks",
            "not interested", "I'll stop here"
        ]

        response_lower = user_response.lower()
        for phrase in end_phrases:
            if phrase in response_lower:
                return True
        return False

    def get_conversation_summary(self) -> Dict:
        """获取对话摘要，用于分析"""
        return {
            "user_profile": {
                "name": self.user_profile.name,
                "personality": self.personality_profile.to_vector(),
                "personality_description": self.personality_profile.get_description()
            }
        }


# ==================== 批量生成用户代理 ====================

class UserAgentFactory:
    """用户代理工厂，用于批量创建不同人格的用户"""

    @staticmethod
    def create_user_from_dataset(
            dataset_entry: Dict,
            personality_vector: List[int]
    ) -> UserAgent:
        """从数据集条目创建用户代理"""
        # 这里可以根据具体数据集格式进行解析
        pass

