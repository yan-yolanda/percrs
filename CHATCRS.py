import typing
import torch
import json
import os

import nltk
import openai
import tiktoken
import numpy as np

from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from accelerate.utils import set_seed
from thefuzz import fuzz
from tqdm import tqdm

# iEvaLM中的ChatCRS

client = openai.OpenAI(
    api_key="sk-zYW5O9J45Cq0dXHjipJlYJHKz63GiSUOGqsoxETlnFyRrsqk",
    base_url="https://yunwu.ai/v1"
)


def annotate_chat(messages, logit_bias=None):
    """简化的Chat API调用"""
    if logit_bias is None:
        logit_bias = {}


    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0,
            logit_bias=logit_bias,
            timeout=30
        )
        content = response.choices[0].message.content
        # print(f"Chat API调用成功，回复长度: {len(content)}")
        return content
    except Exception as e:
        print(f"Chat API调用失败: {e}")
        return "我可以帮您推荐电影。您能告诉我您喜欢什么类型的电影吗？或者您有特定的演员、导演或年份偏好吗？"


def annotate(conv_str):
    """简化的Embedding API调用"""
    # print(f"调用Embedding API，文本长度: {len(conv_str)}")

    try:
        response = client.embeddings.create(
            model='text-embedding-ada-002',
            input=conv_str,
            timeout=30
        )
        print("Embedding API调用成功")
        return response
    except Exception as e:
        print(f"Embedding API调用失败: {e}")

        # 返回一个空的嵌入作为备用
        class MockData:
            embedding = [0.0] * 1536

        class MockResponse:
            data = [MockData()]

        return MockResponse()


class CHATCRS():

    def __init__(self, seed, debug, kg_dataset) -> None:
        self.seed = seed
        self.debug = debug
        if self.seed is not None:
            set_seed(self.seed)

        self.kg_dataset = kg_dataset

        self.kg_dataset_path = f"src/data/{self.kg_dataset}"
        with open(f"{self.kg_dataset_path}/entity2id.json", 'r', encoding="utf-8") as f:
            self.entity2id = json.load(f)
        with open(f"{self.kg_dataset_path}/id2info.json", 'r', encoding="utf-8") as f:
            self.id2info = json.load(f)

        self.id2entityid = {}
        for id, info in self.id2info.items():
            if info['name'] in self.entity2id:
                self.id2entityid[id] = self.entity2id[info['name']]

        self.item_embedding_path = f"src/save/embed/item/{self.kg_dataset}"

        item_emb_list = []
        id2item_id = []
        if os.path.exists(self.item_embedding_path):
            for i, file in tqdm(enumerate(os.listdir(self.item_embedding_path))):
                item_id = os.path.splitext(file)[0]
                if item_id in self.id2entityid:
                    id2item_id.append(item_id)
                    with open(f'{self.item_embedding_path}/{file}', encoding='utf-8') as f:
                        embed = json.load(f)
                        item_emb_list.append(embed)
        else:
            print(f"警告: 嵌入路径不存在 {self.item_embedding_path}")

        self.id2item_id_arr = np.asarray(id2item_id)
        self.item_emb_arr = np.asarray(item_emb_list)

        self.chat_recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation. The recommendation list must contain 10 items that are consistent with user preference. The recommendation list can contain items that the dialog mentioned before. The format of the recommendation list is: no. title. Don't mention anything other than the title of items in your recommendation list.'''

    def get_rec(self, conv_dict):

        rec_labels = [self.entity2id[rec] for rec in conv_dict['rec'] if rec in self.entity2id]

        context = conv_dict['context']
        context_list = []

        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
            else:
                role_str = 'assistant'
            context_list.append({
                'role': role_str,
                'content': text
            })

        conv_str = ""

        for context in context_list[-2:]:
            conv_str += f"{context['role']}: {context['content']} "

        # 使用简化的annotate函数
        conv_embed_response = annotate(conv_str)
        conv_embed = conv_embed_response.data[0].embedding
        conv_embed = np.asarray(conv_embed).reshape(1, -1)

        # 如果没有嵌入数据，跳过相似度计算
        if len(self.item_emb_arr) == 0:
            print("没有项目嵌入数据，返回空推荐")
            return [[]], rec_labels

        sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
        rank_arr = np.argsort(sim_mat, axis=-1).tolist()
        rank_arr = np.flip(rank_arr, axis=-1)[:, :50]
        item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
        item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]

        return item_rank_arr, rec_labels

    def get_conv(self, conv_dict):

        context = conv_dict['context']
        context_list = []
        context_list.append({
            'role': 'system',
            'content': self.chat_recommender_instruction
        })

        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
            else:
                role_str = 'assistant'
            context_list.append({
                'role': role_str,
                'content': text
            })

        gen_inputs = None
        gen_str = annotate_chat(context_list)

        return gen_inputs, gen_str

    def get_choice(self, gen_inputs, options, state, conv_dict):

        updated_options = []
        for i, st in enumerate(state):
            if st >= 0:
                updated_options.append(options[i])

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        logit_bias = {encoding.encode(option)[0]: 10 for option in updated_options}

        context = conv_dict['context']
        context_list = []

        for i, text in enumerate(context[:-1]):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
            else:
                role_str = 'assistant'
            context_list.append({
                'role': role_str,
                'content': text
            })
        context_list.append({
            'role': 'user',
            'content': context[-1]
        })

        response_op = annotate_chat(context_list, logit_bias=logit_bias)
        return response_op[0]