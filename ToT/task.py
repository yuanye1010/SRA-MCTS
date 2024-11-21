import random
import re
from ToT.base import Node
from models.model import *
from ToT.bfs import BFS
from ToT.dfs import DFS
from prompt.prompts import *


def get_proposal(prompt, method='llama', temperature=0.7, max_tokens=512, seed=170, max_length=1024, truncation=True,
                 do_sample=True, max_new_tokens=512):
    response = []
    response = local_inference_model(prompt, max_length=max_length, truncation=truncation, do_sample=do_sample,
                                        max_new_tokens=max_new_tokens, temperature=temperature)
    # print(response)
    return response

def get_value(prompt_answer, method='llama', temperature=0.7, max_tokens=128, seed=170, max_length=1024, low=0, high=10):
    cnt = 2
    value = 0
    while cnt:
        try:
            value = local_value_model(prompt_answer, max_length=max_length)
            break
        except Exception as e:
            print(f'获取<{method}>分数失败!\n错误类型:{e}\n')
            cnt -= 1
    return value


class SearchTask(object):
    def __init__(self, data, propose_method='glm', value_method='glm'):
        super().__init__()
        self.question = data
        self.propose_method = propose_method
        self.value_method = value_method
        self.value_cache = {}

    def clear_cache(self):
        self.value_cache = {}

    @staticmethod
    def single_propose_prompt_wrap(x: str, y: str = '', step: int = 0) -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\nExisting Steps:\n' + y + 'Based on the mentioned steps, possible next step is :\n')
        prompt = code_proposal_prompt_en + x + '\nExisting Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        print('\n', '==============================', 'critic', '==============================', '\n')
        value_prompt = critic_en + x + '\nExisting Steps:\n' + y.strip() + '\nOutput:'
        return value_prompt

    @staticmethod
    def value_outputs_unwrap(value_outputs: list, low=0.0, high=1.0) -> float:
        try:
            print('-'*40 + 'In value_outputs_unwrap' + '-'*40)
            print(value_outputs)
            out_value = 5
            
            match = re.search('[Ss]core[:：]\s*(\d+)', value_outputs, re.DOTALL)
            if match:
                out_value = float(match.group(1))

            print(out_value)
            out_value = min(max(low, out_value), high)
        except Exception as e:
            print(f'分数输出有误！错误类型:{e}\n')
            return low
        return out_value


class ToT_Task(SearchTask):
    def __init__(self, data, propose_method='glm', value_method='glm', algorithm='dfs', branch=3, select_branch=1,
                 max_depth=4, end_gate=8.8, select_method='greedy',
                 temperature=0.7, max_tokens=1024,
                 seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=256, use_case_prompt=False, low=0, high=10, evaluate='', multiply_value=False, lang='en', answer=None, verify_method='string'):
        super().__init__(data, propose_method, value_method)
        assert 0 <= low < high, "Inappropriate value range!"
        self.mode = 'tot'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.algorithm = algorithm
        self.branch = branch
        self.select_branch = select_branch
        self.max_depth = max_depth
        self.use_case_prompt = use_case_prompt
        self.low = low
        self.high = high
        self.evaluate = evaluate
        self.select_method = select_method
        self.end_gate = end_gate
        self.node_count = 1
        self.multiply_value = multiply_value
        self.lang = lang
        self.answer = answer
        self.verify_method = verify_method

    def update_count(self):
        self.node_count += 1

    def clear_cache(self):
        self.value_cache = {}
        self.node_count = 1


    def extract_proposal(self, p, step_n, y):
        print(p)
        p = re.sub(r'```.*?```', '', p, flags=re.DOTALL)
        if "Next step:" in p or 'Next Step' in p:
            # stp = p.split('Next step:')[1].strip()
            match = re.search(r'Next [Ss]teps?:\s*(.*)', p)
            p = match.group(1).strip()

        elif "Step" in p and ":" in p:
            match = re.search(r'Step \d+:\s*(.*)', p)
            if match:
                # 提取 ":" 后面的内容
                p = match.group(1).strip()
            p = re.sub(r'Step \d+:\s*', '', p).strip()
            
        if len(p) < 2:
            print('输出步骤过短！\n')
            return ''
        if p in y:
            print('输出步骤重复！\n')
            return ''

        pattern = r'\d\.\s*(.*)'
        match = re.findall(pattern, p)
        if match:
            p = re.sub(r'\d\.\s*(.*)', '', p, flags=re.DOTALL).strip() + '\n'
        for _ in match:
            p = p +  _ + '\n'

        revised_ = 'Step ' + str(step_n) + ': ' + p
        print(f'标准化后新的步骤:{revised_}\n')
        return revised_ + '\n'

    def get_next_step(self, y, step_n):
        prompt = self.single_propose_prompt_wrap(self.question, y, step_n)
        

        response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, self.max_new_tokens)
        if not response:
            print('获得下一步失败！\n')
            return ''

        return self.extract_proposal(response, step_n, y)

    def get_step_value(self, y):
        if y in self.value_cache.keys():
            return self.value_cache[y]

        prompt = self.value_prompt_wrap(self.question, y)
        response = get_value(prompt, self.value_method, self.temperature, 128, self.seed,
                                self.max_length, self.low, self.high)
        value = self.value_outputs_unwrap(response, self.low, self.high)
        print(f'获得评分:{value}\n')
        self.value_cache.update({y: value})
        return value

    def run(self):
        self.clear_cache()
        if self.algorithm == 'dfs':
            solution, root, final_node = DFS(self)
        elif self.algorithm == 'bfs':
            solution, root, final_node = BFS(self)
        else:
            print('Unsupported algorithm!\n')
            return {}

        final_answer = {'content': self.question, 'solution': solution}

        if self.multiply_value:
            multiply_v = final_node.get_multiply_value()
            final_answer.update({'multiply_value': multiply_v})

        return final_answer, root
