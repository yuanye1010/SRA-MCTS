import re
from prompt.prompts import *
from models.model import *

def get_proposal(prompt, method='llama', temperature=0.7, max_tokens=512, seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=512):
    response = []
    response = local_inference_model(prompt, max_length=max_length, truncation=truncation, do_sample=do_sample,
                                        max_new_tokens=max_new_tokens, temperature=temperature)
    print(response)
    split_response = response.split('\n')
    return split_response


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
    def cot_prompt_wrap(x: str, lang: str = 'zh') -> str:
        print('\n', '==============================', 'proposal', '==============================', '\n')
        prompt = cot_prompt.format(question=x)
        print('propose_prompt: \n', prompt, '\n')
        return prompt
    
    @staticmethod
    def self_critic_prompt_wrap(x: str, y: str) -> str:
        print('\n', '==============================', 'self-critic', '==============================', '\n')
        if not y:
            y = 'None\n'
        critic_prompt = self_critic_prompt + x + '\nSolution:\n' + y + '\nScore:'
        return critic_prompt




class CoT_Task(SearchTask):
    def __init__(self, data, propose_method='glm', value_method='glm', temperature=0.7, max_tokens=2048, seed=170,
                 max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=1024, evaluate='', summary=False, lang='en', answer=None,
                 verify_method='string', do_self_critic=False):
        super().__init__(data, propose_method, value_method)
        self.mode = 'cot'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.evaluate = evaluate
        self.summary = summary
        self.lang = lang
        self.answer = answer
        self.verify_method = verify_method
        self.do_self_critic = do_self_critic

    def get_self_critic(self, solution):
        critic_prompt = self.self_critic_prompt_wrap(self.question, solution)
        output_score = get_proposal(critic_prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                    self.max_length, self.truncation, self.do_sample, 128)
        score_strs = ''
        for out in output_score:
            score_strs = score_strs + out + '\n'

        pattern = r'\d+'
        match = re.findall(pattern, score_strs)
        if not match:
            return None
        else:
            s = min(float(match[-1]), 10)
            s = max(s, 0)
            return s

    def run(self):
        prompt = self.cot_prompt_wrap(self.question, self.lang)
        out = get_proposal(prompt, self.propose_method, temperature=self.temperature,
                           max_tokens=self.max_tokens,
                           seed=self.seed, max_length=self.max_length, truncation=self.truncation,
                           do_sample=self.do_sample, max_new_tokens=self.max_new_tokens)
        solution = ''
        for _ in out:
            solution = solution + _ + '\n'
        solution = solution.strip()
        print(f'获得解答:{solution}\n')

        output = {'content': self.question, 'solution': solution}

        if self.do_self_critic:
            score = None
            cnt = 3
            while score is None and cnt:
                score = self.get_self_critic(solution)
                cnt -= 1
            if score is None:
                score = 0
            output.update({'self_critic': score})

        return output
